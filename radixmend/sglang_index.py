"""Module B (first cut): SECDED on slot indices inside SGLang v0.5.22.

Code: extended Hamming over the 32 low bits of a slot index (6 check bits +
overall parity = 7 bits), the same code as ``radixmend.secded_index`` with
``data_bits=32``. Two structures are covered here, the two RQ1-5 found most
harmful (silent output changes from low-bit flips):

* radix tree ``node.value`` (int64): the code sits in the unused high bits of each
  word, bits 32..38, with a marker at bit 39 so encoded and plain indices can be
  told apart (a flip of the marker only makes the word read as plain). Words are
  encoded on ``insert``; every read of a node path (``collect_full_device_indices``,
  i.e. load-back / prefix / repoint) and of ``match_prefix`` checks and corrects the nodes in place and returns
  plain indices, and ``allocator.free`` checks and strips encoded words (eviction
  and duplicate frees). Codes are per element, so node splits keep them.
* ``req_to_token`` (int32): the word has no room, so the code is kept in a shadow
  int8 table of the same shape (a quarter of its size). Every write re-encodes
  what was written; the choke point is the kv_indices gather used by every
  attention call (``KVIndexTranslator.pack_read_stream``): a second kernel re-reads
  the same entries with their codes, corrects single-bit errors in ``kv_indices``
  and in the table, and counts what it cannot correct.

* the allocator free list ``free_pages`` / ``release_pages`` (int64): same encoding as
  ``node.value``; ``free`` encodes what it returns to the list, ``alloc`` checks and strips
  what it hands out, so a flip in the free list cannot hand out a slot another node owns.
* ``seq_lens`` (int64, GPU): the scheduler keeps an exact CPU mirror (``seq_lens_cpu``);
  after ``prepare_for_decode`` / ``prepare_for_extend`` and before ``ForwardBatch.init_new``
  the GPU copy is replaced by the mirror, and the forward never shares its tensor with the scheduler, and mismatches are counted.

Not covered yet: cross-recovery of uncorrectable entries (node.value <-> req_to_token).

Enable with ``RADIXMEND_INDEX=1``. Install before modules C and D so their
wrappers see plain indices.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from radixmend.secded_index import CORRECTED_CHECK, CORRECTED_DATA, DETECTED, IndexCode

LOW = 0xFFFFFFFF
SHIFT = 32
MARK = 1 << 39
R = 6  # Hamming check bits for 32 data bits

_state = {"dev": None, "err": None, "r2t_pool": None, "step": 0, "seen": {}}
# err: [0] node / free-list corrected, [1] node / free-list uncorrectable, [2] req_to_token corrected,
#      [3] req_to_token uncorrectable, [4] node / [5] req_to_token check-bit-only errors fixed,
#      [6] seq_lens entries restored from the CPU mirror


def _tables(device):
    if _state["dev"] != str(device):
        code = IndexCode(data_bits=32, word_bits=64)
        assert code.r == R
        _state.update(
            dev=str(device), code=code,
            masks=torch.tensor(code.col_masks, dtype=torch.int64, device=device),
            lut_s=torch.tensor(code.lut_status.astype("int32"), dtype=torch.int32, device=device),
            lut_b=torch.tensor(code.lut_bit, dtype=torch.int64, device=device),
            err=torch.zeros(7, dtype=torch.int32, device=device),
        )
    return _state


def _par(x: torch.Tensor) -> torch.Tensor:
    for s in (32, 16, 8, 4, 2, 1):
        x = x ^ (x >> s)
    return x & 1


def code_of(d: torch.Tensor) -> torch.Tensor:
    """7-bit code (6 Hamming bits, then the overall parity bit) of 32-bit data, int64."""
    t = _tables(d.device)
    d = d.to(torch.int64) & LOW
    h = torch.zeros_like(d)
    for k in range(R):
        h |= _par(d & t["masks"][k]) << k
    return h | ((_par(d) ^ _par(h)) << R)


def check(d: torch.Tensor, c: torch.Tensor):
    """(fixed data, fixed code, status) for 32-bit data ``d`` and 7-bit codes ``c`` (int64)."""
    t = _tables(d.device)
    d = d & LOW
    c = c & 127
    h = code_of(d) & 63
    s = h ^ (c & 63)
    odd = (_par(d) ^ _par(c)) == 1
    st = torch.where(odd, t["lut_s"][s].to(torch.int64),
                     torch.where(s != 0, torch.full_like(s, DETECTED), torch.zeros_like(s)))
    bit = torch.where(odd, t["lut_b"][s], torch.full_like(s, -1))
    fix_d = st == CORRECTED_DATA
    one = torch.ones_like(d)
    d2 = torch.where(fix_d, d ^ (one << bit.clamp(min=0)), d)
    c2 = torch.where(fix_d | (st == CORRECTED_CHECK), code_of(d2), c)
    return d2, c2, st


# ---- node.value: code in the high bits --------------------------------------------------


def _encode_value_ref(v: torch.Tensor) -> torch.Tensor:
    """Encoded copy of an int64 slot-index tensor (plain or already encoded input)."""
    d = v.to(torch.int64) & LOW
    return d | (code_of(d) << SHIFT) | MARK


def _check_value_ref(v: torch.Tensor) -> torch.Tensor:
    """Check and correct encoded words of ``v`` in place; returns plain indices (a new tensor)."""
    if v is None or v.numel() == 0:
        return v
    t = _tables(v.device)
    marked = (v & MARK) != 0
    d2, c2, st = check(v & LOW, v >> SHIFT)
    st = torch.where(marked, st, torch.zeros_like(st))
    fixed = d2 | (c2 << SHIFT) | MARK
    v.copy_(torch.where(marked, fixed, v))
    t["err"][0] += (st == CORRECTED_DATA).sum().to(torch.int32)
    t["err"][1] += (st == DETECTED).sum().to(torch.int32)
    t["err"][4] += (st == CORRECTED_CHECK).sum().to(torch.int32)
    return v & LOW


def plain(v: torch.Tensor) -> torch.Tensor:
    return v & LOW if v.dtype == torch.int64 else v


# ---- req_to_token: shadow codes, checked at the kv_indices gather -------------------------


def _shadow_encode_ref(vals: torch.Tensor) -> torch.Tensor:
    return code_of(vals).to(torch.int8)


@triton.jit
def _par64(x):
    x = x ^ (x >> 32)
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


@triton.jit
def _check_gather_kernel(r2t, shadow, rows, lens, indptr, kv_start_idx, out, stride, masks, lut_s, lut_b, err,
                         HAS_START: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    """Re-read the entries ``create_flashinfer_kv_indices_triton`` gathered, with their
    shadow codes; fix single-bit errors in ``out`` and in the table."""
    pid = tl.program_id(0)
    row = tl.load(rows + pid).to(tl.int64)
    base = tl.load(indptr + pid).to(tl.int64)
    st0 = 0
    if HAS_START:
        st0 = tl.load(kv_start_idx + pid).to(tl.int64)
    n = tl.load(lens + pid).to(tl.int64)
    for b in range(0, tl.cdiv(n, BLOCK)):
        j = b * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = j < n
        p = row * stride + st0 + j
        d = tl.load(r2t + p, mask=m, other=0).to(tl.int64) & 0xFFFFFFFF
        c = tl.load(shadow + p, mask=m, other=0).to(tl.int64) & 127
        h = d * 0
        for k in tl.static_range(R):
            h = h | (_par64(d & tl.load(masks + k)) << k)
        s = h ^ (c & 63)
        odd = (_par64(d) ^ _par64(c)) == 1
        status = tl.where(odd, tl.load(lut_s + s), tl.where(s != 0, 3, 0))
        bit = tl.load(lut_b + s)
        fix_d = m & odd & (status == 1)
        one = d * 0 + 1
        d2 = tl.where(fix_d, d ^ (one << tl.maximum(bit, 0)), d)
        tl.store(out + base + j, d2.to(out.dtype.element_ty), mask=fix_d)
        tl.store(r2t + p, d2.to(tl.int32), mask=fix_d)
        h2 = d * 0
        for k in tl.static_range(R):
            h2 = h2 | (_par64(d2 & tl.load(masks + k)) << k)
        c2 = h2 | ((_par64(d2) ^ _par64(h2)) << R)
        fix_c = m & odd & ((status == 1) | (status == 2))
        tl.store(shadow + p, c2.to(tl.int8), mask=fix_c)
        nd = tl.sum(fix_d.to(tl.int32), 0)
        nc = tl.sum((m & odd & (status == 2)).to(tl.int32), 0)
        nb = tl.sum((m & (status == 3)).to(tl.int32), 0)
        if nd > 0:
            tl.atomic_add(err + 2, nd)
        if nb > 0:
            tl.atomic_add(err + 3, nb)
        if nc > 0:
            tl.atomic_add(err + 5, nc)


# ---- fused kernels: one launch per call instead of ~60 small torch ops ----------------


def _use_triton(t: torch.Tensor) -> bool:
    return t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _code7(d, masks, R: tl.constexpr):
    h = d * 0
    for k in tl.static_range(R):
        h = h | (_par64(d & tl.load(masks + k)) << k)
    return h | ((_par64(d) ^ _par64(h)) << R)


@triton.jit
def _encode_kernel(src, dst, n, masks, MODE: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    """MODE 0: dst[i] = int8 code of src[i]; MODE 1: dst[i] = encoded int64 word."""
    j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = j < n
    d = tl.load(src + j, mask=m, other=0).to(tl.int64) & 0xFFFFFFFF
    c = _code7(d, masks, R)
    if MODE == 0:
        tl.store(dst + j, c.to(tl.int8), mask=m)
    else:
        tl.store(dst + j, d | (c << 32) | (1 << 39), mask=m)


@triton.jit
def _value_check_kernel(v, out, n, masks, lut_s, lut_b, err, R: tl.constexpr, BLOCK: tl.constexpr):
    """Check encoded int64 words of ``v``, fix them in place, write plain indices to ``out``."""
    j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = j < n
    w = tl.load(v + j, mask=m, other=0)
    d = w & 0xFFFFFFFF
    marked = m & (((w >> 39) & 1) == 1)
    c = (w >> 32) & 127
    h = d * 0
    for k in tl.static_range(R):
        h = h | (_par64(d & tl.load(masks + k)) << k)
    s = h ^ (c & 63)
    odd = (_par64(d) ^ _par64(c)) == 1
    status = tl.where(odd, tl.load(lut_s + s), tl.where(s != 0, 3, 0))
    status = tl.where(marked, status, 0)
    bit = tl.load(lut_b + s)
    fix_d = status == 1
    one = d * 0 + 1
    d2 = tl.where(fix_d, d ^ (one << tl.maximum(bit, 0)), d)
    fix = (status == 1) | (status == 2)
    tl.store(v + j, d2 | (_code7(d2, masks, R) << 32) | (1 << 39), mask=fix)
    tl.store(out + j, d2, mask=m)
    nd = tl.sum(fix_d.to(tl.int32), 0)
    nb = tl.sum((status == 3).to(tl.int32), 0)
    nc = tl.sum((status == 2).to(tl.int32), 0)
    if nd > 0:
        tl.atomic_add(err + 0, nd)
    if nb > 0:
        tl.atomic_add(err + 1, nb)
    if nc > 0:
        tl.atomic_add(err + 4, nc)


_BLK = 1024


def encode_value(v: torch.Tensor) -> torch.Tensor:
    """Encoded copy of an int64 slot-index tensor (plain or already encoded input)."""
    if not _use_triton(v):
        return _encode_value_ref(v)
    t = _tables(v.device)
    src = v.contiguous()
    out = torch.empty(src.shape, dtype=torch.int64, device=v.device)
    n = src.numel()
    if n:
        _encode_kernel[(triton.cdiv(n, _BLK),)](src, out, n, t["masks"], MODE=1, R=R, BLOCK=_BLK)
    return out


def shadow_encode(vals: torch.Tensor) -> torch.Tensor:
    if not _use_triton(vals):
        return _shadow_encode_ref(vals)
    t = _tables(vals.device)
    src = vals.contiguous()
    out = torch.empty(src.shape, dtype=torch.int8, device=vals.device)
    n = src.numel()
    if n:
        _encode_kernel[(triton.cdiv(n, _BLK),)](src, out, n, t["masks"], MODE=0, R=R, BLOCK=_BLK)
    return out


def check_value_(v: torch.Tensor) -> torch.Tensor:
    """Check and correct encoded words of ``v`` in place; returns plain indices (a new tensor)."""
    if v is None or v.numel() == 0:
        return v
    if not _use_triton(v) or not v.is_contiguous():
        return _check_value_ref(v)
    t = _tables(v.device)
    out = torch.empty_like(v)
    n = v.numel()
    _value_check_kernel[(triton.cdiv(n, _BLK),)](v, out, n, t["masks"], t["lut_s"], t["lut_b"], t["err"],
                                                 R=R, BLOCK=_BLK)
    return out


def check_value_once(v: torch.Tensor) -> torch.Tensor:
    """``check_value_`` at most once per forward step for the same tensor object.

    The scheduler matches every waiting request against the tree on every step (LPM ordering,
    then admission), so a shared node is matched many times between two forwards. The first
    match of a step checks and corrects the node; later ones reuse its plain copy. Node values
    are replaced, not mutated, on split / evict, so the tensor object identifies the content.
    """
    import weakref

    seen = _state["seen"]
    hit = seen.get(id(v))
    if hit is not None and hit[0]() is v and hit[1] == _state["step"]:
        return hit[2]
    out = check_value_(v)
    seen[id(v)] = (weakref.ref(v), _state["step"], out)
    return out


def next_step() -> None:
    _state["step"] += 1
    if len(_state["seen"]) > 4096:
        _state["seen"].clear()


def check_gather(r2t, shadow, rows, lens, indptr, kv_start_idx, out) -> None:
    t = _tables(r2t.device)
    bs = int(lens.numel())
    if bs == 0:
        return
    _check_gather_kernel[(bs,)](
        r2t, shadow, rows, lens, indptr, kv_start_idx, out, r2t.stride(0), t["masks"], t["lut_s"], t["lut_b"],
        t["err"], HAS_START=kv_start_idx is not None, R=R, BLOCK=512,
    )


def stats(reset: bool = False) -> dict:
    err = _state["err"]
    vals = [int(x) for x in err.tolist()] if err is not None else [0] * 7
    if reset and err is not None:
        err.zero_()
    names = ["node_corrected", "node_uncorrectable", "r2t_corrected", "r2t_uncorrectable",
             "node_check_bits_fixed", "r2t_check_bits_fixed", "seq_lens_restored"]
    return dict(zip(names, vals))


def install() -> None:
    if os.environ.get("RADIXMEND_INDEX") != "1":
        return
    from dataclasses import replace

    from sglang.srt.mem_cache import allocation as al
    from sglang.srt.mem_cache import kv_index_translator as kt
    from sglang.srt.mem_cache import memory_pool as mp
    from sglang.srt.mem_cache import unified_radix_cache as urc
    from sglang.srt.mem_cache.allocator import token as tok
    from sglang.srt.mem_cache.unified_cache import unified_tree_core as utc
    from sglang.srt.mem_cache.unified_cache.components.base import BASE_COMPONENT_TYPE

    if getattr(urc.UnifiedRadixCache, "_rm_index", False):
        return

    # req_to_token: shadow codes
    orig_r2t_init = mp.ReqToTokenPool.__init__

    def r2t_init(self, *args, **kwargs):
        orig_r2t_init(self, *args, **kwargs)
        _tables(self.req_to_token.device)
        self.rm_shadow = torch.zeros(self.req_to_token.shape, dtype=torch.int8, device=self.req_to_token.device)
        _state["r2t_pool"] = self

    orig_write = mp.ReqToTokenPool.write

    def r2t_write(self, indices, values):
        if isinstance(values, torch.Tensor) and values.dtype == torch.int64:
            values = values & LOW
        orig_write(self, indices, values)
        sh = getattr(self, "rm_shadow", None)
        if sh is not None:
            sh[indices] = shadow_encode(self.req_to_token[indices])

    orig_wci = al.write_cache_indices

    def write_cache_indices(out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu, prefix_lens_tensor,
                            prefix_lens_cpu, seq_lens_tensor, seq_lens_cpu, extend_lens_tensor, extend_lens_cpu,
                            prefix_tensors, req_to_token_pool):
        orig_wci(out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu, prefix_lens_tensor, prefix_lens_cpu,
                 seq_lens_tensor, seq_lens_cpu, extend_lens_tensor, extend_lens_cpu, prefix_tensors,
                 req_to_token_pool)
        sh = getattr(req_to_token_pool, "rm_shadow", None)
        if sh is not None and seq_lens_cpu.numel():
            # every entry [0, seq_len) of these rows was just written (prefix + new tokens)
            w = int(seq_lens_cpu.max())
            rows = req_pool_indices_tensor.to(torch.int64)
            sh[rows, :w] = shadow_encode(req_to_token_pool.req_to_token[rows, :w])

    orig_pack = kt.KVIndexTranslator.pack_read_stream

    def pack_read_stream(self, plan, *, req_pool_indices, seq_lens, indptr, out, kv_start_idx=None,
                         kind=kt.IdSpaceKind.FULL, token_mapping=None):
        res = orig_pack(self, plan, req_pool_indices=req_pool_indices, seq_lens=seq_lens, indptr=indptr, out=out,
                        kv_start_idx=kv_start_idx, kind=kind, token_mapping=token_mapping)
        next_step()
        pool = _state["r2t_pool"]
        if (pool is not None and token_mapping is None and not self._reads_translated(kind)
                and self.req_to_token is pool.req_to_token):
            bs = int(seq_lens.numel())
            check_gather(pool.req_to_token, pool.rm_shadow, req_pool_indices[:bs], seq_lens, indptr,
                         kv_start_idx, out)
        return res

    # node.value: code in the high bits
    orig_insert = urc.UnifiedRadixCache.insert

    def insert(self, params):
        if params.value is not None and params.value.dtype == torch.int64 and params.value.numel():
            params = replace(params, value=encode_value(params.value))
        return orig_insert(self, params)

    orig_collect = utc.UnifiedTreeCore.collect_full_device_indices

    def collect_full_device_indices(self, from_node_id, until_node_id):
        until = self.node_by_id(until_node_id)
        node = self.node_by_id(from_node_id)
        while node is not None and node is not until:
            v = node.component_data[BASE_COMPONENT_TYPE].value
            if isinstance(v, torch.Tensor) and v.numel():
                check_value_(v)
            node = node.parent
        return plain(orig_collect(self, from_node_id, until_node_id))

    # match_prefix gathers node values itself (not through collect_full_device_indices)
    orig_helper = utc.UnifiedTreeCore._match_prefix_helper

    def _match_prefix_helper(self, key):
        value, *rest = orig_helper(self, key)
        value = [check_value_once(v) if isinstance(v, torch.Tensor) and v.dtype == torch.int64 else v for v in value]
        return (value, *rest)

    orig_free = tok.TokenToKVPoolAllocator.free

    def free(self, free_index):
        if isinstance(free_index, torch.Tensor) and free_index.dtype == torch.int64 and free_index.numel():
            free_index = encode_value(check_value_(free_index.clone()))
        return orig_free(self, free_index)

    orig_clear = tok.TokenToKVPoolAllocator.clear

    def clear(self):
        orig_clear(self)
        self.free_pages = encode_value(self.free_pages)

    orig_alloc = tok.TokenToKVPoolAllocator.alloc

    def alloc(self, need_size):
        sel = orig_alloc(self, need_size)
        if isinstance(sel, torch.Tensor) and sel.numel():
            sel = check_value_(sel.clone())
        return sel

    orig_all_free = tok.TokenToKVPoolAllocator.get_all_free_pages

    def get_all_free_pages(self):
        fp = orig_all_free(self)
        return plain(fp) if isinstance(fp, torch.Tensor) else fp

    from sglang.srt.managers import schedule_batch as sbm

    def _restore_seq_lens(batch):
        gl, cl = getattr(batch, "seq_lens", None), getattr(batch, "seq_lens_cpu", None)
        if (isinstance(gl, torch.Tensor) and isinstance(cl, torch.Tensor) and gl.numel()
                and gl.shape == cl.shape and gl.dtype == cl.dtype):
            fresh = cl.to(gl.device, non_blocking=True)
            t = _tables(gl.device)
            t["err"][6] += (gl != fresh).sum().to(torch.int32)
            batch.seq_lens = fresh  # new tensor, like prepare_for_decode does

    orig_pfd = sbm.ScheduleBatch.prepare_for_decode
    orig_pfe = sbm.ScheduleBatch.prepare_for_extend

    def prepare_for_decode(self, *args, **kwargs):
        _restore_seq_lens(self)  # the prep itself reads seq_lens (slot allocation, req_to_token writes)
        res = orig_pfd(self, *args, **kwargs)
        _restore_seq_lens(self)
        return res

    def prepare_for_extend(self, *args, **kwargs):
        _restore_seq_lens(self)  # the prep itself reads seq_lens (slot allocation, req_to_token writes)
        res = orig_pfe(self, *args, **kwargs)
        _restore_seq_lens(self)
        return res

    mp.ReqToTokenPool.__init__ = r2t_init
    mp.ReqToTokenPool.write = r2t_write
    al.write_cache_indices = write_cache_indices
    kt.KVIndexTranslator.pack_read_stream = pack_read_stream
    urc.UnifiedRadixCache.insert = insert
    utc.UnifiedTreeCore.collect_full_device_indices = collect_full_device_indices
    utc.UnifiedTreeCore._match_prefix_helper = _match_prefix_helper
    tok.TokenToKVPoolAllocator.free = free
    tok.TokenToKVPoolAllocator.clear = clear
    tok.TokenToKVPoolAllocator.alloc = alloc
    tok.TokenToKVPoolAllocator.get_all_free_pages = get_all_free_pages
    # The forward must not share its seq_lens tensor with the scheduler: with overlap scheduling the
    # scheduler's tensor can change (or be hit) while the forward that was handed it is still queued.
    # Build the forward from a private copy of the CPU mirror, then give the scheduler another one.
    from sglang.srt.model_executor import forward_batch_info as fbi

    orig_fb_init = fbi.ForwardBatch.init_new

    def fb_init_new(cls, batch, *args, **kwargs):
        gl, cl = getattr(batch, "seq_lens", None), getattr(batch, "seq_lens_cpu", None)
        ok = (isinstance(gl, torch.Tensor) and isinstance(cl, torch.Tensor) and gl.numel()
              and gl.shape == cl.shape and gl.dtype == cl.dtype)
        if ok:
            _restore_seq_lens(batch)
        fb = orig_fb_init(batch, *args, **kwargs)
        if ok:
            batch.seq_lens = cl.to(gl.device, non_blocking=True)  # the scheduler's own copy
        return fb

    fbi.ForwardBatch.init_new = classmethod(fb_init_new)
    sbm.ScheduleBatch.prepare_for_decode = prepare_for_decode
    sbm.ScheduleBatch.prepare_for_extend = prepare_for_extend
    urc.UnifiedRadixCache._rm_index = True
