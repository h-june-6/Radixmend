"""Module A prototype inside SGLang v0.5.22: SECDED checkwords on the KV pool,
checked and corrected inside the Triton decode attention kernel.

Scope of this prototype (enough to measure decode overhead, RQ3):
* checkword buffers: one int16 per (slot, kv head) for every layer's K and V
* encode-at-store: a separate Triton kernel right after ``set_kv_buffer``
  (re-reads the rows just written; a fused store kernel comes later)
* check-on-load: a copy of ``_fwd_grouped_kernel_stage1`` (GQA, page size 1,
  no MLA / score_mod / xai) that computes the syndrome of every K and V vector
  it loads and corrects single-bit errors in registers
* check-before-extend: before the extend (prefill) attention reads a cached
  prefix, a separate kernel checks every prefix slot of that layer and writes
  corrected values back in place (scrub on read), so later reads are clean
* error counters: decode [K corrected, K uncorrectable, V corrected, V uncorrectable]
  then the same four for extend
* decode check variants (RADIXMEND_SECDED_MODE / _SAMPLE / _POPC): see install()
* NOT covered yet: MHA (kv_group_num == 1) decode path, page size > 1

Syndrome math is the same code as ``radixmend.secded_vec`` but computed in the
cheaper form: the in-element bit syndrome S_b and the overall parity only need
the XOR of all elements of the vector (bit k of S_b is the parity of
``xor_all & b_masks[k]``), so only the element-label syndrome S_e needs a
per-element popcount.

Enable with ``RADIXMEND_SECDED=1`` (loaded by ``radixmend/sglang_site``).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from radixmend.secded_vec import VecCode

_state = {"pools": {}, "code": None, "tables": None, "err": None, "by_ptr": {}}
LOG_CAP = 4096  # error log entries (uncorrectable detections) kept until read


# ---- shared jit helpers ---------------------------------------------------------------


@triton.jit
def _xor(a, b):
    return a ^ b


@triton.jit
def _par16(v):
    v = v ^ (v >> 8)
    v = v ^ (v >> 4)
    v = v ^ (v >> 2)
    v = v ^ (v >> 1)
    return v & 1


@triton.jit
def _par16_fast(v, POPC: tl.constexpr):
    if POPC:
        return tl.extra.cuda.libdevice.popc(v) & 1
    else:
        return _par16(v)


@triton.jit
def _sb_of(c, bmask_ptr, B: tl.constexpr):
    sb = c * 0
    for k in tl.static_range(B):
        m = tl.load(bmask_ptr + k)
        sb = sb | (_par16(c & m) << k)
    return sb


@triton.jit
def _check_tile(x16, cw, mask_n, eidx, e_lab, bmask_ptr, inv_e_ptr, inv_b_ptr, AXIS: tl.constexpr,
                E: tl.constexpr, B: tl.constexpr, GATE: tl.constexpr, POPC: tl.constexpr):
    """Check one K or V tile (vectors along AXIS, raw int16 bits).

    Returns (fix, n_fix, n_bad, n_cw, cw_new): fix is the XOR mask to apply (int32 tile, zero
    when clean). With GATE the expensive element-label syndrome is only computed
    when the overall parity or the bit syndrome of some vector in the tile does
    not match; that misses even-weight errors whose bit labels cancel (two flips
    at the same bit position of two elements of one vector). n_cw counts vectors
    whose data is clean but whose stored checkword has a single-bit error;
    cw_new holds the recomputed checkword there, -2 for uncorrectable vectors
    and -1 elsewhere.
    """
    c = tl.reduce(x16, AXIS, _xor).to(tl.int32) & 0xFFFF
    p1 = _par16(c)
    sb1 = _sb_of(c, bmask_ptr, B)
    fix = x16.to(tl.int32) * 0
    n_fix = tl.sum(mask_n.to(tl.int32), 0) * 0
    n_bad = n_fix
    n_cw = n_fix
    cw_new = cw * 0 - 1
    go = True
    if GATE:
        p0 = cw & 1
        sb0 = (cw >> (1 + E)) & ((1 << B) - 1)
        go = tl.max(tl.where(mask_n & ((p1 != p0) | (sb1 != sb0)), 1, 0), 0) > 0
    if go:
        x = x16.to(tl.int32) & 0xFFFF
        se1 = tl.reduce(tl.where(_par16_fast(x, POPC) == 1, e_lab, 0), AXIS, _xor)
        st, el, bt = _status(p1, se1, sb1, cw, inv_e_ptr, inv_b_ptr, E, B)
        st = tl.where(mask_n, st, 0)
        if tl.max(st, 0) > 0:
            fix = tl.where((tl.expand_dims(st, AXIS) == 1) & (eidx == tl.expand_dims(el, AXIS)),
                           1 << tl.expand_dims(bt, AXIS), 0)
            n_fix = tl.sum((st == 1).to(tl.int32), 0)
            n_bad = tl.sum((st == 3).to(tl.int32), 0)
            n_cw = tl.sum((st == 2).to(tl.int32), 0)
            q1 = _par16(se1) ^ _par16(sb1)
            good = p1 | (se1 << 1) | (sb1 << (1 + E)) | (q1 << (1 + E + B))
            cw_new = tl.where(st == 2, good, tl.where(st == 3, -2, -1))  # -2 marks uncorrectable
    return fix, n_fix, n_bad, n_cw, cw_new


@triton.jit
def _log_bad(log_ptr, logn_ptr, CAP, slot, head, tag, mask):
    """Append (slot, head, tag) for every masked entry to the error log (drops past CAP)."""
    idx = tl.atomic_add(logn_ptr + slot * 0, mask.to(tl.int32) * 0 + 1, mask=mask)
    ok = mask & (idx < CAP)
    tl.store(log_ptr + idx * 3, slot.to(tl.int32), mask=ok)
    tl.store(log_ptr + idx * 3 + 1, (slot * 0 + head).to(tl.int32), mask=ok)
    tl.store(log_ptr + idx * 3 + 2, (slot * 0 + tag).to(tl.int32), mask=ok)


@triton.jit
def _syndrome(x, e_lab, bmask_ptr, AXIS: tl.constexpr, B: tl.constexpr):
    """(p, se, sb) for every vector of a tile; vectors run along AXIS."""
    c = tl.reduce(x, AXIS, _xor)
    p = _par16(c)
    se = tl.reduce(tl.where(_par16(x) == 1, e_lab, 0), AXIS, _xor)
    sb = p * 0
    for k in tl.static_range(B):
        m = tl.load(bmask_ptr + k)
        sb = sb | (_par16(c & m) << k)
    return p, se, sb


@triton.jit
def _status(p1, se1, sb1, cw, inv_e_ptr, inv_b_ptr, E: tl.constexpr, B: tl.constexpr):
    p0 = cw & 1
    se0 = (cw >> 1) & ((1 << E) - 1)
    sb0 = (cw >> (1 + E)) & ((1 << B) - 1)
    q0 = (cw >> (1 + E + B)) & 1
    dp = p1 ^ p0
    dse = se1 ^ se0
    dsb = sb1 ^ sb0
    dq = p1 ^ p0 ^ _par16(se0) ^ _par16(sb0) ^ q0
    ds = (dse << B) | dsb
    syn = dp | (dse << 1) | (dsb << (1 + E)) | (dq << (1 + E + B))
    elem = tl.load(inv_e_ptr + dse)
    bit = tl.load(inv_b_ptr + dsb)
    ds_unit = (ds != 0) & ((ds & (ds - 1)) == 0)
    is_data = (dq == 1) & (dp == 1) & (ds != 0) & (elem >= 0) & (bit >= 0)
    is_check = (dq == 1) & (((dp == 1) & (ds == 0)) | ((dp == 0) & ds_unit) | ((dp == 0) & (ds == 0)))
    status = tl.where(syn == 0, 0, tl.where(is_data, 1, tl.where(is_check, 2, 3)))
    return status, elem, bit


@triton.jit
def _encode_scatter_kernel(buf_ptr, cw_ptr, loc_ptr, elab_ptr, bmask_ptr, stride_s, stride_h, H,
                           D: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    n = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    slot = tl.load(loc_ptr + n).to(tl.int64)
    x = tl.load(buf_ptr + slot * stride_s + h * stride_h + d).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    e_lab = tl.load(elab_ptr + d)
    p, se, sb = _syndrome(x, e_lab, bmask_ptr, 0, B)
    q = _par16(se) ^ _par16(sb)
    cw = p | (se << 1) | (sb << (1 + E)) | (q << (1 + E + B))
    tl.store(cw_ptr + slot * H + h, cw.to(tl.int16))


@triton.jit
def _check_fix_inplace_kernel(buf_ptr, cw_ptr, loc_ptr, indptr_ptr, nb, elab_ptr, bmask_ptr, inv_e_ptr,
                              inv_b_ptr, err_ptr, cwerr_ptr, stride_s, stride_h, H, log_ptr, logn_ptr, LOGCAP,
                              tag, D: tl.constexpr, E: tl.constexpr, B: tl.constexpr, REP: tl.constexpr = False,
                              rep_map=None, rep_buf=None, rep_cw=None, rest_ptr=None):
    """Check buf[loc[n], h] for n < indptr[nb]; fix single data-bit errors in place.

    REP (module D): an uncorrectable vector whose slot has a replica is overwritten
    in place with the replica (data and checkword) and not logged for module C."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    total = tl.load(indptr_ptr + nb)
    if n < total:
        d = tl.arange(0, D)
        slot = tl.load(loc_ptr + n).to(tl.int64)
        ptr = buf_ptr + slot * stride_s + h * stride_h
        x = tl.load(ptr + d).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        p1, se1, sb1 = _syndrome(x, tl.load(elab_ptr + d), bmask_ptr, 0, B)
        cw = tl.load(cw_ptr + slot * H + h).to(tl.int32) & 0xFFFF
        st, el, bt = _status(p1, se1, sb1, cw, inv_e_ptr, inv_b_ptr, E, B)
        if st == 1:
            xe = tl.load(ptr + el).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            fixed = (xe ^ (1 << bt)).to(tl.int16).to(buf_ptr.dtype.element_ty, bitcast=True)
            tl.store(ptr + el, fixed)
            tl.atomic_add(err_ptr + 0, 1)
        if st == 3:
            tl.atomic_add(err_ptr + 1, 1)
            unrep = st == 3
            if REP:
                rs = tl.load(rep_map + slot).to(tl.int64)
                if rs >= 0:
                    tl.store(ptr + d, tl.load(rep_buf + rs * stride_s + h * stride_h + d))
                    tl.store(cw_ptr + slot * H + h, tl.load(rep_cw + rs * H + h))
                    tl.atomic_add(rest_ptr, 1)
                    unrep = st < 0
            _log_bad(log_ptr, logn_ptr, LOGCAP, slot, h, tag, unrep)
        if st == 2:
            # data is clean, the stored checkword took the hit: rewrite it
            q1 = _par16(se1) ^ _par16(sb1)
            good = p1 | (se1 << 1) | (sb1 << (1 + E)) | (q1 << (1 + E + B))
            tl.store(cw_ptr + slot * H + h, good.to(tl.int16))
            tl.atomic_add(cwerr_ptr, 1)


@triton.jit
def _check_fix_blocked_kernel(buf_ptr, cw_ptr, loc_ptr, indptr_ptr, nb, elab_ptr, bmask_ptr, inv_e_ptr,
                              inv_b_ptr, err_ptr, cwerr_ptr, stride_s, stride_h, H, log_ptr, logn_ptr, LOGCAP,
                              tag, BLOCK: tl.constexpr, D: tl.constexpr, E: tl.constexpr, B: tl.constexpr,
                              POPC: tl.constexpr, REP: tl.constexpr = False, rep_map=None, rep_buf=None,
                              rep_cw=None, rest_ptr=None):
    """``_check_fix_inplace_kernel`` on BLOCK slots per program (same decisions, one launch
    unit per BLOCK vectors instead of per vector; the per-vector version spent most of the
    pre-extend time on scheduling ~2M tiny programs per prefill)."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    total = tl.load(indptr_ptr + nb)
    n = pid * BLOCK + tl.arange(0, BLOCK)
    m = n < total
    d = tl.arange(0, D)
    slot = tl.load(loc_ptr + n, mask=m, other=0).to(tl.int64)
    rows = buf_ptr + slot[:, None] * stride_s + h * stride_h + d[None, :]
    x = tl.load(rows, mask=m[:, None], other=0).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    c = tl.reduce(x, 1, _xor)
    p1 = _par16(c)
    se1 = tl.reduce(tl.where(_par16_fast(x, POPC) == 1, tl.load(elab_ptr + d)[None, :], 0), 1, _xor)
    sb1 = _sb_of(c, bmask_ptr, B)
    cw = tl.load(cw_ptr + slot * H + h, mask=m, other=0).to(tl.int32) & 0xFFFF
    st, el, bt = _status(p1, se1, sb1, cw, inv_e_ptr, inv_b_ptr, E, B)
    st = tl.where(m, st, 0)
    if tl.max(st, 0) > 0:
        one = st == 1
        eptr = buf_ptr + slot * stride_s + h * stride_h + tl.where(one, el, 0)
        xe = tl.load(eptr, mask=one, other=0).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        fixed = (xe ^ (1 << tl.where(one, bt, 0))).to(tl.int16).to(buf_ptr.dtype.element_ty, bitcast=True)
        tl.store(eptr, fixed, mask=one)
        tl.atomic_add(err_ptr + 0, tl.sum(one.to(tl.int32), 0))
        bad = st == 3
        tl.atomic_add(err_ptr + 1, tl.sum(bad.to(tl.int32), 0))
        unrep = bad
        if REP:
            rs = tl.load(rep_map + slot, mask=bad, other=-1).to(tl.int64)
            got = bad & (rs >= 0)
            rsafe = tl.where(got, rs, 0)
            tl.store(rows, tl.load(rep_buf + rsafe[:, None] * stride_s + h * stride_h + d[None, :],
                                   mask=got[:, None], other=0), mask=got[:, None])
            tl.store(cw_ptr + slot * H + h, tl.load(rep_cw + rsafe * H + h, mask=got, other=0), mask=got)
            tl.atomic_add(rest_ptr, tl.sum(got.to(tl.int32), 0))
            unrep = bad & (rs < 0)
        _log_bad(log_ptr, logn_ptr, LOGCAP, slot, h, tag, unrep)
        ck = st == 2
        q1 = _par16(se1) ^ _par16(sb1)
        good = p1 | (se1 << 1) | (sb1 << (1 + E)) | (q1 << (1 + E + B))
        tl.store(cw_ptr + slot * H + h, good.to(tl.int16), mask=ck)
        tl.atomic_add(cwerr_ptr, tl.sum(ck.to(tl.int32), 0))


# ---- decode stage 1 with check-on-load -------------------------------------------------


@triton.jit
def _split(q_k, K_Buffer, V_Buffer, K_CW, V_CW, kv_indices, kv_start_idx, split_kv_start, split_kv_end,
           cur_kv_head, H_KV, sm_scale_withk, mask_h, offs_d, offs_dv, mask_d, mask_dv,
           stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh, elab_ptr, bmask_ptr, inv_e_ptr,
           inv_b_ptr, phase_ptr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
           BLOCK_DV: tl.constexpr,
           logit_cap: tl.constexpr, E: tl.constexpr, B: tl.constexpr, TM: tl.constexpr,
           POPC: tl.constexpr, SAMPLE: tl.constexpr, WB: tl.constexpr = False, count_ptr=None,
           DEDUP: tl.constexpr = False, SIGRED: tl.constexpr = False, log_ptr=None, logn_ptr=None,
           LOGCAP=0, layer=0, REP: tl.constexpr = False, rep_map=None, RK=None, RV=None, RKC=None, RVC=None,
           rest_ptr=None):
    """Attention over one KV split. WB: write corrected words back to the KV buffer. TM: 0 no check, 1 full, 2 gate, 3 signature (no correction).

    TM 3 only accumulates, element-wise, the XOR of every K and V word and of the
    stored checkwords; at the end the full syndrome (p, S_e, S_b) of the per-element
    totals is compared with the XOR of the stored checkwords (see ``_sig_bad``). Any
    odd number of flipped bits shows up (p changes) and so does any 2-bit pattern
    except the same element and bit in two vectors of the split. The caller redoes
    the split with TM 1 when ``bad`` is set.
    """
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
    base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
    base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]
    nk_fix = tl.sum(mask_h.to(tl.int32), 0) * 0
    nk_bad = nk_fix
    nv_fix = nk_fix
    nv_bad = nk_fix
    n_cw = nk_fix
    bad = nk_fix
    if TM == 1 or TM == 2:
        elab_k = tl.load(elab_ptr + offs_d)[:, None]
        elab_v = tl.load(elab_ptr + offs_dv)[None, :]
        if SAMPLE > 1:
            phase = tl.load(phase_ptr)
    if TM == 3:
        # SIGRED: fold each tile to one word per element right away (fewer live registers,
        # one XOR reduction per tile) instead of keeping a whole tile of accumulators
        if SIGRED:
            xk_acc = tl.zeros([BLOCK_DMODEL, 1], dtype=tl.int16)
            xv_acc = tl.zeros([1, BLOCK_DV], dtype=tl.int16)
        else:
            xk_acc = tl.zeros([BLOCK_DMODEL, BLOCK_N], dtype=tl.int16)
            xv_acc = tl.zeros([BLOCK_N, BLOCK_DV], dtype=tl.int16)
        ck_acc = tl.zeros([BLOCK_N], dtype=tl.int16)
        cv_acc = tl.zeros([BLOCK_N], dtype=tl.int16)
    for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < split_kv_end
        kv_loc = tl.load(kv_indices + kv_start_idx + offs_n, mask=mask_n, other=0)
        offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
        k = tl.load(K_Buffer + offs_buf_k, mask=mask_n[None, :] & (mask_d[:, None]), other=0.0)
        if TM == 3:
            ck = tl.load(K_CW + kv_loc * H_KV + cur_kv_head, mask=mask_n, other=0)
            if DEDUP:
                # slots read by several requests this step were checked once by the pre-pass
                sh = tl.load(count_ptr + kv_loc, mask=mask_n, other=0) >= 2
                k16 = tl.where(sh[None, :], 0, k.to(tl.int16, bitcast=True))
                ck_acc = ck_acc ^ tl.where(sh, 0, ck)
            else:
                k16 = k.to(tl.int16, bitcast=True)
                ck_acc = ck_acc ^ ck
            if SIGRED:
                xk_acc = xk_acc ^ tl.reduce(k16, 1, _xor, keep_dims=True)
            else:
                xk_acc = xk_acc ^ k16
        if TM == 1 or TM == 2:
            do_k = True
            if SAMPLE > 1:
                # check each tile once every SAMPLE decode steps (rotating phase)
                do_k = ((start_n // BLOCK_N + phase) % SAMPLE) == 0
            if do_k:
                cwk = tl.load(K_CW + kv_loc * H_KV + cur_kv_head, mask=mask_n, other=0).to(tl.int32) & 0xFFFF
                fk, a, b_, c_, cwk_new = _check_tile(k.to(tl.int16, bitcast=True), cwk, mask_n, offs_d[:, None], elab_k,
                                        bmask_ptr, inv_e_ptr, inv_b_ptr, 0, E, B, TM == 2, POPC)
                if a + b_ > 0:
                    k = (k.to(tl.int16, bitcast=True).to(tl.int32) ^ fk).to(tl.int16).to(
                        K_Buffer.dtype.element_ty, bitcast=True)
                    if WB:
                        # only the corrected words; racing programs write the same value
                        tl.store(K_Buffer + offs_buf_k, k, mask=mask_n[None, :] & mask_d[:, None] & (fk != 0))
                    nk_fix += a
                    nk_bad += b_
                    if b_ > 0:
                        bad_k = mask_n & (cwk_new == -2)
                        if REP:
                            # module D: take uncorrectable vectors from the replica (same layout as the pool)
                            rs = tl.load(rep_map + kv_loc, mask=bad_k, other=-1)
                            use = bad_k & (rs >= 0)
                            rs = tl.where(use, rs, 0).to(tl.int64)
                            kr = tl.load(RK + rs[None, :] * stride_buf_kbs + base_offs_k,
                                         mask=use[None, :] & mask_d[:, None], other=0.0)
                            k = tl.where(use[None, :], kr, k)
                            if WB:
                                tl.store(K_Buffer + offs_buf_k, k, mask=use[None, :] & mask_d[:, None])
                                tl.store(K_CW + kv_loc * H_KV + cur_kv_head,
                                         tl.load(RKC + rs * H_KV + cur_kv_head, mask=use, other=0), mask=use)
                            tl.atomic_add(rest_ptr, tl.sum(use.to(tl.int32), 0))
                            bad_k = bad_k & (use == 0)
                        _log_bad(log_ptr, logn_ptr, LOGCAP, kv_loc, cur_kv_head, layer * 4, bad_k)
                if c_ > 0:
                    n_cw += c_
                    if WB:
                        tl.store(K_CW + kv_loc * H_KV + cur_kv_head, cwk_new.to(tl.int16), mask=mask_n & (cwk_new >= 0))
        qk = tl.dot(q_k, k)
        qk *= sm_scale_withk
        if logit_cap > 0:
            qk = logit_cap * tl.extra.cuda.libdevice.tanh(qk / logit_cap)
        qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))
        offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
        v = tl.load(V_Buffer + offs_buf_v, mask=mask_n[:, None] & (mask_dv[None, :]), other=0.0)
        if TM == 3:
            cv = tl.load(V_CW + kv_loc * H_KV + cur_kv_head, mask=mask_n, other=0)
            if DEDUP:
                v16 = tl.where(sh[:, None], 0, v.to(tl.int16, bitcast=True))
                cv_acc = cv_acc ^ tl.where(sh, 0, cv)
            else:
                v16 = v.to(tl.int16, bitcast=True)
                cv_acc = cv_acc ^ cv
            if SIGRED:
                xv_acc = xv_acc ^ tl.reduce(v16, 0, _xor, keep_dims=True)
            else:
                xv_acc = xv_acc ^ v16
        if TM == 1 or TM == 2:
            if do_k:
                cwv = tl.load(V_CW + kv_loc * H_KV + cur_kv_head, mask=mask_n, other=0).to(tl.int32) & 0xFFFF
                fv, a, b_, c_, cwv_new = _check_tile(v.to(tl.int16, bitcast=True), cwv, mask_n, offs_dv[None, :], elab_v,
                                        bmask_ptr, inv_e_ptr, inv_b_ptr, 1, E, B, TM == 2, POPC)
                if a + b_ > 0:
                    v = (v.to(tl.int16, bitcast=True).to(tl.int32) ^ fv).to(tl.int16).to(
                        V_Buffer.dtype.element_ty, bitcast=True)
                    if WB:
                        tl.store(V_Buffer + offs_buf_v, v, mask=mask_n[:, None] & mask_dv[None, :] & (fv != 0))
                    nv_fix += a
                    nv_bad += b_
                    if b_ > 0:
                        bad_v = mask_n & (cwv_new == -2)
                        if REP:
                            rs = tl.load(rep_map + kv_loc, mask=bad_v, other=-1)
                            use = bad_v & (rs >= 0)
                            rs = tl.where(use, rs, 0).to(tl.int64)
                            vr = tl.load(RV + rs[:, None] * stride_buf_vbs + base_offs_v,
                                         mask=use[:, None] & mask_dv[None, :], other=0.0)
                            v = tl.where(use[:, None], vr, v)
                            if WB:
                                tl.store(V_Buffer + offs_buf_v, v, mask=use[:, None] & mask_dv[None, :])
                                tl.store(V_CW + kv_loc * H_KV + cur_kv_head,
                                         tl.load(RVC + rs * H_KV + cur_kv_head, mask=use, other=0), mask=use)
                            tl.atomic_add(rest_ptr, tl.sum(use.to(tl.int32), 0))
                            bad_v = bad_v & (use == 0)
                        _log_bad(log_ptr, logn_ptr, LOGCAP, kv_loc, cur_kv_head, layer * 4 + 2, bad_v)
                if c_ > 0:
                    n_cw += c_
                    if WB:
                        tl.store(V_CW + kv_loc * H_KV + cur_kv_head, cwv_new.to(tl.int16), mask=mask_n & (cwv_new >= 0))
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
    if TM == 3:
        # per-element XOR words: K tiles are [D, N], V tiles [N, DV]
        bad = (_sig_bad(tl.reduce(xk_acc, 1, _xor), ck_acc, tl.load(elab_ptr + offs_d), bmask_ptr, E, B)
               | _sig_bad(tl.reduce(xv_acc, 0, _xor), cv_acc, tl.load(elab_ptr + offs_dv), bmask_ptr, E, B))
    return acc, e_sum, e_max, nk_fix, nk_bad, nv_fix, nv_bad, bad, n_cw


@triton.jit
def _sig_bad(w, c_acc, e_lab, bmask_ptr, E: tl.constexpr, B: tl.constexpr):
    """1 if the (p, S_e, S_b) syndrome of the XOR of all vectors differs from the XOR of their
    stored checkwords (the code is linear, so a clean split always matches).

    w: one XOR word per element (all vectors of the split folded). S_e (one popcount per
    element, once per split) catches what p and S_b alone miss: two flips at the same bit
    position in different elements, e.g. a 2-bit error (14, 14) in one vector or two 1-bit
    errors in two vectors of one split. What still cancels is the same element and bit
    flipped in two vectors of one split (an XOR of zero).
    """
    w = w.to(tl.int32) & 0xFFFF
    c = tl.reduce(w, 0, _xor)
    cw = tl.reduce(c_acc, 0, _xor).to(tl.int32) & 0xFFFF
    p1 = _par16(c)
    sb1 = _sb_of(c, bmask_ptr, B)
    se1 = tl.reduce(tl.where(_par16(w) == 1, e_lab, 0), 0, _xor)
    p0 = cw & 1
    se0 = (cw >> 1) & ((1 << E) - 1)
    sb0 = (cw >> (1 + E)) & ((1 << B) - 1)
    return ((p1 != p0) | (sb1 != sb0) | (se1 != se0)).to(tl.int32)


@triton.jit
def _fwd_grouped_kernel_stage1_secded(
    Q, K_Buffer, V_Buffer, K_CW, V_CW, sm_scale_withk, kv_indptr, kv_indices, Att_Out, Att_Lse,
    num_kv_splits, elab_ptr, bmask_ptr, inv_e_ptr, inv_b_ptr, err_ptr,
    stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh,
    stride_mid_ob, stride_mid_oh, stride_mid_os, H_KV,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr, Lk: tl.constexpr, Lv: tl.constexpr, E: tl.constexpr, B: tl.constexpr,
    CHECK: tl.constexpr, GATE: tl.constexpr = False, POPC: tl.constexpr = False,
    SAMPLE: tl.constexpr = 1, phase_ptr=None, SIG: tl.constexpr = False, WB: tl.constexpr = False,
    count_ptr=None, DEDUP: tl.constexpr = False, bad_ptr=None, REDO: tl.constexpr = False,
    SIGRED: tl.constexpr = False, log_ptr=None, logn_ptr=None, LOGCAP=0, layer=0,
    REP: tl.constexpr = False, rep_map=None, RK=None, RV=None, RKC=None, RVC=None,
):
    """SIG keeps only the cheap signature in this kernel and writes a per-program
    mismatch flag to bad_ptr; a second launch with REDO=True recomputes just the
    flagged splits with the full check (keeps the correction code, and its
    registers, out of the hot kernel)."""
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    kv_len_per_split = tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    flag_off = (cur_batch * tl.num_programs(1) + cur_head_id) * tl.num_programs(2) + split_kv_id
    run = split_kv_end > split_kv_start
    if REDO:
        if run:
            run = tl.load(bad_ptr + flag_off) > 0
    if run:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        q_k = q.to(K_Buffer.dtype.element_ty)
        if not CHECK:
            TM: tl.constexpr = 0
        elif REDO:
            TM: tl.constexpr = 1
        elif SIG:
            TM: tl.constexpr = 3
        elif GATE:
            TM: tl.constexpr = 2
        else:
            TM: tl.constexpr = 1
        acc, e_sum, e_max, nk_fix, nk_bad, nv_fix, nv_bad, bad, n_cw = _split(
            q_k, K_Buffer, V_Buffer, K_CW, V_CW, kv_indices, cur_batch_kv_start_idx, split_kv_start,
            split_kv_end, cur_kv_head, H_KV, sm_scale_withk, mask_h, offs_d, offs_dv, mask_d, mask_dv,
            stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh, elab_ptr, bmask_ptr, inv_e_ptr,
            inv_b_ptr, phase_ptr, BLOCK_N, BLOCK_H, BLOCK_DMODEL, BLOCK_DV, logit_cap, E, B, TM, POPC, SAMPLE, WB, count_ptr, DEDUP, SIGRED,
            log_ptr, logn_ptr, LOGCAP, layer, REP, rep_map, RK, RV, RKC, RVC,
            err_ptr + 10)
        if SIG and not REDO:
            # mismatch somewhere in this split: the REDO launch recomputes it with the full
            # per-vector check and correction, overwriting this program's outputs
            tl.store(bad_ptr + flag_off, bad)
        if REDO:
            tl.atomic_add(err_ptr + 8, 1)

        offs_mid_o = (cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh
                      + split_kv_id * stride_mid_os + offs_dv[None, :])
        tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None], mask=(mask_h[:, None]) & (mask_dv[None, :]))
        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)
        if CHECK:
            if n_cw > 0:
                tl.atomic_add(err_ptr + 9, n_cw)
            if nk_fix + nk_bad + nv_fix + nv_bad > 0:
                tl.atomic_add(err_ptr + 0, nk_fix)
                tl.atomic_add(err_ptr + 1, nk_bad)
                tl.atomic_add(err_ptr + 2, nv_fix)
                tl.atomic_add(err_ptr + 3, nv_bad)


@triton.jit
def _count_readers_kernel(kv_indices, kv_indptr, bs, count_ptr, owner_ptr, NPROG: tl.constexpr,
                          BLOCK: tl.constexpr):
    """count[slot] = readers of slot in this decode step; owner[slot] = first index reading it."""
    total = tl.load(kv_indptr + bs)
    for start in tl.range(tl.program_id(0) * BLOCK, total, NPROG * BLOCK):
        n = start + tl.arange(0, BLOCK)
        m = n < total
        slot = tl.load(kv_indices + n, mask=m, other=0).to(tl.int64)
        tl.atomic_add(count_ptr + slot, 1, mask=m)
        tl.atomic_min(owner_ptr + slot, n.to(tl.int32), mask=m)


@triton.jit
def _shared_prepass_kernel(buf_ptr, cw_ptr, kv_indices, kv_indptr, bs, count_ptr, owner_ptr, elab_ptr,
                           bmask_ptr, inv_e_ptr, inv_b_ptr, err_ptr, cwerr_ptr, stride_s, stride_h, H,
                           log_ptr, logn_ptr, LOGCAP, tag, NPROG: tl.constexpr, D: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    """Full check + in-place fix of each shared slot (count >= 2) once, by its owner index."""
    h = tl.program_id(1)
    total = tl.load(kv_indptr + bs)
    d = tl.arange(0, D)
    e_lab = tl.load(elab_ptr + d)
    for n in tl.range(tl.program_id(0), total, NPROG):
        slot = tl.load(kv_indices + n).to(tl.int64)
        if tl.load(count_ptr + slot) >= 2:
            if tl.load(owner_ptr + slot) == n:
                ptr = buf_ptr + slot * stride_s + h * stride_h
                x = tl.load(ptr + d).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                p1, se1, sb1 = _syndrome(x, e_lab, bmask_ptr, 0, B)
                cw = tl.load(cw_ptr + slot * H + h).to(tl.int32) & 0xFFFF
                st, el, bt = _status(p1, se1, sb1, cw, inv_e_ptr, inv_b_ptr, E, B)
                if st == 1:
                    xe = tl.load(ptr + el).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
                    tl.store(ptr + el, (xe ^ (1 << bt)).to(tl.int16).to(buf_ptr.dtype.element_ty, bitcast=True))
                    tl.atomic_add(err_ptr + 0, 1)
                if st == 3:
                    tl.atomic_add(err_ptr + 1, 1)
                    _log_bad(log_ptr, logn_ptr, LOGCAP, slot, h, tag, st == 3)
                if st == 2:
                    q1 = _par16(se1) ^ _par16(sb1)
                    good = p1 | (se1 << 1) | (sb1 << (1 + E)) | (q1 << (1 + E + B))
                    tl.store(cw_ptr + slot * H + h, good.to(tl.int16))
                    tl.atomic_add(cwerr_ptr, 1)


# ---- host side ---------------------------------------------------------------------------


def _tables(device):
    if _state["tables"] is None:
        code = VecCode(n_elem=128)
        import numpy as np

        inv_e = np.full(1 << code.e_bits, -1, np.int32)
        inv_e[code.e_labels] = np.arange(code.n_elem)
        inv_b = np.full(1 << code.b_bits, -1, np.int32)
        inv_b[code.b_labels] = np.arange(code.elem_bits)
        _state["code"] = code
        _state["tables"] = dict(
            elab=torch.tensor(code.e_labels, dtype=torch.int32, device=device),
            bmask=torch.tensor(code.b_masks.astype(np.int64), dtype=torch.int32, device=device),
            inv_e=torch.tensor(inv_e, device=device),
            inv_b=torch.tensor(inv_b, device=device),
        )
        # [8] sig redo, [9] checkword errors, [10] decode / [11] extend vectors restored from a replica
        _state["err"] = torch.zeros(12, dtype=torch.int32, device=device)
        _state["phase"] = torch.zeros(1, dtype=torch.int32, device=device)
        _state["log"] = torch.zeros(LOG_CAP * 3, dtype=torch.int32, device=device)
        _state["logn"] = torch.zeros(1, dtype=torch.int32, device=device)
    return _state["tables"]


def attach_pool(pool) -> None:
    """Allocate checkword buffers for every layer and register them by K/V buffer address."""
    dev = pool.k_buffer[0].device
    t = _tables(dev)
    n, h, d = pool.k_buffer[0].shape
    assert d == 128, "prototype assumes head_dim 128"
    pool.rm_k_cw = [torch.zeros(n, h, dtype=torch.int16, device=dev) for _ in pool.k_buffer]
    pool.rm_v_cw = [torch.zeros(n, h, dtype=torch.int16, device=dev) for _ in pool.v_buffer]
    for lid, (kb, vb, kc, vc) in enumerate(zip(pool.k_buffer, pool.v_buffer, pool.rm_k_cw, pool.rm_v_cw)):
        _state["by_ptr"][kb.data_ptr()] = (kc, vc, lid)
        # encode whatever the buffers hold now (normally zeros)
        all_loc = torch.arange(n, device=dev)
        encode_rows(kb, kc, all_loc)
        encode_rows(vb, vc, all_loc)
    _state["pools"][id(pool)] = pool
    _state["count"] = torch.zeros(n, dtype=torch.int32, device=dev)
    _state["owner"] = torch.zeros(n, dtype=torch.int32, device=dev)


def encode_rows(buf, cw, loc) -> None:
    t = _state["tables"]
    code = _state["code"]
    n = loc.numel()
    if n == 0:
        return
    _encode_scatter_kernel[(n, buf.shape[1])](
        buf, cw, loc, t["elab"], t["bmask"], buf.stride(0), buf.stride(1), buf.shape[1],
        D=buf.shape[2], E=code.e_bits, B=code.b_bits,
    )


def launch_stage1(q, k_buffer, v_buffer, k_cw, v_cw, att_out, att_lse, kv_indptr, kv_indices,
                  num_kv_splits, max_kv_splits, sm_scale_withk, logit_cap=0.0, check=True,
                  block_h=16, min_block_kv=32, mode="full", popc=None, sample=1, wb=False, dedup=False, sigred=False, layer=0,
                  cfg=None):
    """Grouped decode stage 1 with check-on-load; same outputs as SGLang's stage 1.

    dedup (sig mode only): ``shared_prepass`` must have run for this layer; slots
    with reader count >= 2 are then left out of the signature.
    """
    t = _tables(k_buffer.device)
    code = _state["code"]
    Lk, Lv = k_buffer.shape[-1], v_buffer.shape[-1]
    kv_head_num = k_buffer.shape[-2]
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num // kv_head_num
    grid = (batch, triton.cdiv(head_num, min(block_h, kv_group_num)), max_kv_splits)
    sig = check and mode == "sig"
    bad = torch.empty(grid[0] * grid[1] * grid[2], dtype=torch.int32, device=q.device) if sig else None
    _launch_stage1_grid(grid, q, k_buffer, v_buffer, k_cw, v_cw, att_out, att_lse, kv_indptr, kv_indices,
                        num_kv_splits, sm_scale_withk, logit_cap, check, block_h, min_block_kv, mode, popc,
                        sample, wb, dedup, bad, False, sigred, layer, cfg)
    if sig:
        _launch_stage1_grid(grid, q, k_buffer, v_buffer, k_cw, v_cw, att_out, att_lse, kv_indptr, kv_indices,
                            num_kv_splits, sm_scale_withk, logit_cap, check, block_h, min_block_kv, mode, popc,
                            1, wb, False, bad, True, False, layer)


def _launch_stage1_grid(grid, q, k_buffer, v_buffer, k_cw, v_cw, att_out, att_lse, kv_indptr, kv_indices,
                        num_kv_splits, sm_scale_withk, logit_cap, check, block_h, min_block_kv, mode, popc,
                        sample, wb, dedup, bad, redo, sigred=False, layer=0, cfg=None):
    # cfg: launch config of the main (non-redo) kernel, e.g. {"num_warps": 8, "num_stages": 1,
    # "block_n": 64}; SGLang's grouped decode uses (4, 2, 32)
    # default (4,1,32): RQ3 week-8 sweep, stage 1 1.79x -> 1.13x of SGLang at gsp c32, TPOT c8/c32 +2.4/+2.2%
    c = {"num_warps": 4, "num_stages": 1, "block_n": 32, **(cfg or {})}
    t = _state["tables"]
    code = _state["code"]
    Lk, Lv = k_buffer.shape[-1], v_buffer.shape[-1]
    kv_head_num = k_buffer.shape[-2]
    head_num = q.shape[1]
    kv_group_num = head_num // kv_head_num
    h = _fwd_grouped_kernel_stage1_secded[grid](
        q, k_buffer, v_buffer, k_cw, v_cw, sm_scale_withk, kv_indptr, kv_indices, att_out, att_lse,
        num_kv_splits, t["elab"], t["bmask"], t["inv_e"], t["inv_b"], _state["err"],
        q.stride(0), q.stride(1), k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        att_out.stride(0), att_out.stride(1), att_out.stride(2), kv_head_num,
        kv_group_num=kv_group_num, q_head_num=head_num, BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_DV=triton.next_power_of_2(Lv), BLOCK_N=c["block_n"], BLOCK_H=block_h,
        MIN_BLOCK_KV=min_block_kv, logit_cap=logit_cap, Lk=Lk, Lv=Lv,
        E=code.e_bits, B=code.b_bits, CHECK=check, GATE=(mode == "gate"), SIG=(mode == "sig"),
        POPC=(k_buffer.is_cuda if popc is None else popc), SAMPLE=sample, WB=wb,
        count_ptr=_state["count"] if dedup else None, DEDUP=bool(dedup and mode == "sig"),
        bad_ptr=bad, REDO=redo, SIGRED=bool(sigred and mode == "sig"),
        log_ptr=_state["log"], logn_ptr=_state["logn"], LOGCAP=LOG_CAP, layer=layer,
        phase_ptr=_state["phase"], num_warps=c["num_warps"], num_stages=c["num_stages"],
        **_rep_args(k_buffer, v_buffer, layer, check and (redo or mode != "sig")),
    )
    # registers / spills of the compiled kernel (GPU only), for the RQ3 bench
    _state.setdefault("kinfo", {})[("redo" if redo else mode, bool(wb), bool(dedup), bool(sigred))] = (
        getattr(h, "n_regs", None), getattr(h, "n_spills", None))


def _rep_args(k_buffer, v_buffer, layer: int, on: bool) -> dict:
    """Kernel arguments of the module D replica of ``layer`` (REP off when there is none)."""
    rep = _state.get("rep")
    if not on or rep is None or layer not in rep["layers"]:
        return {"REP": False}
    rk, rv, rkc, rvc = rep["layers"][layer]
    assert rk.stride() == k_buffer.stride() and rv.stride() == v_buffer.stride(), "replica layout must match the pool"
    return {"REP": True, "rep_map": rep["map"], "RK": rk, "RV": rv, "RKC": rkc, "RVC": rvc}


_NPROG = 512
# pre-extend check: vectors per program (RADIXMEND_SECDED_EXT_BLOCK, 0 = one per vector, the old path)
EXT_BLOCK = int(os.environ.get("RADIXMEND_SECDED_EXT_BLOCK", "16"))


def count_readers(kv_indices, kv_indptr) -> None:
    """Once per decode step (same kv_indices for every layer); graph capturable."""
    _state["count"].zero_()
    _state["owner"].fill_(2 ** 31 - 1)
    _count_readers_kernel[(_NPROG,)](kv_indices, kv_indptr, kv_indptr.numel() - 1, _state["count"],
                                     _state["owner"], NPROG=_NPROG, BLOCK=128)


def shared_prepass(k_buffer, v_buffer, k_cw, v_cw, kv_indices, kv_indptr, layer: int = 0) -> None:
    """Check every slot read by >= 2 requests once for this layer, fixing in place (decode counters)."""
    t = _tables(k_buffer.device)
    code = _state["code"]
    for buf, cw, off in ((k_buffer, k_cw, 0), (v_buffer, v_cw, 2)):
        _shared_prepass_kernel[(_NPROG, buf.shape[1])](
            buf, cw, kv_indices, kv_indptr, kv_indptr.numel() - 1, _state["count"], _state["owner"],
            t["elab"], t["bmask"], t["inv_e"], t["inv_b"], _state["err"][off:], _state["err"][9:],
            buf.stride(0), buf.stride(1), buf.shape[1], _state["log"], _state["logn"], LOG_CAP,
            layer * 4 + off, NPROG=_NPROG, D=buf.shape[2], E=code.e_bits,
            B=code.b_bits,
        )


def check_fix_inplace(buf, cw, kv_indices, kv_indptr, err_off: int, tag: int = 1, rep_buf=None,
                      rep_cw=None, block: int | None = None) -> None:
    """Check the prefix slots kv_indices[:kv_indptr[-1]] of one layer buffer, fix in place
    (with ``rep_buf``/``rep_cw``: restore uncorrectable vectors that have a module D replica)."""
    t = _tables(buf.device)
    code = _state["code"]
    n = kv_indices.numel()
    if n == 0:
        return
    rep = ({"REP": True, "rep_map": _state["rep"]["map"], "rep_buf": rep_buf, "rep_cw": rep_cw,
            "rest_ptr": _state["err"][11:]} if rep_buf is not None else {})
    args = (buf, cw, kv_indices, kv_indptr, kv_indptr.numel() - 1, t["elab"], t["bmask"], t["inv_e"],
            t["inv_b"], _state["err"][err_off:], _state["err"][9:], buf.stride(0), buf.stride(1), buf.shape[1],
            _state["log"], _state["logn"], LOG_CAP, tag)
    blk = EXT_BLOCK if block is None else block
    if blk <= 0:  # reference: one program per vector
        _check_fix_inplace_kernel[(n, buf.shape[1])](*args, D=buf.shape[2], E=code.e_bits, B=code.b_bits, **rep)
        return
    _check_fix_blocked_kernel[(triton.cdiv(n, blk), buf.shape[1])](
        *args, BLOCK=blk, D=buf.shape[2], E=code.e_bits, B=code.b_bits, POPC=buf.is_cuda, **rep)


def check_prefix(k_buffer, v_buffer, kv_indices, kv_indptr) -> bool:
    """Pre-extend check of a pool layer; False if the buffers are not a registered pool layer."""
    cws = _state["by_ptr"].get(k_buffer.data_ptr())
    if cws is None or k_buffer.dim() != 3:
        return False
    lid = cws[2]
    rep = _state.get("rep")
    r = rep["layers"].get(lid) if rep is not None else None
    check_fix_inplace(k_buffer, cws[0], kv_indices, kv_indptr, 4, lid * 4 + 1,  # always the full check
                      *((r[0], r[2]) if r else ()))
    check_fix_inplace(v_buffer, cws[1], kv_indices, kv_indptr, 6, lid * 4 + 3, *((r[1], r[3]) if r else ()))
    return True


def errors() -> list[int]:
    return [] if _state["err"] is None else [int(x) for x in _state["err"].tolist()]


def error_log(reset: bool = False) -> dict:
    """Uncorrectable (2-bit) detections: slot, kv head, layer, K/V, path. For module C."""
    if _state.get("log") is None:
        return {"total": 0, "entries": []}
    n = int(_state["logn"].item())
    rows = _state["log"][: 3 * min(n, LOG_CAP)].view(-1, 3).tolist()
    out = {"total": n, "dropped": max(0, n - LOG_CAP),
           "entries": [{"slot": s, "head": h, "layer": t >> 2, "kind": "v" if t & 2 else "k",
                        "path": "extend" if t & 1 else "decode"} for s, h, t in rows]}
    if reset:
        _state["logn"].zero_()
    return out


def log_fake(entries) -> int:
    """Append (slot, head, tag) rows to the error log as if the check had found them
    uncorrectable, without touching the data (control runs for module C)."""
    if _state.get("log") is None:
        raise RuntimeError("error log missing: start the server with RADIXMEND_SECDED=1")
    n = int(_state["logn"].item())
    rows = [e for e in entries][: max(0, LOG_CAP - n)]
    if rows:
        t = torch.tensor(rows, dtype=torch.int32, device=_state["log"].device).flatten()
        _state["log"][3 * n: 3 * n + t.numel()] = t
        _state["logn"] += len(rows)
    return len(rows)


def install() -> None:
    if os.environ.get("RADIXMEND_SECDED") != "1":
        return
    from sglang.kernels.ops.attention import decode_attention as da
    from sglang.srt.mem_cache import memory_pool as mp

    if getattr(mp.MHATokenToKVPool, "_rm_secded", False):
        return
    check = os.environ.get("RADIXMEND_SECDED_CHECK", "1") == "1"
    # decode check variant: full (every tile, full syndrome), gate (parity/bit-syndrome
    # gate before the element syndrome), sig (one signature per split, redo the split
    # with the full check on mismatch); RADIXMEND_SECDED_SAMPLE=P checks each tile
    # once every P decode steps. Extend always runs the full check.
    mode = os.environ.get("RADIXMEND_SECDED_MODE", "full")
    assert mode in ("full", "gate", "sig"), mode
    sample = int(os.environ.get("RADIXMEND_SECDED_SAMPLE", "1"))
    popc = {"1": True, "0": False}.get(os.environ.get("RADIXMEND_SECDED_POPC", ""), None)
    # RADIXMEND_SECDED_WB=1: decode writes corrected words back to the KV buffer, so a
    # fault is fixed once instead of on every step (and cannot pair up with a later flip)
    wb = os.environ.get("RADIXMEND_SECDED_WB") == "1"
    # RADIXMEND_SECDED_DEDUP=1 (sig mode): slots read by several requests in one decode
    # step (a shared prompt) are checked once per layer by a pre-pass, not once per request
    dedup = os.environ.get("RADIXMEND_SECDED_DEDUP") == "1" and mode == "sig" and check
    sigred = os.environ.get("RADIXMEND_SECDED_SIGRED") == "1"
    # RADIXMEND_SECDED_CFG="warps,stages,block_n" for the main decode kernel (RQ3 tuning)
    cfg = None
    if os.environ.get("RADIXMEND_SECDED_CFG"):
        w, st, bn = map(int, os.environ["RADIXMEND_SECDED_CFG"].split(","))
        cfg = {"num_warps": w, "num_stages": st, "block_n": bn}

    orig_init = mp.MHATokenToKVPool.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        attach_pool(self)

    orig_set = mp.MHATokenToKVPool.set_kv_buffer

    def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, *args, **kwargs):
        orig_set(self, layer, loc_info, cache_k, cache_v, *args, **kwargs)
        loc, _, _ = mp.unwrap_write_loc(loc_info)
        ov = kwargs.get("layer_id_override")
        lid = (ov if ov is not None else layer.layer_id) - self.start_layer
        encode_rows(self.k_buffer[lid], self.rm_k_cw[lid], loc)
        encode_rows(self.v_buffer[lid], self.rm_v_cw[lid], loc)

    orig_grouped = da._decode_grouped_att_m_fwd

    def _decode_grouped_att_m_fwd(q, k_buffer, v_buffer, att_out, att_lse, kv_indptr, kv_indices,
                                  num_kv_splits, max_kv_splits, sm_scale_withk, logit_cap,
                                  xai_temperature_len=-1, has_mla=False, use_pdl=False, page_size=1,
                                  score_mod=None, aux_tensors=None, tune_mla=False, forced_kv_splits=0):
        cws = _state["by_ptr"].get(k_buffer.data_ptr())
        if (cws is None or has_mla or page_size != 1 or score_mod is not None or forced_kv_splits
                or xai_temperature_len > 0 or k_buffer.dim() != 3):
            return orig_grouped(q, k_buffer, v_buffer, att_out, att_lse, kv_indptr, kv_indices, num_kv_splits,
                                max_kv_splits, sm_scale_withk, logit_cap, xai_temperature_len, has_mla=has_mla,
                                use_pdl=use_pdl, page_size=page_size, score_mod=score_mod,
                                aux_tensors=aux_tensors, tune_mla=tune_mla, forced_kv_splits=forced_kv_splits)
        k_cw, v_cw, lid = cws
        if sample > 1 and lid == 0:
            _state["phase"].add_(1)  # one step per forward; capturable in CUDA graphs
        if dedup:
            if lid == 0:
                count_readers(kv_indices, kv_indptr)
            shared_prepass(k_buffer, v_buffer, k_cw, v_cw, kv_indices, kv_indptr, lid)
        launch_stage1(q, k_buffer, v_buffer, k_cw, v_cw, att_out, att_lse, kv_indptr, kv_indices,
                      num_kv_splits, max_kv_splits, sm_scale_withk, logit_cap, check,
                      block_h=da._GROUPED_BLOCK_H, min_block_kv=da._MIN_BLOCK_KV,
                      mode=mode, popc=popc, sample=sample, wb=wb, dedup=dedup,
                      sigred=sigred, layer=lid, cfg=cfg)

    from sglang.kernels.ops.attention import extend_attention as ea

    orig_ext = ea.extend_attention_fwd

    def extend_attention_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr,
                             kv_indptr, kv_indices, *args, **kwargs):
        if check and kwargs.get("page_size", 1) == 1 and not kwargs.get("skip_prefix", False):
            check_prefix(k_buffer, v_buffer, kv_indices, kv_indptr)
        return orig_ext(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr,
                        kv_indptr, kv_indices, *args, **kwargs)

    orig_ext_u = ea.extend_attention_fwd_unified

    def extend_attention_fwd_unified(q, o, k_buffer, v_buffer, k_scale, v_scale, qo_indptr, kv_indptr,
                                     kv_indices, *args, **kwargs):
        if check and kwargs.get("page_size", 1) == 1:
            check_prefix(k_buffer, v_buffer, kv_indices, kv_indptr)
        return orig_ext_u(q, o, k_buffer, v_buffer, k_scale, v_scale, qo_indptr, kv_indptr, kv_indices,
                          *args, **kwargs)

    ea.extend_attention_fwd = extend_attention_fwd
    ea.extend_attention_fwd_unified = extend_attention_fwd_unified
    mp.MHATokenToKVPool.__init__ = __init__
    mp.MHATokenToKVPool.set_kv_buffer = set_kv_buffer
    da._decode_grouped_att_m_fwd = _decode_grouped_att_m_fwd
    mp.MHATokenToKVPool._rm_secded = True
