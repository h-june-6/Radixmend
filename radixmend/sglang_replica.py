"""Module D (minimal): replicas of shared prefix KV, used to repair errors module A
can detect but not correct, without recomputing anything.

Inside SGLang v0.5.22, on top of ``sglang_secded`` (module A) and, for vectors
without a replica, ``sglang_recover`` (module C).

* Replica store: a second, smaller KV area with the pool's layout (K, V and the
  module A checkwords for every layer) of ``RADIXMEND_REPLICA_TOKENS`` slots, and
  ``map[slot]`` = replica slot or -1.
* What gets a replica: a radix-tree prefix once it is reused, i.e. on a prefix
  match of at least ``RADIXMEND_REPLICA_MIN`` tokens (default 32), the matched
  slots (the first ``RADIXMEND_REPLICA_SKIP`` positions left out) while there is
  room. Before copying, every vector is checked and single-bit errors are fixed
  in place, so the copy starts clean; a slot found uncorrectable is not copied
  (module C handles it).
* Use: where module A finds an uncorrectable vector (decode check, its redo
  launch in sig mode, or the pre-extend prefix check), it takes the vector from
  the replica instead, for the attention of this step and, with write-back, into
  the pool (data and checkword). Nothing is logged for module C, so no request
  is retracted and outputs stay bitwise equal to a fault-free run.
* Invalidation: when a slot receives new K/V (it was freed and reallocated) its
  replica is dropped. This runs on the GPU without a host sync and does not
  depend on host state, so it is also correct inside captured CUDA graphs.

Enable with ``RADIXMEND_REPLICA=1`` (needs ``RADIXMEND_SECDED=1``; overlap
scheduling is fine for D itself, module C needs it disabled).
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)

_state = {"stats": {"replicated": 0, "replica_full": 0, "skipped_bad": 0, "invalidated_calls": 0,
                    "match_skipped": 0}}


def attach_pool(pool, capacity: int) -> None:
    from radixmend import sglang_secded as ss

    dev = pool.k_buffer[0].device
    n = pool.k_buffer[0].shape[0]
    _state["seen"] = set()  # a new store: every node needs its replicas again
    _state["full"] = None  # host hint: None = room may be left, else the pending free-count refresh
    layers = {}
    for lid, (kb, vb) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
        layers[lid] = (torch.zeros((capacity,) + tuple(kb.shape[1:]), dtype=kb.dtype, device=dev),
                       torch.zeros((capacity,) + tuple(vb.shape[1:]), dtype=vb.dtype, device=dev),
                       torch.zeros(capacity, kb.shape[1], dtype=torch.int16, device=dev),
                       torch.zeros(capacity, vb.shape[1], dtype=torch.int16, device=dev))
    ss._state["rep"] = {
        "map": torch.full((n,), -1, dtype=torch.int32, device=dev),
        "owner": torch.full((capacity,), -1, dtype=torch.int64, device=dev),
        "layers": layers,
    }
    _state["pool"] = pool
    _state["capacity"] = capacity
    mb = sum(t.numel() * t.element_size() for ts in layers.values() for t in ts) / 2**20
    log.warning("radixmend: module D replica store %d tokens (%.0f MiB)", capacity, mb)


def rep() -> dict | None:
    from radixmend import sglang_secded as ss

    return ss._state.get("rep")


def invalidate(loc: torch.Tensor) -> None:
    """Drop the replicas of slots that are being overwritten (no host sync, no host tensors)."""
    r = rep()
    if r is None or loc.numel() == 0:
        return
    loc = loc.to(torch.int64)
    rs = r["map"][loc].to(torch.int64)
    # owner[rs] = -1 where rs >= 0; rows with rs < 0 rewrite owner[0] with its own value
    idx = rs.clamp(min=0)
    r["owner"].index_put_((idx,), torch.where(rs >= 0, torch.full_like(rs, -1), r["owner"][idx]))
    r["map"].index_fill_(0, loc, -1)  # no host scalar tensor: capturable in CUDA graphs


def replicate(pool, slots: torch.Tensor, distinct: bool = False) -> int:
    """Copy (after a check + single-bit fix) the slots that have no replica yet; returns how many.

    ``distinct=True``: the caller guarantees no repeated slot (a tree path), so torch.unique,
    a host sync (~11 ms per call under overlap scheduling), is skipped.
    """
    from radixmend import sglang_secded as ss

    r = rep()
    if r is None or slots.numel() == 0:
        return 0
    if store_full_hint(r):
        _state["stats"]["replica_full"] += 1
        return 0
    slots = slots.to(torch.int64)
    if not distinct:
        slots = torch.unique(slots)
    need = slots[r["map"][slots] < 0]
    if need.numel() == 0:
        return 0
    free = torch.nonzero(r["owner"] < 0).flatten()
    if free.numel() < need.numel():
        _state["stats"]["replica_full"] += 1
        need = need[: free.numel()]
        if need.numel() == 0:
            _state["full"] = {}  # no room: skip further calls until a refresh shows freed rows
            return 0
    free = free[: need.numel()]
    # verify the source first (fixes 1-bit errors in place; logs uncorrectable ones)
    n0 = int(ss._state["logn"].item())
    indptr = torch.tensor([0, need.numel()], dtype=torch.int32, device=need.device)
    for lid in range(len(pool.k_buffer)):
        ss.check_fix_inplace(pool.k_buffer[lid], pool.rm_k_cw[lid], need, indptr, 4, lid * 4 + 1)
        ss.check_fix_inplace(pool.v_buffer[lid], pool.rm_v_cw[lid], need, indptr, 6, lid * 4 + 3)
    if int(ss._state["logn"].item()) != n0:
        bad = ss._state["log"][3 * n0: 3 * min(int(ss._state["logn"].item()), ss.LOG_CAP)].view(-1, 3)[:, 0]
        keep = ~torch.isin(need, bad.to(torch.int64))
        _state["stats"]["skipped_bad"] += int((~keep).sum().item())
        need, free = need[keep], free[: int(keep.sum().item())]
    for lid, (rk, rv, rkc, rvc) in r["layers"].items():
        rk[free] = pool.k_buffer[lid][need]
        rv[free] = pool.v_buffer[lid][need]
        rkc[free] = pool.rm_k_cw[lid][need]
        rvc[free] = pool.rm_v_cw[lid][need]
    r["map"][need] = free.to(torch.int32)
    r["owner"][free] = need
    _state["stats"]["replicated"] += int(need.numel())
    return int(need.numel())


def store_full_hint(r) -> bool:
    """True while the store is known full, without a host sync on the match path.

    Rows are freed only on the GPU (``invalidate``), so once the store was seen full the free
    count is refreshed by an async copy into pinned memory, read on a later call when ready.
    Before this, every match of a new group paid torch.unique/mask/nonzero syncs (~11 ms each
    under overlap scheduling) just to find no room: +20-30% TTFT on 864-token shared prompts.
    """
    st = _state.get("full")
    if st is None:
        return False
    if "host" in st:
        if st["ev"] is None or st["ev"].query():
            nfree = int(st["host"].item())  # host memory, already copied: no device sync
            _state["full"] = None if nfree > 0 else {}
            if nfree > 0:
                return False
        else:
            return True
    cnt = (r["owner"] < 0).sum()
    if cnt.is_cuda:
        host = torch.empty((), dtype=cnt.dtype, pin_memory=True)
        host.copy_(cnt, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
    else:
        host, ev = cnt, None
    _state["full"] = {"host": host, "ev": ev}
    if ev is None or ev.query():
        nfree = int(host.item())
        _state["full"] = None if nfree > 0 else {}
        return nfree == 0
    return True


_SEEN_CAP = 1 << 20


def first_visit(node) -> bool:
    """True the first time a matched tree node is seen; later matches of it skip replicate().

    The scheduler re-matches every waiting request each step, and replicate() needs several
    host syncs (unique, masks, item), which stall the overlap pipeline: ~60 ms per request on
    short shared prompts. A node's slots stay put while the node lives (eviction frees them,
    and the replicas are dropped on GPU when the slots are rewritten), so once per node is
    enough. Node ids are a never-reused counter; nodes without one are always processed.
    (b6840e0 read ``node.id`` from what is already an int id, so it never skipped.)
    """
    # SGLang 0.5.22's unified tree returns node ids (int) as last_device_node, not nodes
    nid = node if isinstance(node, int) else getattr(node, "id", None)
    if nid is None:
        return True
    seen = _state.setdefault("seen", set())
    if nid in seen:
        _state["stats"]["match_skipped"] += 1
        return False
    if len(seen) >= _SEEN_CAP:
        seen.clear()
    seen.add(nid)
    return True


def stats(reset: bool = False) -> dict:
    from radixmend import sglang_secded as ss

    r = rep()
    err = ss.errors()
    out = dict(_state["stats"],
               replicas_now=int((r["owner"] >= 0).sum().item()) if r is not None else 0,
               capacity=_state.get("capacity", 0),
               restored_decode=err[10] if len(err) > 10 else 0,
               restored_extend=err[11] if len(err) > 11 else 0)
    if reset:
        for k in _state["stats"]:
            _state["stats"][k] = 0
        if ss._state["err"] is not None:
            ss._state["err"][10:].zero_()
    return out


def install() -> None:
    if os.environ.get("RADIXMEND_REPLICA") != "1":
        return
    from sglang.srt.mem_cache import memory_pool as mp
    from sglang.srt.mem_cache import unified_radix_cache as urc

    if getattr(urc.UnifiedRadixCache, "_rm_replica", False):
        return
    capacity = int(os.environ.get("RADIXMEND_REPLICA_TOKENS", "2048"))
    min_len = int(os.environ.get("RADIXMEND_REPLICA_MIN", "32"))
    skip = int(os.environ.get("RADIXMEND_REPLICA_SKIP", "0"))

    orig_init = mp.MHATokenToKVPool.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)  # module A has attached its checkwords by now
        attach_pool(self, capacity)

    orig_set = mp.MHATokenToKVPool.set_kv_buffer

    def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, *args, **kwargs):
        orig_set(self, layer, loc_info, cache_k, cache_v, *args, **kwargs)
        ov = kwargs.get("layer_id_override")
        if (ov if ov is not None else layer.layer_id) == self.start_layer:
            loc, _, _ = mp.unwrap_write_loc(loc_info)
            invalidate(loc)

    orig_match = urc.UnifiedRadixCache.match_prefix

    def match_prefix(self, params):
        res = orig_match(self, params)
        n = res.device_prefix_len
        if n >= min_len and n > skip and _state.get("pool") is not None and first_visit(res.last_device_node):
            extra_key = getattr(params.key, "extra_key", None)
            slots = self.tree_core.collect_full_device_indices(res.last_device_node, self.root_node_handle(extra_key))
            replicate(_state["pool"], slots[skip:n], distinct=True)
        return res

    mp.MHATokenToKVPool.__init__ = __init__
    mp.MHATokenToKVPool.set_kv_buffer = set_kv_buffer
    urc.UnifiedRadixCache.match_prefix = match_prefix
    urc.UnifiedRadixCache._rm_replica = True
