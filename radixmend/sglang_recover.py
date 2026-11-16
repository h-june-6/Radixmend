"""Module C (minimal): recovery from errors module A can detect but not correct.

Inside SGLang v0.5.22, on top of ``sglang_secded`` (module A). Requires
``--disable-overlap-schedule`` (the result of a step is processed before the
next step is scheduled).

When module A finds an uncorrectable (2-bit) vector it appends (slot, head,
layer, K/V, path) to its error log. After every forward this module:

1. reads the log (one 4-byte device read; the scheduler already waits for the
   step's outputs at this point);
2. finds the requests of the batch whose KV rows contain a bad slot. Their
   output of this step is discarded (they are marked retracted before SGLang
   commits the step, so no token is appended, nothing is streamed and, for a
   prefill, nothing is checkpointed into the tree: KV derived from the bad
   prefix never enters the tree);
3. retracts those requests (their non-tree KV is freed, tree locks are
   released) and puts them back in the waiting queue, so they are recomputed;
4. poisons the bad slots. A prefix match that would reuse a poisoned slot is
   turned into a full miss, and when the recomputed prefix is inserted again
   (SGLang keeps the tree's old slots for an existing key), the fresh K/V and
   checkwords are copied into the old slots and the poison is cleared.

So a 2-bit error costs a recompute of the affected requests instead of a
silently wrong output, and derived contamination cannot reach the tree.

Enable with ``RADIXMEND_RECOVER=1`` (needs ``RADIXMEND_SECDED=1``). This module
consumes module A's error log every step, so the ``secded_log`` hook op sees it
empty; use ``recover_stats`` instead. Limits of the minimal version: a bad slot
inside a request that is still in chunked prefill is not retracted (counted in
``chunked_skipped``), and overlap scheduling is not supported.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)

_state = {
    "poison": None, "n_poisoned": 0, "pending": [],
    "stats": {"detections": 0, "bad_slots": 0, "reqs_retracted": 0, "matches_blocked": 0,
              "tokens_repaired": 0, "steps_with_errors": 0,
              "chunked_skipped": 0},
}


def attach_pool(pool) -> None:
    n = pool.k_buffer[0].shape[0]
    _state["poison"] = torch.zeros(n, dtype=torch.bool, device=pool.k_buffer[0].device)
    _state["pool"] = pool


def new_bad_slots() -> torch.Tensor | None:
    """Slots logged as uncorrectable since the last call (log is reset), or None."""
    from radixmend import sglang_secded as ss

    logn = ss._state.get("logn")
    if logn is None:
        return None
    n = int(logn.item())
    if n == 0:
        return None
    k = min(n, ss.LOG_CAP)
    slots = ss._state["log"][: 3 * k].view(-1, 3)[:, 0].to(torch.int64)
    if n > ss.LOG_CAP:
        log.warning("radixmend: error log overflow (%d > %d entries)", n, ss.LOG_CAP)
    logn.zero_()
    _state["stats"]["detections"] += n
    return torch.unique(slots)


def affected_rows(req_to_token: torch.Tensor, pool_idx: torch.Tensor, seq_lens: torch.Tensor,
                  bad: torch.Tensor) -> list[int]:
    """Batch positions whose first seq_len slots contain a bad slot."""
    if pool_idx.numel() == 0:
        return []
    width = int(seq_lens.max().item())
    rows = req_to_token[pool_idx.to(torch.int64), :width].to(torch.int64)
    valid = torch.arange(width, device=rows.device)[None, :] < seq_lens.to(rows.device)[:, None]
    hit = (torch.isin(rows, bad.to(rows.device)) & valid).any(1)
    return [int(i) for i in torch.nonzero(hit).flatten().tolist()]


def poison(slots: torch.Tensor) -> None:
    _state["poison"][slots] = True
    _state["n_poisoned"] = int(_state["poison"].sum().item())
    _state["stats"]["bad_slots"] += int(slots.numel())


def unpoison_written(loc: torch.Tensor) -> None:
    """A slot that gets new K/V is no longer poisoned (reallocated after a free)."""
    if _state["n_poisoned"]:
        _state["poison"].index_fill_(0, loc.to(torch.int64), False)


def first_poisoned(slots: torch.Tensor) -> int:
    """Index of the first poisoned slot in ``slots`` or -1 (one host sync)."""
    if not _state["n_poisoned"] or slots.numel() == 0:
        return -1
    hit = _state["poison"][slots.to(torch.int64)]
    if not bool(hit.any().item()):
        return -1
    return int(torch.nonzero(hit)[0].item())


def write_back(pool, old: torch.Tensor, new: torch.Tensor) -> int:
    """Copy K/V and module A checkwords of every layer new -> old; clears old's poison."""
    old = old.to(torch.int64)
    new = new.to(torch.int64)
    keep = old != new
    old, new = old[keep], new[keep]
    if old.numel() == 0:
        return 0
    for lid in range(len(pool.k_buffer)):
        pool.k_buffer[lid][old] = pool.k_buffer[lid][new]
        pool.v_buffer[lid][old] = pool.v_buffer[lid][new]
        for name in ("rm_k_cw", "rm_v_cw"):
            cws = getattr(pool, name, None)
            if cws is not None:
                cws[lid][old] = cws[lid][new]
    _state["poison"][old] = False
    _state["n_poisoned"] = int(_state["poison"].sum().item())
    return int(old.numel())


def repair_on_insert(pool, token_ids, value) -> int:
    done = 0
    for key, old in list(_state["pending"]):
        n = len(key)
        if value is None or len(value) < n or len(token_ids) < n or tuple(token_ids[:n]) != key:
            continue
        done += write_back(pool, old, value[:n].to(old.device))
        _state["pending"].remove((key, old))
    _state["stats"]["tokens_repaired"] += done
    return done


def stats(reset: bool = False) -> dict:
    out = dict(_state["stats"], poisoned_now=_state["n_poisoned"], pending=len(_state["pending"]))
    if reset:
        for k in _state["stats"]:
            _state["stats"][k] = 0
    return out


def handle_step(sched, batch, run_orig) -> None:
    """Wrap one process_batch_result call: discard + retract requests that read a bad slot."""
    bad = new_bad_slots()
    if bad is None or not batch.reqs:
        run_orig()
        return
    st = _state["stats"]
    st["steps_with_errors"] += 1
    idx = affected_rows(sched.req_to_token_pool.req_to_token, batch.req_pool_indices, batch.seq_lens, bad)
    poison(bad)
    chunked = getattr(sched, "chunked_req", None)
    victims = [batch.reqs[i] for i in idx if not batch.reqs[i].finished() and batch.reqs[i] is not chunked]
    if chunked is not None and any(batch.reqs[i] is chunked for i in idx):
        st["chunked_skipped"] = st.get("chunked_skipped", 0) + 1
        log.warning("radixmend: bad slot in a chunked-prefill request; not retracted")
    proc = sched.batch_result_processor
    old_overlap = proc.enable_overlap
    for r in victims:
        r.is_retracted = True  # SGLang then skips the step's commit for these requests
    if victims and batch.forward_mode.is_decode():
        # the decode loop honours is_retracted only in overlap mode; the processor is a
        # frozen dataclass, so bypass its __setattr__ (the flag is read only at that check)
        object.__setattr__(proc, "enable_overlap", True)
    try:
        run_orig()
    finally:
        if proc.enable_overlap != old_overlap:
            object.__setattr__(proc, "enable_overlap", old_overlap)
    if not victims:
        return
    for r in victims:
        r.is_retracted = False
    keep = [i for i, r in enumerate(batch.reqs) if r not in victims]
    pos = {id(r): i for i, r in enumerate(batch.reqs)}
    for n, r in enumerate(victims):
        batch.release_req(pos[id(r)], len(victims) - n, offload_kv=False)
    batch.filter_batch(keep_indices=keep)
    for r in victims:
        sched._add_request_to_queue(r, is_retracted=True)
    st["reqs_retracted"] += len(victims)
    log.warning("radixmend: %d uncorrectable slot(s), retracted %d request(s)", int(bad.numel()), len(victims))


def install() -> None:
    if os.environ.get("RADIXMEND_RECOVER") != "1":
        return
    from sglang.srt.managers import scheduler as sm
    from sglang.srt.mem_cache import memory_pool as mp
    from sglang.srt.mem_cache import unified_radix_cache as urc
    from sglang.srt.mem_cache.base_prefix_cache import zero_match_result

    if getattr(urc.UnifiedRadixCache, "_rm_recover", False):
        return

    orig_init = mp.MHATokenToKVPool.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        attach_pool(self)

    orig_set = mp.MHATokenToKVPool.set_kv_buffer

    def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, *args, **kwargs):
        orig_set(self, layer, loc_info, cache_k, cache_v, *args, **kwargs)
        ov = kwargs.get("layer_id_override")
        if (ov if ov is not None else layer.layer_id) == self.start_layer:
            loc, _, _ = mp.unwrap_write_loc(loc_info)
            unpoison_written(loc)

    orig_match = urc.UnifiedRadixCache.match_prefix

    def match_prefix(self, params):
        res = orig_match(self, params)
        n = res.device_prefix_len
        if n <= 0 or not _state["n_poisoned"]:
            return res
        extra_key = getattr(params.key, "extra_key", None)
        slots = self.tree_core.collect_full_device_indices(res.last_device_node, self.root_node_handle(extra_key))
        slots = slots[:n]
        if first_poisoned(slots) < 0:
            return res
        _state["stats"]["matches_blocked"] += 1
        key = tuple(int(t) for t in list(params.key.token_ids)[:n])
        if all(k != key for k, _ in _state["pending"]):
            _state["pending"].append((key, slots.clone()))
        return zero_match_result(self, res, extra_key=extra_key)

    orig_insert = urc.UnifiedRadixCache.insert

    def insert(self, params):
        if _state["pending"] and params.key is not None and params.value is not None:
            repair_on_insert(_state["pool"], list(params.key.token_ids), params.value)
        return orig_insert(self, params)

    orig_pbr = sm.Scheduler.process_batch_result

    def process_batch_result(self, batch, result):
        if getattr(self, "enable_overlap", False):
            return orig_pbr(self, batch, result)
        if getattr(result, "copy_done", None) is not None:
            result.copy_done.synchronize()
        handle_step(self, batch, lambda: orig_pbr(self, batch, result))

    mp.MHATokenToKVPool.__init__ = __init__
    mp.MHATokenToKVPool.set_kv_buffer = set_kv_buffer
    urc.UnifiedRadixCache.match_prefix = match_prefix
    urc.UnifiedRadixCache.insert = insert
    sm.Scheduler.process_batch_result = process_batch_result
    urc.UnifiedRadixCache._rm_recover = True
