"""Hit-time hash baseline (prior work: hash shared KV blocks, verify on a cache
hit, recompute on mismatch) inside SGLang v0.5.22.

How it maps onto SGLang:
* digest-at-store: every KV slot gets one 64-bit digest over all layers' K and
  V rows, built while ``set_kv_buffer`` writes them (layer 0 resets the slot's
  digest, later layers add their part). A block's digest is the list of its
  slots' digests, so radix-node splits need no rehashing.
* verify-at-hit: ``UnifiedRadixCache.match_prefix`` is wrapped. For a device
  hit it re-reads the matched slots in every layer, recomputes their digests
  and compares. On any mismatch the match is turned into a full miss, so the
  request recomputes its whole prompt (and its new child KV is built from clean
  values).
* The corrupted node itself stays in the tree (SGLang keeps the old slots when
  the recomputed prefix is inserted again), so every later hit on it detects
  and recomputes again. That is the baseline's cost, not a repair.

Digest: sum over (layer, K/V, head, element) of raw16(x) * w with fixed random
odd 64-bit weights, wrapping mod 2^64. Any single flipped bit changes it (w is
odd); this stands in for SHA-256, which has no GPU kernel here, and detects at
least as well for the faults studied.

Enable with ``RADIXMEND_HASH=1`` (loaded by ``radixmend/sglang_site``).
"""

from __future__ import annotations

import os

import torch

_state = {"weights": None, "pools": [], "stats": {"hits_checked": 0, "tokens_checked": 0,
                                                    "mismatches": 0, "tokens_recomputed": 0}}


def _weights(n_layers: int, heads: int, dim: int, device) -> torch.Tensor:
    """[layers, 2, heads*dim] fixed random odd int64 weights."""
    w = _state["weights"]
    if w is None or w.shape != (n_layers, 2, heads * dim) or w.device != torch.device(device):
        g = torch.Generator().manual_seed(0x5EED)
        w = torch.randint(-(2 ** 62), 2 ** 62, (n_layers, 2, heads * dim), generator=g, dtype=torch.int64)
        w = (w | 1).to(device)
        _state["weights"] = w
    return w


def _raw(rows: torch.Tensor) -> torch.Tensor:
    """[n, H, D] 16-bit rows -> [n, H*D] int64 holding the raw bits (0..65535)."""
    return rows.reshape(rows.shape[0], -1).view(torch.int16).to(torch.int64) & 0xFFFF


def layer_part(k_rows: torch.Tensor, v_rows: torch.Tensor, w_layer: torch.Tensor) -> torch.Tensor:
    """Digest contribution of one layer for n slots: [n] int64 (wrapping)."""
    return (_raw(k_rows) * w_layer[0]).sum(-1) + (_raw(v_rows) * w_layer[1]).sum(-1)


def attach_pool(pool) -> None:
    n, h, d = pool.k_buffer[0].shape
    dev = pool.k_buffer[0].device
    _weights(len(pool.k_buffer), h, d, dev)
    pool.rm_digest = torch.zeros(n, dtype=torch.int64, device=dev)
    _state["pools"].append(pool)


def store(pool, lid: int, loc: torch.Tensor) -> None:
    """Fold layer ``lid``'s freshly written rows at ``loc`` into the slot digests."""
    if loc.numel() == 0:
        return
    loc = loc.to(torch.int64)
    part = layer_part(pool.k_buffer[lid][loc], pool.v_buffer[lid][loc], _state["weights"][lid])
    if lid == 0:
        pool.rm_digest[loc] = part
    else:
        pool.rm_digest.index_add_(0, loc, part)


def digest_now(pool, slots: torch.Tensor) -> torch.Tensor:
    """Recompute the digests of ``slots`` from the current KV contents."""
    slots = slots.to(torch.int64)
    acc = torch.zeros(slots.numel(), dtype=torch.int64, device=slots.device)
    w = _state["weights"]
    for lid in range(len(pool.k_buffer)):
        acc += layer_part(pool.k_buffer[lid][slots], pool.v_buffer[lid][slots], w[lid])
    return acc


def verify(pool, slots: torch.Tensor) -> int:
    """Number of slots whose stored digest no longer matches (one host sync)."""
    if slots.numel() == 0:
        return 0
    return int((digest_now(pool, slots) != pool.rm_digest[slots.to(torch.int64)]).sum().item())


def stats(reset: bool = False) -> dict:
    out = dict(_state["stats"])
    if reset:
        for k in _state["stats"]:
            _state["stats"][k] = 0
    return out


def install() -> None:
    if os.environ.get("RADIXMEND_HASH") != "1":
        return
    from sglang.srt.mem_cache import memory_pool as mp
    from sglang.srt.mem_cache import unified_radix_cache as urc
    from sglang.srt.mem_cache.base_prefix_cache import zero_match_result

    if getattr(mp.MHATokenToKVPool, "_rm_hash", False):
        return

    orig_init = mp.MHATokenToKVPool.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        attach_pool(self)

    orig_set = mp.MHATokenToKVPool.set_kv_buffer

    def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, *args, **kwargs):
        orig_set(self, layer, loc_info, cache_k, cache_v, *args, **kwargs)
        loc, _, _ = mp.unwrap_write_loc(loc_info)
        ov = kwargs.get("layer_id_override")
        store(self, (ov if ov is not None else layer.layer_id) - self.start_layer, loc)

    orig_match = urc.UnifiedRadixCache.match_prefix

    def match_prefix(self, params):
        res = orig_match(self, params)
        n = res.device_prefix_len
        if n <= 0 or not _state["pools"]:
            return res
        extra_key = getattr(params.key, "extra_key", None)
        slots = self.tree_core.collect_full_device_indices(res.last_device_node, self.root_node_handle(extra_key))
        slots = slots[:n]
        pool = _state["pools"][-1]
        st = _state["stats"]
        st["hits_checked"] += 1
        st["tokens_checked"] += n
        if verify(pool, slots):
            st["mismatches"] += 1
            st["tokens_recomputed"] += n
            return zero_match_result(self, res, extra_key=extra_key)
        return res

    mp.MHATokenToKVPool.__init__ = __init__
    mp.MHATokenToKVPool.set_kv_buffer = set_kv_buffer
    urc.UnifiedRadixCache.match_prefix = match_prefix
    mp.MHATokenToKVPool._rm_hash = True
