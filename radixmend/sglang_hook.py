"""In-process fault-injection hook for an SGLang v0.5.22 server.

The scheduler runs in a spawned subprocess, so the hook is installed by
``sitecustomize`` (see ``radixmend/sglang_site/sitecustomize.py``) when the
environment variable ``RADIXMEND_CMD_DIR`` is set. It wraps
``Scheduler.ingest_requests``, which both event loops call once per iteration,
and executes JSON command files dropped into that directory by a driver:

    <cmd_dir>/<id>.cmd.json   written by the driver (atomically, via rename)
    <cmd_dir>/<id>.done.json  written by the hook with the result

Commands (field ``op``):

* ``tree_info``  first-level nodes: token count and lock_ref.
* ``inject_kv``  flip bits in K/V of slots taken from the largest first-level
  node, positions ``[pos_lo, pos_hi)``. Fields: ``seed``, ``kinds``, ``bits``,
  ``layers`` (default all), ``flips_per_vector``, ``vectors``.
  ``dry_run: true`` samples and logs the same faults without flipping
  (golden runs go through the identical code path).
* ``revert``     undo every flip made since the last revert.
* ``inject_index`` flip one bit in an index tensor (RQ1-5). Field ``target``:
  ``node_value`` (int64 slot list of the largest first-level node, positions
  ``[pos_lo, pos_hi)``) and ``free_pages`` (one of the first ``head`` free
  slots) run at once; ``req_to_token`` (int32 row of the first running request,
  prefix positions) and ``seq_lens`` (running batch) wait until a running
  request has produced ``after_tokens`` output tokens, so the command stays
  pending until then. Fields: ``seed``, ``bits`` (candidate bit positions).
  The result classifies the new value: ``out_of_range``, ``other_node`` (a
  slot owned by another part of the tree), ``same_node`` or ``free``.
  Index flips are not reverted: ``flush_cache`` rebuilds all of these.
* ``cancel``     drop a pending command (field ``target_id``).
* ``snapshot_kv`` / ``compare_kv`` (RQ1-6): follow ``tokens`` down the tree,
  take the slots of positions ``[pos_lo, pos_hi)`` and save their K/V for all
  layers to ``<cmd_dir>/<name>.pt`` / compare the current K/V with that file
  (per-layer max abs difference and number of differing elements).

Only TP size 1 is supported. Commands run between batches on the scheduler
thread, followed by ``torch.cuda.synchronize()``.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from radixmend import fault_inject as fi

_state = {"records": [], "last_poll": 0.0}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _first_level(sched):
    root = sched.tree_cache.tree_core.root_node
    return fi.first_level_nodes(root)


def _kv_pool(sched):
    return sched.token_to_kv_pool_allocator.get_kvcache()


def _op_tree_info(sched, cmd):
    nodes = _first_level(sched)
    return {
        "first_level": [
            {"tokens": int(len(fi.node_value(n))), "lock_ref": int(fi.node_lock_ref(n))} for n in nodes
        ]
    }


def _op_inject_kv(sched, cmd):
    nodes = _first_level(sched)
    if not nodes:
        raise RuntimeError("radix tree is empty: send the warm-up request first")
    node = max(nodes, key=lambda n: len(fi.node_value(n)))
    slots_all = fi.node_slots(node)
    lo, hi = int(cmd.get("pos_lo", 0)), int(cmd.get("pos_hi", len(slots_all)))
    hi = min(hi, len(slots_all))
    if lo >= hi:
        raise RuntimeError(f"empty position range [{lo},{hi}) for node of {len(slots_all)} tokens")
    slots = slots_all[lo:hi]
    pool = _kv_pool(sched)
    k, v = pool.k_buffer, pool.v_buffer
    _, head_num, head_dim = v[0].shape
    layers = cmd.get("layers") or list(range(len(v)))
    gen = torch.Generator().manual_seed(int(cmd["seed"]))
    faults = fi.sample_kv_faults(
        gen, slots, layers, head_num, head_dim,
        kinds=cmd.get("kinds", ["v"]), bits=cmd.get("bits"),
        flips_per_vector=int(cmd.get("flips_per_vector", 1)), vectors=int(cmd.get("vectors", 1)),
    )
    pos_of = {s: lo + i for i, s in enumerate(slots)}
    if cmd.get("dry_run"):
        recs = [{"target": "kv", "fault": asdict(f), "dry_run": True} for f in faults]
    else:
        records = fi.inject_kv(k, v, faults)
        _sync()
        _state["records"] += records
        recs = [asdict(r) for r in records]
    for r in recs:
        r["position"] = pos_of.get(r["fault"]["slot"])
    return {"node_tokens": len(slots_all), "range": [lo, hi], "flips": recs}


def _op_revert(sched, cmd):
    pool = _kv_pool(sched)
    n = len(_state["records"])
    fi.revert(_state["records"], k_buffer=pool.k_buffer, v_buffer=pool.v_buffer)
    _sync()
    _state["records"] = []
    return {"reverted": n}


def _slot_owner(sched, slot: int, own_node) -> str:
    pool = _kv_pool(sched)
    size = pool.v_buffer[0].shape[0]
    if slot < 0 or slot >= size:
        return "out_of_range"
    for n, _ in fi.iter_nodes(sched.tree_cache.tree_core.root_node):
        if slot in set(fi.node_slots(n)):
            return "same_node" if n is own_node else "other_node"
    return "free"


def _flip_one(t: torch.Tensor, flat: int, bit: int):
    (old, new), = fi.flip_bits_(t, [flat], [bit])
    _sync()
    return old, new


def _op_inject_index(sched, cmd):
    gen = torch.Generator().manual_seed(int(cmd["seed"]))
    bits = cmd["bits"]
    bit = int(bits[int(torch.randint(len(bits), (1,), generator=gen))])
    target = cmd["target"]

    def pick(lo, hi):
        return lo + int(torch.randint(hi - lo, (1,), generator=gen))

    if target in ("node_value", "req_to_token"):
        nodes = _first_level(sched)
        node = max(nodes, key=lambda n: len(fi.node_value(n))) if nodes else None
    if target == "node_value":
        if node is None:
            raise RuntimeError("radix tree is empty")
        val = fi.node_value(node)
        pos = pick(int(cmd.get("pos_lo", 0)), min(int(cmd.get("pos_hi", len(val))), len(val)))
        old, new = _flip_one(val, pos, bit)
        return {"target": target, "position": pos, "bit": bit, "old": old, "new": new,
                "new_owner": _slot_owner(sched, new, node)}
    if target == "free_pages":
        fp = sched.token_to_kv_pool_allocator.free_pages
        idx = pick(0, min(int(cmd.get("head", 64)), len(fp)))
        old, new = _flip_one(fp, idx, bit)
        return {"target": target, "index": idx, "bit": bit, "old": old, "new": new,
                "new_owner": _slot_owner(sched, new, None)}
    # deferred targets: wait for a running request with enough output
    batch = sched.running_batch
    reqs = [r for r in (batch.reqs if batch is not None else []) if len(r.output_ids) >= int(cmd.get("after_tokens", 1))]
    if not reqs:
        return None
    req = reqs[0]
    if target == "req_to_token":
        row = req.kv.req_pool_idx
        r2t = sched.req_to_token_pool.req_to_token
        pos = pick(int(cmd.get("pos_lo", 0)), int(cmd.get("pos_hi", 1)))
        old, new = _flip_one(r2t, row * r2t.shape[1] + pos, bit)
        return {"target": target, "row": int(row), "position": pos, "bit": bit, "old": old, "new": new,
                "new_owner": _slot_owner(sched, new, node), "at_output_tokens": len(req.output_ids)}
    if target == "seq_lens":
        i = batch.reqs.index(req)
        old, new = _flip_one(batch.seq_lens, i, bit)
        return {"target": target, "batch_index": i, "bit": bit, "old": old, "new": new,
                "at_output_tokens": len(req.output_ids)}
    raise ValueError(f"unknown index target {target}")


def _path_slots(sched, tokens: list[int]) -> list[int]:
    """KV slots of the longest cached prefix of ``tokens`` (walks the tree by key)."""
    node = sched.tree_cache.tree_core.root_node
    pos, slots = 0, []
    while pos < len(tokens):
        nxt = None
        for c in node.children.values():
            key = list(c.key.token_ids) if c.key is not None else []
            if key and key[0] == tokens[pos]:
                nxt, ckey = c, key
                break
        if nxt is None:
            break
        m = 0
        while m < len(ckey) and pos + m < len(tokens) and ckey[m] == tokens[pos + m]:
            m += 1
        slots += fi.node_slots(nxt)[:m]
        pos += m
        if m < len(ckey):
            break
        node = nxt
    return slots


def _gather_kv(sched, cmd):
    slots = _path_slots(sched, [int(t) for t in cmd["tokens"]])
    lo, hi = int(cmd["pos_lo"]), int(cmd["pos_hi"])
    if hi > len(slots):
        raise RuntimeError(f"only {len(slots)} of the tokens are cached, need up to {hi}")
    idx = torch.tensor(slots[lo:hi], dtype=torch.long, device=_kv_pool(sched).v_buffer[0].device)
    pool = _kv_pool(sched)
    k = torch.stack([b.index_select(0, idx) for b in pool.k_buffer])
    v = torch.stack([b.index_select(0, idx) for b in pool.v_buffer])
    return torch.stack([k, v], dim=1).cpu()  # [layers, 2, n, heads, dim]


def _op_snapshot_kv(sched, cmd):
    kv = _gather_kv(sched, cmd)
    torch.save(kv, Path(os.environ["RADIXMEND_CMD_DIR"]) / f"{cmd['name']}.pt")
    return {"tokens": int(kv.shape[2])}


def _op_compare_kv(sched, cmd):
    cur = _gather_kv(sched, cmd).float()
    ref = torch.load(Path(os.environ["RADIXMEND_CMD_DIR"]) / f"{cmd['name']}.pt").float()
    if ref.shape != cur.shape:
        raise RuntimeError(f"shape mismatch {tuple(ref.shape)} vs {tuple(cur.shape)}")
    d = (cur - ref).abs()
    d = torch.nan_to_num(d, nan=float("inf"))
    per_layer = d.amax(dim=(1, 2, 3, 4)).tolist()
    per_token = (d.amax(dim=(0, 1, 3, 4)) > 0).tolist()
    return {"max_abs_per_layer": per_layer, "max_abs": max(per_layer), "n_diff": int((d > 0).sum()),
            "n_elems": int(d.numel()), "tokens_differing": int(sum(per_token)), "n_tokens": len(per_token)}


def _op_cancel(sched, cmd):
    p = Path(os.environ["RADIXMEND_CMD_DIR"]) / f"{cmd['target_id']}.cmd.json"
    existed = p.exists()
    p.unlink(missing_ok=True)
    return {"cancelled": existed}


def _op_secded_stats(sched, cmd):
    from radixmend import sglang_secded as ss

    errs = ss.errors()
    if cmd.get("reset") and ss._state["err"] is not None:
        ss._state["err"].zero_()
    errs = errs or [0] * 8
    names = ["k_corrected", "k_uncorrectable", "v_corrected", "v_uncorrectable"]
    out = {"enabled": ss._state["err"] is not None}
    out.update({f"decode_{n}": errs[i] for i, n in enumerate(names)})
    out.update({f"extend_{n}": errs[4 + i] for i, n in enumerate(names)})
    return out


_OPS = {"tree_info": _op_tree_info, "inject_kv": _op_inject_kv, "revert": _op_revert,
        "inject_index": _op_inject_index, "cancel": _op_cancel,
        "snapshot_kv": _op_snapshot_kv, "compare_kv": _op_compare_kv,
        "secded_stats": _op_secded_stats}


def poll(sched, cmd_dir: Path) -> None:
    now = time.monotonic()
    if now - _state["last_poll"] < 0.005:
        return
    _state["last_poll"] = now
    for path in sorted(cmd_dir.glob("*.cmd.json")):
        cid = path.name[: -len(".cmd.json")]
        try:
            cmd = json.loads(path.read_text())
            res = _OPS[cmd["op"]](sched, cmd)
            if res is None:  # deferred: keep the command for a later poll
                continue
            out = {"ok": True, "id": cid, **res}
        except FileNotFoundError:
            continue  # cancelled between glob and read
        except Exception as e:  # report to the driver instead of killing the server
            out = {"ok": False, "id": cid, "error": repr(e), "trace": traceback.format_exc()}
        _write_json(cmd_dir / f"{cid}.done.json", out)
        path.unlink(missing_ok=True)


def install() -> None:
    cmd_dir = os.environ.get("RADIXMEND_CMD_DIR")
    if not cmd_dir:
        return
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler, "_radixmend_hooked", False):
        return
    cmd_path = Path(cmd_dir)
    cmd_path.mkdir(parents=True, exist_ok=True)
    orig = Scheduler.ingest_requests

    def ingest_requests(self, *args, **kwargs):
        poll(self, cmd_path)
        return orig(self, *args, **kwargs)

    Scheduler.ingest_requests = ingest_requests
    Scheduler._radixmend_hooked = True
