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


_OPS = {"tree_info": _op_tree_info, "inject_kv": _op_inject_kv, "revert": _op_revert}


def poll(sched, cmd_dir: Path) -> None:
    now = time.monotonic()
    if now - _state["last_poll"] < 0.005:
        return
    _state["last_poll"] = now
    for path in sorted(cmd_dir.glob("*.cmd.json")):
        cid = path.name[: -len(".cmd.json")]
        try:
            cmd = json.loads(path.read_text())
            out = {"ok": True, "id": cid, **_OPS[cmd["op"]](sched, cmd)}
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
