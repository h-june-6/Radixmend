import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from radixmend import sglang_hook as hk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rq1_kv_sweep as sweep  # noqa: E402


class Node:
    def __init__(self, value, lock_ref=0):
        self.value, self.lock_ref, self.children = value, lock_ref, {}


def fake_sched(layers=3, slots=64, heads=2, dim=8):
    root = Node(None)
    root.children = {"a": Node(torch.arange(10, 30)), "b": Node(torch.arange(40, 45))}
    k = [torch.randn(slots, heads, dim, dtype=torch.bfloat16) for _ in range(layers)]
    v = [torch.randn(slots, heads, dim, dtype=torch.bfloat16) for _ in range(layers)]
    pool = SimpleNamespace(k_buffer=k, v_buffer=v)
    return SimpleNamespace(
        tree_cache=SimpleNamespace(tree_core=SimpleNamespace(root_node=root)),
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
    ), pool


def send(tmp, sched, op):
    (tmp / "x.cmd.json").write_text(json.dumps(op))
    hk._state["last_poll"] = 0.0
    hk.poll(sched, tmp)
    return json.loads((tmp / "x.done.json").read_text())


def test_inject_revert_roundtrip(tmp_path):
    sched, pool = fake_sched()
    before = [t.clone() for t in pool.v_buffer]
    info = send(tmp_path, sched, {"op": "tree_info"})
    assert sorted(n["tokens"] for n in info["first_level"]) == [5, 20]
    res = send(tmp_path, sched, {"op": "inject_kv", "seed": 1, "kinds": ["v"], "bits": [14], "pos_lo": 2, "pos_hi": 6})
    assert res["ok"] and len(res["flips"]) == 1
    f = res["flips"][0]
    assert 12 <= f["fault"]["slot"] < 16 and 2 <= f["position"] < 6  # largest node, positions 2..5
    assert any(not torch.equal(a, b) for a, b in zip(before, pool.v_buffer))
    assert send(tmp_path, sched, {"op": "revert"})["reverted"] == 1
    assert all(torch.equal(a, b) for a, b in zip(before, pool.v_buffer))


def test_dry_run_changes_nothing_and_same_fault(tmp_path):
    sched, pool = fake_sched()
    before = [t.clone() for t in pool.v_buffer]
    op = {"op": "inject_kv", "seed": 7, "kinds": ["v"], "bits": [3]}
    dry = send(tmp_path, sched, {**op, "dry_run": True})
    assert all(torch.equal(a, b) for a, b in zip(before, pool.v_buffer))
    real = send(tmp_path, sched, op)
    assert dry["flips"][0]["fault"] == real["flips"][0]["fault"]
    send(tmp_path, sched, {"op": "revert"})


def test_errors_are_reported(tmp_path):
    sched, _ = fake_sched()
    res = send(tmp_path, sched, {"op": "inject_kv", "seed": 1, "pos_lo": 50, "pos_hi": 60})
    assert not res["ok"] and "empty position range" in res["error"]


def test_sweep_helpers():
    assert sweep.parse_bits("0-3,7,14-15") == [0, 1, 2, 3, 7, 14, 15]
    assert sweep.first_diff([1, 2, 3], [1, 2, 3]) is None
    assert sweep.first_diff([1, 2, 3], [1, 9, 3]) == 1
    assert sweep.first_diff([1, 2], [1, 2, 3]) == 2
    assert sweep.positions("sink", 900) == (0, 4) and sweep.positions("body", 900) == (4, 900)
    lo, hi = sweep.wilson(25, 50)
    assert 0.36 < lo < 0.37 and 0.63 < hi < 0.64
