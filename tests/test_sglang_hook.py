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


def test_index_node_value_and_free_pages(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIXMEND_CMD_DIR", str(tmp_path))
    sched, _ = fake_sched()
    sched.token_to_kv_pool_allocator.free_pages = torch.arange(50, 64)
    res = send(tmp_path, sched, {"op": "inject_index", "target": "node_value", "seed": 3, "bits": [0], "pos_lo": 0, "pos_hi": 20})
    assert res["ok"] and res["new"] == res["old"] ^ 1
    assert res["new_owner"] in ("same_node", "other_node", "free")
    res = send(tmp_path, sched, {"op": "inject_index", "target": "free_pages", "seed": 3, "bits": [40]})
    assert res["new_owner"] == "out_of_range"


def test_index_deferred_waits_for_running_request(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIXMEND_CMD_DIR", str(tmp_path))
    sched, _ = fake_sched()
    sched.running_batch = SimpleNamespace(reqs=[], seq_lens=torch.tensor([30]))
    sched.req_to_token_pool = SimpleNamespace(req_to_token=torch.zeros(4, 40, dtype=torch.int32))
    (tmp_path / "d.cmd.json").write_text(json.dumps({"op": "inject_index", "target": "seq_lens", "seed": 1, "bits": [2], "after_tokens": 3}))
    hk._state["last_poll"] = 0.0
    hk.poll(sched, tmp_path)
    assert (tmp_path / "d.cmd.json").exists() and not (tmp_path / "d.done.json").exists()
    req = SimpleNamespace(output_ids=[1, 2, 3], kv=SimpleNamespace(req_pool_idx=2))
    sched.running_batch.reqs = [req]
    hk._state["last_poll"] = 0.0
    hk.poll(sched, tmp_path)
    res = json.loads((tmp_path / "d.done.json").read_text())
    assert res["ok"] and res["old"] == 30 and res["new"] == 26
    res = send(tmp_path, sched, {"op": "inject_index", "target": "req_to_token", "seed": 1, "bits": [5], "pos_lo": 0, "pos_hi": 10, "after_tokens": 1})
    assert res["row"] == 2 and res["new"] == 32


def test_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIXMEND_CMD_DIR", str(tmp_path))
    sched, _ = fake_sched()
    sched.running_batch = None
    (tmp_path / "p.cmd.json").write_text(json.dumps({"op": "inject_index", "target": "seq_lens", "seed": 1, "bits": [2]}))
    res = send(tmp_path, sched, {"op": "cancel", "target_id": "p"})
    assert res["cancelled"] and not (tmp_path / "p.cmd.json").exists()


def test_snapshot_and_compare_follow_tree_path(tmp_path, monkeypatch):
    monkeypatch.setenv("RADIXMEND_CMD_DIR", str(tmp_path))
    sched, pool = fake_sched()
    root = sched.tree_cache.tree_core.root_node
    parent = Node(torch.arange(10, 14))
    parent.key = SimpleNamespace(token_ids=[1, 2, 3, 4])
    child = Node(torch.arange(20, 23))
    child.key = SimpleNamespace(token_ids=[5, 6, 7])
    other = Node(torch.arange(30, 32))
    other.key = SimpleNamespace(token_ids=[9, 9])
    parent.children = {5: child, 9: other}
    root.children = {1: parent}
    assert hk._path_slots(sched, [1, 2, 3, 4, 5, 6, 8]) == [10, 11, 12, 13, 20, 21]
    cmd = {"tokens": [1, 2, 3, 4, 5, 6, 7], "pos_lo": 4, "pos_hi": 7}
    assert send(tmp_path, sched, {"op": "snapshot_kv", **cmd, "name": "g"})["tokens"] == 3
    res = send(tmp_path, sched, {"op": "compare_kv", **cmd, "name": "g"})
    assert res["n_diff"] == 0 and res["max_abs"] == 0
    pool.v_buffer[2][21, 1, 3] += 1
    res = send(tmp_path, sched, {"op": "compare_kv", **cmd, "name": "g"})
    assert res["n_diff"] == 1 and res["tokens_differing"] == 1 and res["max_abs_per_layer"][2] > 0
    res = send(tmp_path, sched, {"op": "compare_kv", **{**cmd, "pos_hi": 9}, "name": "g"})
    assert not res["ok"] and "cached" in res["error"]


def test_inject_kv_deferred_until_decoding(tmp_path):
    sched, pool = fake_sched()
    sched.running_batch = SimpleNamespace(reqs=[SimpleNamespace(output_ids=[1])])
    before = [t.clone() for t in pool.v_buffer]
    op = {"op": "inject_kv", "seed": 1, "kinds": ["v"], "bits": [14], "after_tokens": 3}
    (tmp_path / "d.cmd.json").write_text(json.dumps(op))
    hk._state["last_poll"] = 0.0
    hk.poll(sched, tmp_path)
    assert (tmp_path / "d.cmd.json").exists() and all(torch.equal(a, b) for a, b in zip(before, pool.v_buffer))
    sched.running_batch.reqs[0].output_ids = [1, 2, 3, 4]
    hk._state["last_poll"] = 0.0
    hk.poll(sched, tmp_path)
    res = json.loads((tmp_path / "d.done.json").read_text())
    assert res["ok"] and res["at_output_tokens"] == 4 and len(res["flips"]) == 1
    send(tmp_path, sched, {"op": "revert"})
    assert all(torch.equal(a, b) for a, b in zip(before, pool.v_buffer))


def test_stats_ops_without_modules(tmp_path):
    sched, _ = fake_sched()
    assert send(tmp_path, sched, {"op": "secded_stats"})["ok"]
    assert send(tmp_path, sched, {"op": "hash_stats", "reset": True})["ok"]


def test_bits_each_flips_distinct_elements_at_given_bits(tmp_path):
    sched, pool = fake_sched()
    res = send(tmp_path, sched, {"op": "inject_kv", "seed": 3, "kinds": ["k"], "bits_each": [14, 7]})
    fs = [r["fault"] for r in res["flips"]]
    assert sorted(f["bit"] for f in fs) == [7, 14]
    assert len({f["elem"] for f in fs}) == 2
    assert len({(f["layer"], f["slot"], f["head"]) for f in fs}) == 1
    send(tmp_path, sched, {"op": "revert"})


def test_deepest_node_and_checkword_target(tmp_path):
    sched, pool = fake_sched()
    a = sched.tree_cache.tree_core.root_node.children["a"]
    a.children = {"c": Node(torch.arange(50, 53))}
    pool.rm_k_cw = [torch.zeros(64, 2, dtype=torch.int16) for _ in range(3)]
    pool.rm_v_cw = [torch.zeros(64, 2, dtype=torch.int16) for _ in range(3)]
    data_before = [t.clone() for t in pool.v_buffer]
    res = send(tmp_path, sched, {"op": "inject_kv", "seed": 1, "kinds": ["v"], "bits": [15],
                                 "node": "deepest", "target": "cw"})
    assert res["node_depth"] == 2 and res["node_tokens"] == 3
    f = res["flips"][0]
    assert 50 <= f["fault"]["slot"] < 53 and f["new"] == 0x8000
    assert int(pool.rm_v_cw[f["fault"]["layer"]][f["fault"]["slot"], f["fault"]["head"]]) == -0x8000
    assert all(torch.equal(x, y) for x, y in zip(data_before, pool.v_buffer))
    assert send(tmp_path, sched, {"op": "revert"})["reverted"] == 1
    assert all(int(t.abs().sum()) == 0 for t in pool.rm_v_cw)


def test_deepest_on_token_path_skips_unread_leaf(tmp_path):
    sched, pool = fake_sched()
    root = sched.tree_cache.tree_core.root_node
    parent = Node(torch.arange(10, 14))
    parent.key = SimpleNamespace(token_ids=[1, 2, 3, 4])
    child = Node(torch.arange(20, 25))
    child.key = SimpleNamespace(token_ids=[5, 6, 7, 8, 9])
    answer = Node(torch.arange(30, 45))  # warm-up's generated answer, deeper but not on the path
    answer.key = SimpleNamespace(token_ids=[9] * 15)
    other = Node(torch.arange(50, 52))
    other.key = SimpleNamespace(token_ids=[7, 7])
    parent.children = {5: child, 7: other}
    other.children = {9: answer}
    root.children = {1: parent}
    res = send(tmp_path, sched, {"op": "inject_kv", "seed": 2, "kinds": ["v"], "bits": [14],
                                 "node": "deepest", "tokens": [1, 2, 3, 4, 5, 6, 7, 0]})
    assert res["node_depth"] == 2 and res["node_tokens"] == 3  # only the reused part of the child
    assert 20 <= res["flips"][0]["fault"]["slot"] < 23
    send(tmp_path, sched, {"op": "revert"})
