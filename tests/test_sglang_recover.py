from dataclasses import dataclass
from types import SimpleNamespace

import torch

from radixmend import sglang_recover as sr
from radixmend import sglang_secded as ss


def make_pool(layers=2, slots=32, heads=2, dim=4):
    k = [torch.randn(slots, heads, dim) for _ in range(layers)]
    v = [torch.randn(slots, heads, dim) for _ in range(layers)]
    kc = [torch.randint(0, 1 << 15, (slots, heads), dtype=torch.int16) for _ in range(layers)]
    vc = [torch.randint(0, 1 << 15, (slots, heads), dtype=torch.int16) for _ in range(layers)]
    pool = SimpleNamespace(k_buffer=k, v_buffer=v, rm_k_cw=kc, rm_v_cw=vc)
    sr.attach_pool(pool)
    sr._state["pending"].clear()
    sr.stats(reset=True)
    return pool


def set_log(slots):
    log = torch.zeros(ss.LOG_CAP * 3, dtype=torch.int32)
    for i, s in enumerate(slots):
        log[3 * i] = s
    ss._state["log"] = log
    ss._state["logn"] = torch.tensor([len(slots)], dtype=torch.int32)


def test_new_bad_slots_reads_and_resets():
    make_pool()
    set_log([5, 9, 5])
    assert sr.new_bad_slots().tolist() == [5, 9]
    assert sr.new_bad_slots() is None
    assert sr.stats()["detections"] == 3


def test_affected_rows_respects_seq_len():
    r2t = torch.zeros(4, 8, dtype=torch.int32)
    r2t[0, :3] = torch.tensor([1, 2, 3])
    r2t[1, :5] = torch.tensor([1, 2, 3, 4, 7])
    r2t[2, :6] = torch.tensor([10, 11, 12, 13, 14, 7])
    idx = sr.affected_rows(r2t, torch.tensor([0, 1, 2]), torch.tensor([3, 5, 5]), torch.tensor([7]))
    assert idx == [1]  # row 2 holds slot 7 only beyond its seq_len


def test_poison_and_first_poisoned():
    make_pool()
    assert sr.first_poisoned(torch.tensor([1, 2, 3])) == -1
    sr.poison(torch.tensor([2, 6]))
    assert sr.first_poisoned(torch.tensor([1, 2, 3])) == 1
    sr.unpoison_written(torch.tensor([2]))
    assert sr.first_poisoned(torch.tensor([1, 2, 3])) == -1
    assert sr.first_poisoned(torch.tensor([6])) == 0


def test_repair_on_insert_copies_into_old_slots():
    pool = make_pool()
    old = torch.tensor([3, 4, 5])
    new = torch.tensor([20, 21, 22])
    sr.poison(old)
    sr._state["pending"].append(((1, 2, 3), old.clone()))
    # a different key does not trigger the repair
    assert sr.repair_on_insert(pool, [9, 9, 9, 9], torch.tensor([20, 21, 22, 23])) == 0
    n = sr.repair_on_insert(pool, [1, 2, 3, 8], torch.tensor([20, 21, 22, 23]))
    assert n == 3 and not sr._state["pending"]
    for lid in range(2):
        assert torch.equal(pool.k_buffer[lid][old], pool.k_buffer[lid][new])
        assert torch.equal(pool.v_buffer[lid][old], pool.v_buffer[lid][new])
        assert torch.equal(pool.rm_k_cw[lid][old], pool.rm_k_cw[lid][new])
        assert torch.equal(pool.rm_v_cw[lid][old], pool.rm_v_cw[lid][new])
    assert sr.stats()["poisoned_now"] == 0 and sr.stats()["tokens_repaired"] == 3


def test_write_back_skips_identical_slots():
    pool = make_pool()
    sr.poison(torch.tensor([3]))
    assert sr.write_back(pool, torch.tensor([3, 4]), torch.tensor([3, 9])) == 1
    assert sr.first_poisoned(torch.tensor([3])) == 0  # same slot: nothing recomputed


@dataclass(kw_only=True, slots=True, frozen=True)
class Proc:  # like SGLang's SchedulerBatchResultProcessor
    enable_overlap: bool = False


class Req:
    def __init__(self, name, done=False, pool_idx=None, n_tokens=0):
        self.name, self.done, self.is_retracted = name, done, False
        self.kv = SimpleNamespace(req_pool_idx=pool_idx)
        self.origin_input_ids, self.output_ids = [0] * n_tokens, []

    def finished(self):
        return self.done


def fake_batch(reqs, rows, decode=True):
    r2t = torch.zeros(len(reqs), 8, dtype=torch.int32)
    for i, row in enumerate(rows):
        r2t[i, : len(row)] = torch.tensor(row)
    released, kept = [], []
    batch = SimpleNamespace(
        reqs=list(reqs),
        req_pool_indices=torch.arange(len(reqs)),
        seq_lens=torch.tensor([len(r) for r in rows]),
        forward_mode=SimpleNamespace(is_decode=lambda: decode),
        release_req=lambda i, n, offload_kv: released.append(batch.reqs[i].name),
        filter_batch=lambda keep_indices: kept.extend(keep_indices),
    )
    return batch, r2t, released, kept


def test_handle_step_retracts_only_affected_requests():
    make_pool()
    a, b, c = Req("a"), Req("b"), Req("c", done=True)
    batch, r2t, released, kept = fake_batch([a, b, c], [[1, 2, 3], [4, 5], [5, 6]])
    queued, seen = [], {}
    proc = Proc()

    def run_orig():  # SGLang's commit: must see the victims as retracted (overlap path for decode)
        seen.update(a=a.is_retracted, b=b.is_retracted, overlap=proc.enable_overlap)

    sched = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=r2t), batch_result_processor=proc,
                            _add_request_to_queue=lambda r, is_retracted: queued.append((r.name, is_retracted)))
    set_log([5])
    sr.handle_step(sched, batch, run_orig)
    assert seen == {"a": False, "b": True, "overlap": True}
    assert proc.enable_overlap is False
    assert released == ["b"] and kept == [0, 2] and queued == [("b", True)]
    assert sr.first_poisoned(torch.tensor([5])) == 0
    st = sr.stats()
    assert st["reqs_retracted"] == 1 and st["steps_with_errors"] == 1


def test_handle_step_without_errors_just_runs():
    make_pool()
    set_log([])
    calls = []
    batch, r2t, released, _ = fake_batch([Req("a")], [[1]])
    sched = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=r2t),
                            batch_result_processor=SimpleNamespace(enable_overlap=False))
    sr.handle_step(sched, batch, lambda: calls.append(1))
    assert calls == [1] and not released


def test_log_fake_feeds_new_bad_slots():
    make_pool()
    ss._state["log"] = torch.zeros(ss.LOG_CAP * 3, dtype=torch.int32)
    ss._state["logn"] = torch.zeros(1, dtype=torch.int32)
    assert ss.log_fake([(7, 0, 2), (9, 1, 6)]) == 2
    assert ss.error_log()["total"] == 2
    assert sr.new_bad_slots().tolist() == [7, 9]


def test_overlap_marks_victims_then_retracts_at_update_running_batch():
    make_pool()
    sr._state["victims"].clear()
    # requests carry their own pool row and length; the step-N batch is a trimmed copy (no seq_lens)
    a, b, c = Req("a", pool_idx=0, n_tokens=3), Req("b", pool_idx=1, n_tokens=3), Req("c", pool_idx=2, n_tokens=2)
    batch = SimpleNamespace(reqs=[a, b])
    running, r2t, released, kept = fake_batch([a, b, c], [[1, 2, 3], [4, 5, 6], [7, 9]])
    queued, seen = [], {}

    def run_orig():
        seen.update(a=a.is_retracted, b=b.is_retracted, c=c.is_retracted)

    sched = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=r2t), running_batch=running,
                            _add_request_to_queue=lambda r, is_retracted: queued.append((r.name, is_retracted)))
    set_log([5, 9])  # 5: read by b; 9: read by c only in the in-flight step
    sr.handle_step_overlap(sched, batch, run_orig)
    assert seen == {"a": False, "b": True, "c": True}  # commit of step N already skips b (and c)
    assert not released and not queued                # nothing released while N+1 is in flight
    sr.retract_pending(sched, running)                 # next update_running_batch
    assert released == ["b", "c"] and kept == [0]
    assert queued == [("b", True), ("c", True)] and not sr._state["victims"]
    assert sr.stats()["reqs_retracted"] == 2


def test_overlap_without_errors_just_runs():
    make_pool()
    set_log([])
    calls = []
    sched = SimpleNamespace(running_batch=None)
    sr.handle_step_overlap(sched, SimpleNamespace(reqs=[]), lambda: calls.append(1))
    assert calls == [1]


def test_overlap_uses_host_snapshot_not_device_count():
    make_pool()
    sr._state["victims"].clear()
    set_log([5])  # device log has an entry, but the snapshot of step N says 0: it belongs to step N+1
    before = sr.stats()
    calls = []
    sched = SimpleNamespace(running_batch=None)
    sr.handle_step_overlap(sched, SimpleNamespace(reqs=[]), lambda: calls.append(1),
                           torch.zeros(1, dtype=torch.int32))
    st = sr.stats()
    assert calls == [1] and st["overlap_snapshot_reads"] == before["overlap_snapshot_reads"] + 1
    assert st["overlap_item_reads"] == before["overlap_item_reads"]
    assert st["detections"] == before["detections"]   # left for the next step's snapshot
    assert int(ss._state["logn"]) == 1


def test_overlap_snapshot_nonzero_reads_log():
    make_pool()
    sr._state["victims"].clear()
    a = Req("a", pool_idx=0, n_tokens=2)
    running, r2t, _, _ = fake_batch([a], [[5, 6]])
    set_log([5])
    sched = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=r2t), running_batch=running)
    sr.handle_step_overlap(sched, SimpleNamespace(reqs=[a]), lambda: None, torch.ones(1, dtype=torch.int32))
    assert a.is_retracted and int(ss._state["logn"]) == 0
    sr._state["victims"].clear()


def test_snapshot_logn_skips_cpu_log():
    set_log([])
    r = SimpleNamespace()
    sr.snapshot_logn(r)  # CPU log (tests): nothing to snapshot, falls back to logn.item()
    assert not hasattr(r, "_rm_logn")


def test_install_targets_exist_in_sglang():
    import pytest

    pytest.importorskip("sglang")
    from sglang.srt.managers import scheduler as sm
    from sglang.srt.managers import utils as mu
    from sglang.srt.mem_cache import base_prefix_cache as bpc
    from sglang.srt.mem_cache import memory_pool as mp
    from sglang.srt.mem_cache import unified_radix_cache as urc

    for obj, attr in [(sm.Scheduler, "process_batch_result"), (sm.Scheduler, "update_running_batch"),
                      (sm.Scheduler, "_add_request_to_queue"), (mp.MHATokenToKVPool, "set_kv_buffer"),
                      (mp, "unwrap_write_loc"), (urc.UnifiedRadixCache, "match_prefix"),
                      (urc.UnifiedRadixCache, "insert"), (bpc, "zero_match_result"),
                      (mu.GenerationBatchResult, "copy_to_cpu")]:
        assert hasattr(obj, attr), attr
