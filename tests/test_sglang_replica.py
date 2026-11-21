"""Module D replicas (radixmend.sglang_replica + the REP paths of module A's kernels).

On a machine without a GPU run with TRITON_INTERPRET=1.
"""
import os
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)

from radixmend import sglang_replica as sd
from radixmend import sglang_secded as ss

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H_KV, GROUP, D, POOL, LAYERS = 2, 4, 128, 256, 2
DT = torch.bfloat16 if torch.cuda.is_available() else torch.float16


def make_pool(seed=0, capacity=64):
    g = torch.Generator().manual_seed(seed)
    k = [torch.randn(POOL, H_KV, D, generator=g).to(DT).to(DEV) for _ in range(LAYERS)]
    v = [torch.randn(POOL, H_KV, D, generator=g).to(DT).to(DEV) for _ in range(LAYERS)]
    ss._tables(k[0].device)
    ss._state["err"].zero_()
    ss._state["logn"].zero_()
    pool = SimpleNamespace(k_buffer=k, v_buffer=v,
                           rm_k_cw=[torch.zeros(POOL, H_KV, dtype=torch.int16, device=DEV) for _ in k],
                           rm_v_cw=[torch.zeros(POOL, H_KV, dtype=torch.int16, device=DEV) for _ in v])
    allloc = torch.arange(POOL, device=DEV)
    for lid in range(LAYERS):
        ss.encode_rows(k[lid], pool.rm_k_cw[lid], allloc)
        ss.encode_rows(v[lid], pool.rm_v_cw[lid], allloc)
    sd.attach_pool(pool, capacity)
    sd.stats(reset=True)
    return pool


def flip2(buf, slot, head, e1=3, e2=90, b1=14, b2=7):
    buf.view(torch.int16)[slot, head, e1] ^= 1 << b1
    buf.view(torch.int16)[slot, head, e2] ^= 1 << b2


def decode(pool, lid, q, indptr, idx, splits=2, **kw):
    b, hq = q.shape[:2]
    out = torch.zeros(b, hq, splits, D, dtype=torch.float32, device=DEV)
    lse = torch.zeros(b, hq, splits, dtype=torch.float32, device=DEV)
    nsp = torch.full((b,), splits, dtype=torch.int32, device=DEV)
    ss.launch_stage1(q, pool.k_buffer[lid], pool.v_buffer[lid], pool.rm_k_cw[lid], pool.rm_v_cw[lid], out, lse,
                     indptr, idx, nsp, splits, D ** -0.5, 0.0, True, layer=lid, **kw)
    return out, lse


def batch(seed=1, seqs=(40, 70)):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(len(seqs), H_KV * GROUP, D, generator=g).to(DT).to(DEV)
    idx = torch.randperm(POOL, generator=g)[: sum(seqs)].to(torch.int64).to(DEV)
    indptr = torch.tensor([0] + list(torch.tensor(seqs).cumsum(0)), dtype=torch.int32, device=DEV)
    return q, idx, indptr


def test_replicate_copies_and_maps():
    pool = make_pool()
    slots = torch.tensor([5, 9, 9, 17], device=DEV)
    assert sd.replicate(pool, slots) == 3
    r = sd.rep()
    rs = r["map"][torch.tensor([5, 9, 17], device=DEV)].to(torch.int64)
    assert (rs >= 0).all() and len(set(rs.tolist())) == 3
    for lid, (rk, rv, rkc, rvc) in r["layers"].items():
        assert torch.equal(rk[rs], pool.k_buffer[lid][[5, 9, 17]])
        assert torch.equal(rvc[rs], pool.rm_v_cw[lid][[5, 9, 17]])
    assert sd.replicate(pool, slots) == 0  # already there
    assert sd.stats()["replicas_now"] == 3


def test_replicate_skips_uncorrectable_and_fixes_single_bits():
    pool = make_pool()
    clean = pool.k_buffer[1][7].clone()
    pool.k_buffer[1].view(torch.int16)[7, 0, 11] ^= 1 << 9  # 1-bit: fixed before copying
    flip2(pool.v_buffer[0], 8, 1)  # 2-bit: not copied, left to module C
    assert sd.replicate(pool, torch.tensor([7, 8], device=DEV)) == 1
    assert torch.equal(pool.k_buffer[1][7], clean)
    r = sd.rep()
    assert int(r["map"][8]) == -1 and int(r["map"][7]) >= 0
    assert sd.stats()["skipped_bad"] == 1
    assert int(ss._state["logn"].item()) >= 1


def test_invalidate_frees_replica():
    pool = make_pool(capacity=2)
    assert sd.replicate(pool, torch.tensor([1, 2], device=DEV)) == 2
    assert sd.replicate(pool, torch.tensor([3], device=DEV)) == 0  # full
    sd.invalidate(torch.tensor([2, 50], device=DEV))  # 50 has no replica
    r = sd.rep()
    assert int(r["map"][2]) == -1 and int(r["map"][1]) >= 0
    n = sd.replicate(pool, torch.tensor([3], device=DEV))
    if n == 0:  # the free-count refresh may still be in flight on a GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        n = sd.replicate(pool, torch.tensor([3], device=DEV))
    assert n == 1
    assert sd.stats()["replicas_now"] == 2


def test_full_store_skips_device_work(monkeypatch):
    pool = make_pool(capacity=2)
    assert sd.replicate(pool, torch.tensor([1, 2], device=DEV)) == 2
    assert sd.replicate(pool, torch.tensor([3], device=DEV)) == 0  # sees full, sets the hint
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def boom(*a, **k):
        raise AssertionError("nonzero called while the store is known full")

    monkeypatch.setattr(torch, "nonzero", boom)
    for _ in range(3):
        assert sd.replicate(pool, torch.tensor([4, 5], device=DEV)) == 0


@pytest.mark.parametrize("mode", ["full", "sig"])
def test_decode_restores_double_flip_bitwise(mode):
    pool = make_pool()
    q, idx, indptr = batch()
    lid = 1
    clean = decode(pool, lid, q, indptr, idx, mode=mode, wb=True)
    s_k, s_v = int(idx[10]), int(idx[60])
    sd.replicate(pool, torch.stack([idx[10], idx[60]]))
    k0, v0 = pool.k_buffer[lid].clone(), pool.v_buffer[lid].clone()
    flip2(pool.k_buffer[lid], s_k, 1)
    flip2(pool.v_buffer[lid], s_v, 0)
    ss._state["err"].zero_()
    got = decode(pool, lid, q, indptr, idx, mode=mode, wb=True)
    assert torch.equal(clean[0], got[0]) and torch.equal(clean[1], got[1])
    err = ss.errors()
    assert err[10] >= 2 and err[1] >= 1 and err[3] >= 1  # detected uncorrectable, then restored
    assert int(ss._state["logn"].item()) == 0  # nothing for module C
    assert torch.equal(pool.k_buffer[lid], k0) and torch.equal(pool.v_buffer[lid], v0)  # written back


def test_decode_without_replica_still_logs():
    pool = make_pool()
    q, idx, indptr = batch()
    flip2(pool.k_buffer[0], int(idx[10]), 1)
    decode(pool, 0, q, indptr, idx, mode="sig", wb=True)
    assert ss.errors()[10] == 0
    assert int(ss._state["logn"].item()) >= 1


def test_prefix_check_restores_in_place():
    pool = make_pool()
    for lid in range(LAYERS):
        ss._state["by_ptr"][pool.k_buffer[lid].data_ptr()] = (pool.rm_k_cw[lid], pool.rm_v_cw[lid], lid)
    idx = torch.arange(20, 60, device=DEV)
    indptr = torch.tensor([0, idx.numel()], dtype=torch.int32, device=DEV)
    sd.replicate(pool, idx)
    v0 = pool.v_buffer[1].clone()
    flip2(pool.v_buffer[1], 33, 1)
    ss._state["err"].zero_()
    assert ss.check_prefix(pool.k_buffer[1], pool.v_buffer[1], idx, indptr)
    assert torch.equal(pool.v_buffer[1], v0)
    assert ss.errors()[11] == 1 and int(ss._state["logn"].item()) == 0


def test_first_visit_once_per_node():
    from types import SimpleNamespace

    from radixmend import sglang_replica as sr

    sr._state["seen"] = set()
    a, b = SimpleNamespace(id=10**9 + 1), SimpleNamespace(id=10**9 + 2)
    before = sr._state["stats"]["match_skipped"]
    assert sr.first_visit(a) and sr.first_visit(b)
    assert not sr.first_visit(a) and not sr.first_visit(SimpleNamespace(id=a.id))
    assert sr._state["stats"]["match_skipped"] == before + 2
    assert sr.first_visit(object())  # no id: always processed


def test_first_visit_accepts_int_node_ids():
    # SGLang 0.5.22 MatchResult.last_device_node is the node id itself
    from radixmend import sglang_replica as sr

    sr._state["seen"] = set()
    assert sr.first_visit(10**9 + 7)
    assert not sr.first_visit(10**9 + 7)


def test_first_visit_cleared_on_attach():
    from types import SimpleNamespace

    from radixmend import sglang_replica as sr

    sr._state["seen"] = {123}
    assert not sr.first_visit(SimpleNamespace(id=123))
    sr._state["seen"] = set()  # what attach_pool does
    assert sr.first_visit(SimpleNamespace(id=123))
