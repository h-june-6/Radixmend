"""Module A fused decode kernel (radixmend.sglang_secded); no SGLang needed.

On a machine without a GPU run with TRITON_INTERPRET=1.
"""
import os

import pytest
import torch

if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)

from radixmend import sglang_secded as ss
from radixmend.kernels.secded_triton import TritonVecCode
from radixmend.secded_vec import VecCode

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H_KV, GROUP, D, POOL = 2, 4, 128, 256
# the interpreter mishandles bf16 in tl.dot; on the GPU test the real dtype
DT = torch.bfloat16 if torch.cuda.is_available() else torch.float16


def setup(seqs=(37, 70), splits=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(POOL, H_KV, D, generator=g).to(DT).to(DEV)
    v = torch.randn(POOL, H_KV, D, generator=g).to(DT).to(DEV)
    q = torch.randn(len(seqs), H_KV * GROUP, D, generator=g).to(DT).to(DEV)
    perm = torch.randperm(POOL, generator=g)[: sum(seqs)]
    kv_indices = perm.to(torch.int64).to(DEV)
    kv_indptr = torch.tensor([0] + list(torch.tensor(seqs).cumsum(0)), dtype=torch.int32, device=DEV)
    return q, k, v, kv_indptr, kv_indices, seqs


def run(q, k, v, kcw, vcw, kv_indptr, kv_indices, splits=1, check=True, **kw):
    b, hq = q.shape[:2]
    out = torch.zeros(b, hq, splits, D, dtype=torch.float32, device=DEV)  # empty splits stay 0
    lse = torch.zeros(b, hq, splits, dtype=torch.float32, device=DEV)
    nsp = torch.full((b,), splits, dtype=torch.int32, device=DEV)
    ss.launch_stage1(q, k, v, kcw, vcw, out, lse, kv_indptr, kv_indices, nsp, splits, D ** -0.5, 0.0, check, **kw)
    return out, lse


def encode(buf):
    ss._tables(buf.device)
    cw = torch.zeros(buf.shape[:2], dtype=torch.int16, device=DEV)
    ss.encode_rows(buf, cw, torch.arange(buf.shape[0], device=DEV))
    return cw


def test_encode_matches_reference_kernel():
    q, k, v, *_ = setup()
    ref = TritonVecCode(VecCode(n_elem=D), device=DEV).encode(k)
    assert torch.equal(encode(k), ref)


def test_clean_output_matches_torch():
    q, k, v, indptr, idx, seqs = setup()
    out, lse = run(q, k, v, encode(k), encode(v), indptr, idx)
    for b, n in enumerate(seqs):
        loc = idx[indptr[b]: indptr[b + 1]]
        kk = k[loc].float().repeat_interleave(GROUP, 1)  # [n, hq, D]
        vv = v[loc].float().repeat_interleave(GROUP, 1)
        s = torch.einsum("hd,nhd->hn", q[b].float(), kk) * D ** -0.5
        ref = torch.einsum("hn,nhd->hd", s.softmax(-1), vv)
        torch.testing.assert_close(out[b, :, 0], ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse[b, :, 0], s.logsumexp(-1), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("splits", [1, 3])
def test_single_bit_flips_are_corrected_bitwise(splits):
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    clean = run(q, k, v, kcw, vcw, indptr, idx, splits)
    ss._state["err"].zero_()
    k2, v2 = k.clone(), v.clone()
    s1, s2 = int(idx[5]), int(idx[50])
    k2.view(torch.int16)[s1, 1, 77] ^= 1 << 14
    v2.view(torch.int16)[s2, 0, 3] ^= 1 << 7
    fixed = run(q, k2, v2, kcw, vcw, indptr, idx, splits)
    assert torch.equal(clean[0], fixed[0]) and torch.equal(clean[1], fixed[1])
    assert ss.errors()[:4] == [1, 0, 1, 0]
    # without the check the flip shows up in the output
    raw = run(q, k2, v2, kcw, vcw, indptr, idx, splits, check=False)
    assert not torch.equal(clean[0], raw[0])


def test_double_flip_is_counted_uncorrectable():
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    ss._state["err"].zero_()
    k2 = k.clone()
    s1 = int(idx[10])
    k2.view(torch.int16)[s1, 0, 1] ^= 1 << 3
    k2.view(torch.int16)[s1, 0, 9] ^= 1 << 12
    run(q, k2, v, kcw, vcw, indptr, idx)
    assert ss.errors()[:2] == [0, 1]


def test_prefix_check_fixes_in_place():
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    ss._state["by_ptr"][k.data_ptr()] = (kcw, vcw, 0)
    clean_k, clean_v = k.clone(), v.clone()
    ss._state["err"].zero_()
    s1, s2, s3 = int(idx[3]), int(idx[90]), int(idx[100])
    k.view(torch.int16)[s1, 0, 5] ^= 1 << 14
    v.view(torch.int16)[s2, 1, 127] ^= 1 << 0
    v.view(torch.int16)[s3, 1, 2] ^= 1 << 1  # double error: detected, left as is
    v.view(torch.int16)[s3, 1, 60] ^= 1 << 9
    # slot outside kv_indices[:indptr[-1]] is not touched
    spare = torch.tensor(sorted(set(range(POOL)) - set(idx.tolist()))[:1], device=DEV)
    k.view(torch.int16)[int(spare), 0, 0] ^= 1
    padded = torch.cat([idx, spare])
    assert ss.check_prefix(k, v, padded, indptr)
    assert torch.equal(k[s1], clean_k[s1]) and torch.equal(v[s2], clean_v[s2])
    assert not torch.equal(v[s3], clean_v[s3])
    assert not torch.equal(k[int(spare)], clean_k[int(spare)])
    assert ss.errors()[4:8] == [1, 0, 1, 1]
    del ss._state["by_ptr"][k.data_ptr()]


# popc needs libdevice, so it only runs on the GPU
VARIANTS = [dict(mode="full"), dict(mode="gate"), dict(mode="sig"), dict(mode="sig", sigred=True)] + (
    [dict(mode="full", popc=True), dict(mode="gate", popc=True)] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("kw", VARIANTS, ids=lambda d: "-".join(f"{k}={v}" for k, v in d.items()))
def test_variants_correct_single_flips(kw):
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    clean = run(q, k, v, kcw, vcw, indptr, idx, 2, check=False)
    assert torch.equal(clean[0], run(q, k, v, kcw, vcw, indptr, idx, 2, **kw)[0])
    ss._state["err"].zero_()
    k2, v2 = k.clone(), v.clone()
    k2.view(torch.int16)[int(idx[5]), 1, 77] ^= 1 << 14
    v2.view(torch.int16)[int(idx[50]), 0, 3] ^= 1 << 7
    fixed = run(q, k2, v2, kcw, vcw, indptr, idx, 2, **kw)
    assert torch.equal(clean[0], fixed[0]) and torch.equal(clean[1], fixed[1])
    assert ss.errors()[:4] == [1, 0, 1, 0]


def test_gate_misses_same_position_double_but_full_flags_it():
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    k2 = k.clone()
    s1 = int(idx[10])
    k2.view(torch.int16)[s1, 0, 1] ^= 1 << 6
    k2.view(torch.int16)[s1, 0, 9] ^= 1 << 6
    ss._state["err"].zero_()
    run(q, k2, v, kcw, vcw, indptr, idx, mode="gate")
    assert ss.errors()[:2] == [0, 0]
    run(q, k2, v, kcw, vcw, indptr, idx, mode="full")
    assert ss.errors()[:2] == [0, 1]


def test_sample_checks_each_tile_once_per_period():
    q, k, v, indptr, idx, seqs = setup()
    kcw, vcw = encode(k), encode(v)
    k2 = k.clone()
    k2.view(torch.int16)[int(idx[5]), 1, 77] ^= 1 << 14  # tile 0 of request 0
    P = 3
    hits = []
    for step in range(P):
        ss._state["phase"].fill_(step)
        ss._state["err"].zero_()
        run(q, k2, v, kcw, vcw, indptr, idx, sample=P)
        hits.append(ss.errors()[0])
    assert sorted(hits) == [0, 0, 1]


def test_sig_redoes_only_bad_splits():
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    ss._state["err"].zero_()
    run(q, k, v, kcw, vcw, indptr, idx, 3, mode="sig")
    assert ss.errors()[8] == 0
    v2 = v.clone()
    v2.view(torch.int16)[int(idx[50]), 0, 3] ^= 1 << 7  # request 1, one split
    run(q, k, v2, kcw, vcw, indptr, idx, 3, mode="sig")
    e = ss.errors()
    assert e[8] == 1 and e[2] == 1


@pytest.mark.parametrize("sigred", [False, True])
def test_sig_flags_same_bit_double_in_one_vector(sigred):
    # p and S_b of the split total cancel; S_e of the per-element totals does not
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    for buf in ("k", "v"):
        k2, v2 = k.clone(), v.clone()
        b = k2 if buf == "k" else v2
        s1 = int(idx[10])
        b.view(torch.int16)[s1, 0, 1] ^= 1 << 14
        b.view(torch.int16)[s1, 0, 9] ^= 1 << 14
        ss._state["err"].zero_()
        run(q, k2, v2, kcw, vcw, indptr, idx, mode="sig", sigred=sigred)
        e = ss.errors()
        assert e[8] >= 1 and e[1 if buf == "k" else 3] == 1, e


@pytest.mark.parametrize("sigred", [False, True])
def test_sig_flags_two_single_flips_same_bit_two_vectors(sigred):
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    v2 = v.clone()
    v2.view(torch.int16)[int(idx[3]), 1, 5] ^= 1 << 7     # request 0, two vectors, same bit
    v2.view(torch.int16)[int(idx[20]), 1, 60] ^= 1 << 7
    ss._state["err"].zero_()
    out, _ = run(q, k, v2, kcw, vcw, indptr, idx, mode="sig", sigred=sigred)
    ref, _ = run(q, k, v, kcw, vcw, indptr, idx, mode="full")
    e = ss.errors()
    assert e[8] >= 1 and e[2] == 2, e
    assert torch.equal(out, ref)


@pytest.mark.parametrize("mode", ["full", "gate", "sig"])
def test_write_back_fixes_buffer_once(mode):
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    clean = run(q, k, v, kcw, vcw, indptr, idx, 2, check=False)
    k2, v2 = k.clone(), v.clone()
    k2.view(torch.int16)[int(idx[5]), 1, 77] ^= 1 << 14
    v2.view(torch.int16)[int(idx[50]), 0, 3] ^= 1 << 7
    ss._state["err"].zero_()
    out = run(q, k2, v2, kcw, vcw, indptr, idx, 2, mode=mode, wb=True)
    assert torch.equal(clean[0], out[0])
    assert ss.errors()[:4] == [1, 0, 1, 0]
    assert torch.equal(k2.view(torch.int16), k.view(torch.int16))
    assert torch.equal(v2.view(torch.int16), v.view(torch.int16))
    ss._state["err"].zero_()
    run(q, k2, v2, kcw, vcw, indptr, idx, 2, mode=mode, wb=True)
    assert ss.errors()[:4] == [0, 0, 0, 0]


def test_no_write_back_leaves_buffer():
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    k2 = k.clone()
    k2.view(torch.int16)[int(idx[5]), 1, 77] ^= 1 << 14
    run(q, k2, v, kcw, vcw, indptr, idx, mode="gate")
    assert not torch.equal(k2.view(torch.int16), k.view(torch.int16))


@pytest.mark.parametrize("mode", ["full", "sig"])
def test_checkword_error_counted_rewritten_data_kept(mode):
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    clean = run(q, k, v, kcw, vcw, indptr, idx, 2, check=False)
    good = kcw.clone()
    s = int(idx[5])
    kcw[s, 1] = (kcw[s, 1].to(torch.int32) ^ 1).to(torch.int16)  # parity bit of the stored checkword
    ss._state["err"].zero_()
    out = run(q, k, v, kcw, vcw, indptr, idx, 2, mode=mode, wb=True)
    assert torch.equal(clean[0], out[0])
    e = ss.errors()
    assert e[:4] == [0, 0, 0, 0] and e[9] == 1
    assert torch.equal(kcw, good)
    ss._state["err"].zero_()
    run(q, k, v, kcw, vcw, indptr, idx, 2, mode=mode, wb=True)
    assert ss.errors()[9] == 0 and ss.errors()[8] == 0


def test_prefix_check_rewrites_bad_checkword():
    q, k, v, indptr, idx, _ = setup()
    kcw = encode(k)
    good = kcw.clone()
    s = int(idx[3])
    kcw[s, 0] = (kcw[s, 0].to(torch.int32) ^ (1 << 3)).to(torch.int16)
    before = k.clone()
    ss._state["err"].zero_()
    ss.check_fix_inplace(k, kcw, idx, indptr, 4)
    assert ss.errors()[4:6] == [0, 0] and ss.errors()[9] == 1
    assert torch.equal(kcw, good) and torch.equal(k.view(torch.int16), before.view(torch.int16))


def shared_setup(shared=40, own=(20, 33), seed=5):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(POOL, H_KV, D, generator=g).to(DT).to(DEV)
    v = torch.randn(POOL, H_KV, D, generator=g).to(DT).to(DEV)
    q = torch.randn(len(own), H_KV * GROUP, D, generator=g).to(DT).to(DEV)
    perm = torch.randperm(POOL, generator=g)
    pre = perm[:shared]
    rows, off = [], shared
    for o in own:
        rows.append(torch.cat([pre, perm[off:off + o]]))
        off += o
    idx = torch.cat(rows).to(torch.int64).to(DEV)
    lens = [shared + o for o in own]
    indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=DEV)
    return q, k, v, indptr, idx, pre


@pytest.mark.parametrize("sigred", [False, True])
def test_dedup_checks_shared_slots_once(sigred):
    q, k, v, indptr, idx, pre = shared_setup()
    kcw, vcw = encode(k), encode(v)
    ss._state["count"] = torch.zeros(POOL, dtype=torch.int32, device=DEV)
    ss._state["owner"] = torch.zeros(POOL, dtype=torch.int32, device=DEV)
    clean = run(q, k, v, kcw, vcw, indptr, idx, 2, check=False)
    k2, v2 = k.clone(), v.clone()
    k2.view(torch.int16)[int(pre[7]), 1, 77] ^= 1 << 14      # shared by both requests
    v2.view(torch.int16)[int(idx[45]), 0, 3] ^= 1 << 7       # request 0 only
    ss._state["err"].zero_()
    ss.count_readers(idx, indptr)
    assert int(ss._state["count"][pre[7]]) == 2 and int(ss._state["count"][idx[45]]) == 1
    ss.shared_prepass(k2, v2, kcw, vcw, idx, indptr)
    assert ss.errors()[:4] == [1, 0, 0, 0]                    # shared flip fixed once, in place
    assert torch.equal(k2.view(torch.int16), k.view(torch.int16))
    out = run(q, k2, v2, kcw, vcw, indptr, idx, 2, mode="sig", wb=True, dedup=True, sigred=sigred)
    assert torch.equal(clean[0], out[0]) and torch.equal(clean[1], out[1])
    e = ss.errors()
    assert e[:4] == [1, 0, 1, 0] and e[8] == 1                # private flip via one sig redo
    ss._state["err"].zero_()
    run(q, k2, v2, kcw, vcw, indptr, idx, 2, mode="sig", wb=True, dedup=True, sigred=sigred)
    assert ss.errors()[8] == 0


def test_error_log_records_uncorrectable():
    q, k, v, indptr, idx, _ = setup()
    kcw, vcw = encode(k), encode(v)
    s1 = int(idx[10])
    k2 = k.clone()
    k2.view(torch.int16)[s1, 1, 1] ^= 1 << 6
    k2.view(torch.int16)[s1, 1, 9] ^= 1 << 3
    ss.error_log(reset=True)
    run(q, k2, v, kcw, vcw, indptr, idx, mode="full", layer=7)
    log = ss.error_log(reset=True)
    assert log["total"] == 1
    assert log["entries"][0] == {"slot": s1, "head": 1, "layer": 7, "kind": "k", "path": "decode"}
    v2 = v.clone()
    v2.view(torch.int16)[s1, 0, 5] ^= 1 << 14
    v2.view(torch.int16)[s1, 0, 6] ^= 1 << 14
    ss.check_fix_inplace(v2, vcw, idx, indptr, 6, tag=3 * 4 + 3)
    log = ss.error_log(reset=True)
    assert log["entries"] == [{"slot": s1, "head": 0, "layer": 3, "kind": "v", "path": "extend"}]
    assert ss.error_log()["total"] == 0


@pytest.mark.parametrize("block", [1, 16, 32])
def test_blocked_prefix_check_matches_reference(block):
    _, k, _, indptr, idx, _ = setup(seqs=(37, 70))
    kcw = encode(k)
    k16 = k.view(torch.int16)
    s = [int(idx[i]) for i in (0, 5, 17, 40, 99)]
    k16[s[0], 0, 3] ^= 1 << 14                       # 1-bit data
    k16[s[1], 1, 127] ^= 1 << 0
    k16[s[2], 0, 10] ^= 1 << 7                       # 2-bit (uncorrectable)
    k16[s[2], 0, 11] ^= 1 << 7
    kcw[s[3], 1] = (kcw[s[3], 1].to(torch.int32) ^ (1 << 5)).to(torch.int16)  # checkword hit
    k16[s[4], 1, 64] ^= 1 << 9
    outs = []
    for b in (0, block):
        kb, cb = k.clone(), kcw.clone()
        ss._state["err"].zero_()
        ss.error_log(reset=True)
        ss.check_fix_inplace(kb, cb, idx, indptr, 4, tag=5, block=b)
        outs.append((kb.view(torch.int16).clone(), cb.clone(), ss.errors(), ss.error_log(reset=True)["entries"]))
    ref, got = outs
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    assert ref[2] == got[2] and ref[2][4:6] == [3, 1] and ref[2][9] == 1
    assert ref[3] == got[3] and len(ref[3]) == 1 and ref[3][0]["slot"] == s[2]
