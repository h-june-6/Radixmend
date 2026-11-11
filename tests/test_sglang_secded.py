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
VARIANTS = [dict(mode="full"), dict(mode="gate"), dict(mode="sig")] + (
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
