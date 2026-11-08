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


def run(q, k, v, kcw, vcw, kv_indptr, kv_indices, splits=1, check=True):
    b, hq = q.shape[:2]
    out = torch.zeros(b, hq, splits, D, dtype=torch.float32, device=DEV)  # empty splits stay 0
    lse = torch.zeros(b, hq, splits, dtype=torch.float32, device=DEV)
    nsp = torch.full((b,), splits, dtype=torch.int32, device=DEV)
    ss.launch_stage1(q, k, v, kcw, vcw, out, lse, kv_indptr, kv_indices, nsp, splits, D ** -0.5, 0.0, check)
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
    assert ss.errors() == [1, 0, 1, 0]
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
