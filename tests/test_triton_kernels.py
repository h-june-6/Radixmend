"""Kernel tests; on a machine without a GPU run with TRITON_INTERPRET=1."""
import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)

from radixmend.kernels.secded_triton import TritonVecCode
from radixmend.secded_vec import CLEAN, CORRECTED_CHECK, CORRECTED_DATA, DETECTED, VecCode

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tc():
    return TritonVecCode(VecCode(), device=DEV)


def kv(n=32, h=4, d=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, h, d, generator=g).to(torch.bfloat16).to(DEV)


def test_encode_matches_reference(tc):
    x = kv()
    cw = tc.encode(x).cpu().numpy().view(np.uint16)
    ref = tc.code.encode(x.cpu().view(torch.int16).numpy().view(np.uint16))
    assert np.array_equal(cw, ref)


def test_check_load_corrects_and_logs(tc):
    buf = kv(n=64)
    cw = tc.encode(buf)
    clean = buf.clone()
    iv = buf.view(torch.int16)
    # slot 3 head 1: data bit; slot 10 head 0: checkword bit; slot 20 head 2: double
    iv[3, 1, 77] ^= 1 << 9
    cw[10, 0] ^= 1 << 4
    iv[20, 2, 5] ^= 1
    iv[20, 2, 6] ^= 1 << 14
    loc = torch.tensor([3, 10, 20, 3, 40], dtype=torch.int64, device=DEV)
    out, status, log, count = tc.check_load(buf, cw, loc)
    st = status.cpu().numpy()
    assert st[0, 1] == CORRECTED_DATA and st[3, 1] == CORRECTED_DATA
    assert st[1, 0] == CORRECTED_CHECK and st[2, 2] == DETECTED
    assert (st[4] == CLEAN).all()
    assert torch.equal(out[0].view(torch.int16), clean[3].view(torch.int16))  # fixed in registers
    assert not torch.equal(buf[3], clean[3])  # memory untouched
    assert count.item() == 4
    n = count.item()
    res = tc.scrub(buf, cw, log, n).cpu().numpy()
    assert torch.equal(buf[3], clean[3])  # memory restored
    assert torch.equal(cw[10], tc.encode(clean)[10])
    # the second log entry for slot 3 sees an already-fixed vector or fixes it identically
    assert set(res.tolist()) <= {CLEAN, CORRECTED_DATA, CORRECTED_CHECK, DETECTED}
    assert (res == DETECTED).sum() == 1
    out2, status2, _, count2 = tc.check_load(buf, cw, loc)
    assert count2.item() == 1  # only the double-error vector is still logged


def test_all_single_flips_one_vector(tc):
    buf = kv(n=1, h=1)
    cw = tc.encode(buf)
    nb = 128 * 16 + 15
    bufs = buf.repeat(nb, 1, 1).contiguous()
    cws = cw.repeat(nb, 1).contiguous()
    iv = bufs.view(torch.int16)
    for p in range(nb):
        if p < 2048:
            iv[p, 0, p // 16] ^= np.int16(np.uint16(1 << (p % 16)).view(np.int16)).item()
        else:
            cws[p, 0] ^= 1 << (p - 2048)
    loc = torch.arange(nb, device=DEV)
    out, status, _, count = tc.check_load(bufs, cws, loc, log_cap=nb)
    assert (status[:2048] == CORRECTED_DATA).all() and (status[2048:] == CORRECTED_CHECK).all()
    assert torch.equal(out[:2048].view(torch.int16), buf.view(torch.int16).repeat(2048, 1, 1))
