"""Module B codec and the req_to_token gather check (CPU / TRITON_INTERPRET=1)."""
import os

import pytest
import torch

from radixmend import sglang_index as sb
from radixmend.secded_index import IndexCode


def _reset():
    sb._state["dev"] = None
    sb._tables(torch.device("cpu"))


def test_code_matches_reference():
    _reset()
    ref = IndexCode(data_bits=32, word_bits=64)
    d = torch.randint(0, 2**31 - 1, (200,), dtype=torch.int64)
    h = sb.code_of(d) & 63
    for x, c in zip(d.tolist(), h.tolist()):
        want = 0
        for k in range(sb.R):
            want |= (bin(x & int(ref.col_masks[k])).count("1") & 1) << k
        assert c == want


def test_clean_roundtrip_and_plain_passthrough():
    _reset()
    v = torch.randint(1, 180_000, (64,), dtype=torch.int64)
    e = sb.encode_value(v)
    assert ((e & sb.MARK) != 0).all()
    out = sb.check_value_(e.clone())
    assert torch.equal(out, v)
    assert torch.equal(sb.check_value_(v.clone()), v)  # unencoded words pass through
    assert sum(sb.stats(reset=True).values()) == 0


@pytest.mark.parametrize("bit", list(range(32)) + list(range(32, 39)))
def test_single_bit_corrected(bit):
    _reset()
    v = torch.randint(1, 180_000, (16,), dtype=torch.int64)
    e = sb.encode_value(v)
    good = e.clone()
    e[3] ^= 1 << bit
    out = sb.check_value_(e)
    assert torch.equal(out, v)
    assert torch.equal(e, good)  # fixed in place
    st = sb.stats(reset=True)
    if bit < 32:
        assert st["node_corrected"] == 1 and st["node_check_bits_fixed"] == 0
    else:
        assert st["node_corrected"] == 0 and st["node_check_bits_fixed"] == 1
    assert st["node_uncorrectable"] == 0


def test_double_bit_detected():
    _reset()
    gen = torch.Generator().manual_seed(1)
    v = torch.randint(1, 180_000, (200,), dtype=torch.int64, generator=gen)
    e = sb.encode_value(v)
    for i in range(200):
        a, b = torch.randperm(39, generator=gen)[:2].tolist()
        e[i] ^= (1 << a) | (1 << b)
    sb.check_value_(e)
    assert sb.stats(reset=True)["node_uncorrectable"] == 200


def test_marker_flip_reads_plain():
    _reset()
    v = torch.tensor([12345], dtype=torch.int64)
    e = sb.encode_value(v)
    e ^= sb.MARK
    assert int(sb.check_value_(e)) == 12345


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(),
                    reason="needs GPU or TRITON_INTERPRET=1")
def test_check_gather_fixes_out_and_table():
    _reset()
    dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1" else "cpu")
    sb._state["dev"] = None
    sb._tables(dev)
    rows_n, width = 6, 700
    r2t = torch.randint(1, 180_000, (rows_n, width), dtype=torch.int32, device=dev)
    shadow = sb.shadow_encode(r2t)
    rows = torch.tensor([4, 1, 2], dtype=torch.int64, device=dev)
    lens = torch.tensor([600, 5, 33], dtype=torch.int32, device=dev)
    start = torch.tensor([0, 10, 3], dtype=torch.int64, device=dev)
    indptr = torch.zeros(4, dtype=torch.int32, device=dev)
    indptr[1:] = torch.cumsum(lens, 0)
    clean = r2t.clone()
    # one data flip, one check-bit flip, one double flip
    r2t[4, 520] ^= 1 << 7
    shadow[1, 12] ^= 1 << 3
    r2t[2, 3 + 20] ^= (1 << 1) | (1 << 9)
    out = torch.cat([r2t[r, s:s + n] for r, s, n in zip(rows.tolist(), start.tolist(), lens.tolist())]).to(torch.int64)
    sb.check_gather(r2t, shadow, rows, lens, indptr, start, out)
    want = torch.cat([clean[r, s:s + n] for r, s, n in zip(rows.tolist(), start.tolist(), lens.tolist())]).to(torch.int64)
    bad = int(indptr[2]) + 20
    keep = torch.ones_like(want, dtype=torch.bool)
    keep[bad] = False
    assert torch.equal(out[keep], want[keep])
    assert int(r2t[4, 520]) == int(clean[4, 520])
    assert torch.equal(shadow[1], sb.shadow_encode(clean[1]))
    st = sb.stats(reset=True)
    assert st["r2t_corrected"] == 1 and st["r2t_check_bits_fixed"] == 1 and st["r2t_uncorrectable"] == 1


def test_kernels_match_reference_paths():
    """Under TRITON_INTERPRET=1 the public functions run the fused kernels; compare with the torch refs."""
    _reset()
    gen = torch.Generator().manual_seed(7)
    v = torch.randint(1, 2**31 - 1, (3000,), dtype=torch.int64, generator=gen)
    assert torch.equal(sb.encode_value(v), sb._encode_value_ref(v))
    assert torch.equal(sb.shadow_encode(v.to(torch.int32)), sb._shadow_encode_ref(v.to(torch.int32)))
    e = sb.encode_value(v)
    flips = torch.randint(0, 39, (3000,), generator=gen)
    two = torch.rand(3000, generator=gen) < 0.1
    flips2 = (flips + 1 + torch.randint(0, 37, (3000,), generator=gen)) % 39
    bad = e ^ (1 << flips) ^ torch.where(two, 1 << flips2, torch.zeros_like(flips))
    a, b = bad.clone(), bad.clone()
    pa = sb.check_value_(a)
    sa = sb.stats(reset=True)
    pb = sb._check_value_ref(b)
    sbb = sb.stats(reset=True)
    assert torch.equal(pa, pb) and torch.equal(a, b) and sa == sbb
    assert sa["node_uncorrectable"] == int(two.sum())


def test_check_once_per_step():
    _reset()
    v = sb.encode_value(torch.arange(1, 65, dtype=torch.int64))
    p1 = sb.check_value_once(v)
    v[5] ^= 1 << 3  # flip after the first check in the same step: reused copy, not rechecked
    assert sb.check_value_once(v) is p1
    sb.next_step()
    p2 = sb.check_value_once(v)
    assert int(p2[5]) == 6 and sb.stats(reset=True)["node_corrected"] == 1
