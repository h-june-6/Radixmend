import numpy as np
import pytest

from radixmend.secded_vec import (
    CLEAN, CORRECTED_CHECK, CORRECTED_DATA, DETECTED, VecCode, decode_heads, encode_heads, split_code,
)


@pytest.fixture(scope="module")
def code():
    return VecCode()


def rand(rng, *shape):
    return rng.integers(0, 1 << 16, shape, dtype=np.uint16)


def test_layout(code):
    assert (code.e_bits, code.b_bits, code.check_bits) == (8, 5, 15)
    assert len(set(code.e_labels)) == 128 and 0 not in code.e_labels
    assert len(set(code.b_labels)) == 16 and 0 not in code.b_labels


def test_popcount_form_matches_definition(code):
    x = rand(np.random.default_rng(0), 6, 128)
    assert np.array_equal(code.encode(x), code.encode_bitloop(x))


def test_columns_match_real_flips(code):
    rng = np.random.default_rng(1)
    x = rand(rng, 1, 128)[0]
    cw = code.encode(x)
    cols = code.columns()
    for p in rng.choice(code.codeword_bits, 200, replace=False):
        y, c = x.copy(), cw.copy()
        if p < code.data_bits:
            y[p // 16] ^= np.uint16(1 << (p % 16))
        else:
            c ^= np.uint16(1 << (p - code.data_bits))
        assert code.syndrome(y, c) == cols[p]


def test_every_single_corrected(code):
    cols = code.columns()
    st, pos = code.lut_status[cols], code.lut_pos[cols]
    assert (pos == np.arange(code.codeword_bits)).all()
    assert (st[: code.data_bits] == CORRECTED_DATA).all()
    assert (st[code.data_bits :] == CORRECTED_CHECK).all()


def test_every_double_detected(code):
    cols = code.columns()
    j, k = np.triu_indices(len(cols), 1)
    assert (code.lut_status[cols[j] ^ cols[k]] == DETECTED).all()


def test_decode_restores_data_and_checkword(code):
    rng = np.random.default_rng(2)
    x = rand(rng, 64, 128)
    cw = code.encode(x)
    y, c = x.copy(), cw.copy()
    rows = np.arange(64)
    elem, bit = rng.integers(0, 128, 64), rng.integers(0, 16, 64)
    y[rows[:32], elem[:32]] ^= (1 << bit[:32]).astype(np.uint16)
    c[32:48] ^= (1 << rng.integers(0, 15, 16)).astype(np.uint16)
    xf, cwf, st, _ = code.decode(y, c)
    assert np.array_equal(xf, x) and np.array_equal(cwf, cw)
    assert (st[:32] == CORRECTED_DATA).all() and (st[32:48] == CORRECTED_CHECK).all()
    assert (st[48:] == CLEAN).all()


@pytest.mark.parametrize("splits", [1, 2, 4])
def test_split_heads(splits):
    code = split_code(128, splits)
    rng = np.random.default_rng(3)
    x = rand(rng, 5, 8, 128)  # [tokens, heads, head_dim]
    cw = encode_heads(code, x, splits)
    assert cw.shape == (5, 8, splits)
    y = x.copy()
    y[2, 3, 100] ^= np.uint16(1 << 7)
    xf, _, st, _ = decode_heads(code, y, cw, splits)
    assert np.array_equal(xf, x)
    assert (st == CORRECTED_DATA).sum() == 1


def test_keys_change_labels():
    a, b = VecCode(key=b"a"), VecCode(key=b"b")
    assert not np.array_equal(a.e_labels, b.e_labels)
    assert np.array_equal(a.e_labels, VecCode(key=b"a").e_labels)
