import numpy as np
import pytest

from radixmend.secded_index import CLEAN, CORRECTED_CHECK, CORRECTED_DATA, DETECTED, IndexCode


@pytest.mark.parametrize("d,w", [(23, 32), (25, 32), (40, 64), (56, 64)])
def test_fits_without_sign(d, w):
    c = IndexCode(d, w)
    assert c.code_bits <= w - 1
    v = np.random.default_rng(0).integers(0, 1 << d, 1000, dtype=np.int64)
    enc = c.encode(v)
    assert (enc >= 0).all() and (enc & c.data_mask == v).all()


def test_too_many_bits():
    with pytest.raises(ValueError):
        IndexCode(26, 32)


def test_single_and_double():
    c = IndexCode(23, 32)
    v = np.random.default_rng(1).integers(0, 1 << 23, 500, dtype=np.int64)
    w = c.encode(v)
    for b in range(31):
        idx, fixed, st, bit = c.decode(w ^ (1 << b))
        assert (idx == v).all()
        if b < c.code_bits:
            assert (st != CLEAN).all() and (st != DETECTED).all() and (bit == b).all()
            assert (fixed == w).all()
    for b1 in range(c.code_bits):
        for b2 in range(b1 + 1, c.code_bits):
            assert (c.decode(w ^ (1 << b1) ^ (1 << b2))[2] == DETECTED).all()


def test_status_kinds():
    c = IndexCode(23, 32)
    w = c.encode(np.array([12345]))
    assert c.decode(w ^ 1)[2][0] == CORRECTED_DATA
    assert c.decode(w ^ (1 << 23))[2][0] == CORRECTED_CHECK
    assert c.decode(w ^ (1 << 28))[2][0] == CORRECTED_CHECK
