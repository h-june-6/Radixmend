"""SECDED for slot-index tensors in their unused high bits (module B).

A slot index needs ``data_bits`` bits (23 for a 2**23-slot KV pool). The rest of
the int32/int64 word is free, so we store a systematic extended Hamming code
there and keep the sign bit clear:

    bits 0 .. D-1          slot index, unchanged (so ``v & data_mask`` is the slot)
    bits D .. D+r-1        Hamming check bits
    bit  D+r               overall parity of bits 0 .. D+r-1
    bits above D+r         must be zero; flips there are cleared on decode

Keeping the data bits in place means a choke point can decode with a mask and a
few XORs, and code that forgets to decode at least sees an out-of-range value
instead of another request's valid slot.

int32 holds up to 25 data bits (25 + 5 + 1 = 31, sign bit untouched), i.e. a pool
of 2**25 slots. int64 holds up to 56 data bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLEAN = 0
CORRECTED_DATA = 1
CORRECTED_CHECK = 2
DETECTED = 3


def _hamming_r(data_bits: int) -> int:
    r = 1
    while (1 << r) - 1 - r < data_bits:
        r += 1
    return r


def _parity(a: np.ndarray) -> np.ndarray:
    return (np.bitwise_count(a) & 1).astype(np.int64)


@dataclass
class IndexCode:
    data_bits: int = 23
    word_bits: int = 32

    r: int = field(init=False)
    data_cols: np.ndarray = field(init=False, repr=False)
    col_masks: np.ndarray = field(init=False, repr=False)
    lut_status: np.ndarray = field(init=False, repr=False)
    lut_bit: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.r = _hamming_r(self.data_bits)
        if self.code_bits > self.word_bits - 1:
            raise ValueError(
                f"{self.data_bits} data bits need {self.code_bits} bits, "
                f"more than the {self.word_bits - 1} non-sign bits"
            )
        # Data bit i gets the i-th non-power-of-two r-bit value as its column.
        cols = [v for v in range(1, 1 << self.r) if v & (v - 1)]
        self.data_cols = np.asarray(cols[: self.data_bits], np.int64)
        # col_masks[k]: data bits whose column has bit k set.
        self.col_masks = np.array(
            [sum(1 << i for i, c in enumerate(self.data_cols) if (c >> k) & 1) for k in range(self.r)],
            np.int64,
        )
        # Syndrome -> what to do, for the overall-parity-odd case.
        n = 1 << self.r
        self.lut_status = np.full(n, DETECTED, np.int8)
        self.lut_bit = np.full(n, -1, np.int64)
        self.lut_status[0] = CORRECTED_CHECK  # only the overall parity bit flipped
        self.lut_bit[0] = self.data_bits + self.r
        for k in range(self.r):
            self.lut_status[1 << k] = CORRECTED_CHECK
            self.lut_bit[1 << k] = self.data_bits + k
        for i, c in enumerate(self.data_cols):
            self.lut_status[c] = CORRECTED_DATA
            self.lut_bit[c] = i

    @property
    def code_bits(self) -> int:
        return self.data_bits + self.r + 1

    @property
    def data_mask(self) -> int:
        return (1 << self.data_bits) - 1

    @property
    def code_mask(self) -> int:
        return (1 << self.code_bits) - 1

    @property
    def max_slots(self) -> int:
        return 1 << self.data_bits

    def _check(self, d: np.ndarray) -> np.ndarray:
        c = np.zeros(d.shape, np.int64)
        for k in range(self.r):
            c |= _parity(d & self.col_masks[k]) << k
        return c

    def encode(self, idx) -> np.ndarray:
        d = np.asarray(idx, np.int64)
        if (d < 0).any() or (d > self.data_mask).any():
            raise ValueError("index out of range for data_bits")
        c = self._check(d)
        w = d | (c << self.data_bits)
        w |= _parity(w) << (self.data_bits + self.r)
        return w

    def decode(self, w):
        """Return (index, fixed_word, status, flipped_bit).

        ``flipped_bit`` is the corrected bit position or -1. Flips in the unused
        high bits are cleared silently (status stays CLEAN unless the code bits
        also carry an error), since they hold no information.
        """
        w = np.asarray(w, np.int64) & self.code_mask
        d = w & self.data_mask
        c = (w >> self.data_bits) & ((1 << self.r) - 1)
        s = self._check(d) ^ c
        overall = _parity(w)
        status = np.where(s == 0, CLEAN, DETECTED).astype(np.int8)
        bit = np.full(w.shape, -1, np.int64)
        odd = overall == 1
        status = np.where(odd, self.lut_status[s], status)
        bit = np.where(odd, self.lut_bit[s], bit)
        fixed = np.where(bit >= 0, w ^ (np.int64(1) << np.maximum(bit, 0)), w)
        return fixed & self.data_mask, fixed, status, bit
