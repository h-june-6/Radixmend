"""Keyed secret label permutations (module A).

Each data bit of a KV head vector is labelled (e, b): e for the element index,
b for the bit position inside the element. Labels are drawn from the non-zero
values of an E-bit / B-bit space with a keyed permutation, so that

* label 0 never appears (a data bit with label 0 would collide with the P
  check bit), and
* the mapping is unknown to an attacker who does not hold the key, which turns
  a 3+ bit flip into a probabilistic miscorrection instead of a chosen one.

The permutation is derived from BLAKE2b in keyed mode, so the same key always
gives the same labels on any machine.
"""

from __future__ import annotations

import hashlib

import numpy as np


def bits_needed(n: int) -> int:
    """Smallest width w such that the non-zero values 1..2**w-1 hold n labels."""
    w = 1
    while (1 << w) - 1 < n:
        w += 1
    return w


def _keyed_stream(key: bytes, domain: bytes):
    counter = 0
    while True:
        block = hashlib.blake2b(
            domain + counter.to_bytes(8, "little"), key=key[:64], digest_size=64
        ).digest()
        for i in range(0, 64, 8):
            yield int.from_bytes(block[i : i + 8], "little")
        counter += 1


def keyed_labels(n: int, width: int, key: bytes, domain: bytes) -> np.ndarray:
    """Pick n distinct non-zero width-bit labels with a keyed Fisher-Yates shuffle."""
    space = (1 << width) - 1
    if n > space:
        raise ValueError(f"{n} labels do not fit in {width} bits without zero")
    pool = list(range(1, space + 1))
    stream = _keyed_stream(key, domain)
    for i in range(space - 1, 0, -1):
        # Rejection sampling keeps the shuffle unbiased.
        bound = (1 << 64) - ((1 << 64) % (i + 1))
        r = next(stream)
        while r >= bound:
            r = next(stream)
        j = r % (i + 1)
        pool[i], pool[j] = pool[j], pool[i]
    return np.asarray(pool[:n], dtype=np.int64)


def identity_labels(n: int) -> np.ndarray:
    """Labels 1..n, for tests and for the no-secret ablation."""
    return np.arange(1, n + 1, dtype=np.int64)
