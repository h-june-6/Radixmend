"""Exhaustive verification of the module A vector SECDED code.

Because the code is linear over GF(2), the decoder's verdict for an error
pattern depends only on its syndrome, which is the XOR of the parity-check
matrix columns of the flipped positions. That lets us check every single,
every double and every triple error over the full codeword (data + check
bits) without touching data:

* single: every position must be corrected at exactly that position
* double: every pair must be DETECTED (never clean, never "corrected")
* triple: counted per outcome; any corrected verdict is a silent miscorrection

Usage: python scripts/verify_secded_exhaustive.py [--splits 1 2 4] [--keys 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radixmend.secded_vec import (  # noqa: E402
    CLEAN,
    CORRECTED_CHECK,
    CORRECTED_DATA,
    DETECTED,
    VecCode,
)


def check_singles(code: VecCode, cols: np.ndarray) -> dict:
    status = code.lut_status[cols]
    pos = code.lut_pos[cols]
    n_data = code.data_bits
    want_status = np.where(np.arange(len(cols)) < n_data, CORRECTED_DATA, CORRECTED_CHECK)
    ok = (status == want_status) & (pos == np.arange(len(cols)))
    return {"total": int(len(cols)), "correct": int(ok.sum()), "all_correct": bool(ok.all())}


def check_doubles(code: VecCode, cols: np.ndarray) -> dict:
    n = len(cols)
    j, k = np.triu_indices(n, 1)
    status = code.lut_status[cols[j] ^ cols[k]]
    counts = np.bincount(status, minlength=4)
    return {
        "total": int(len(j)),
        "detected": int(counts[DETECTED]),
        "clean": int(counts[CLEAN]),
        "miscorrected": int(counts[CORRECTED_DATA] + counts[CORRECTED_CHECK]),
        "all_detected": bool(counts[DETECTED] == len(j)),
    }


def count_triples(code: VecCode, cols: np.ndarray) -> dict:
    n = len(cols)
    j, k = np.triu_indices(n, 1)  # row-major: pairs with j > i form a suffix
    pair_syn = (cols[j] ^ cols[k]).astype(np.int32)
    starts = np.searchsorted(j, np.arange(n), side="right")
    lut = code.lut_status
    counts = np.zeros(4, np.int64)
    for i in range(n - 2):
        s = pair_syn[starts[i] :] ^ cols[i]
        counts += np.bincount(lut[s], minlength=4)
    total = int(counts.sum())
    mis = int(counts[CORRECTED_DATA] + counts[CORRECTED_CHECK])
    return {
        "total": total,
        "detected": int(counts[DETECTED]),
        "miscorrected_data": int(counts[CORRECTED_DATA]),
        "miscorrected_check": int(counts[CORRECTED_CHECK]),
        "clean": int(counts[CLEAN]),
        "miscorrection_rate": mis / total,
    }


def end_to_end(code: VecCode, rng: np.random.Generator, vectors: int, doubles: int) -> dict:
    """Real data through encode/decode: all single flips, sampled doubles."""
    x = rng.integers(0, 1 << code.elem_bits, (vectors, code.n_elem), dtype=np.uint16)
    cw = code.encode(x)
    assert np.array_equal(cw[:4], code.encode_bitloop(x[:4])), "popcount form != definition"
    assert (code.decode(x, cw)[2] == CLEAN).all()
    nb = code.codeword_bits
    fails = 0
    for v in range(vectors):
        xs = np.repeat(x[v : v + 1], nb, 0)
        cws = np.repeat(cw[v : v + 1], nb, 0)
        for p in range(nb):
            if p < code.data_bits:
                xs[p, p // code.elem_bits] ^= np.uint16(1 << (p % code.elem_bits))
            else:
                cws[p] ^= np.uint16(1 << (p - code.data_bits))
        xf, cwf, st, _ = code.decode(xs, cws)
        fails += int((~(xf == x[v]).all(-1) | (cwf != cw[v])).sum())
    # sampled doubles
    v = rng.integers(0, vectors, doubles)
    p = np.stack([rng.choice(nb, 2, replace=False) for _ in range(doubles)])
    xs, cws = x[v].copy(), cw[v].copy()
    for r in range(doubles):
        for q in p[r]:
            if q < code.data_bits:
                xs[r, q // code.elem_bits] ^= np.uint16(1 << (q % code.elem_bits))
            else:
                cws[r] ^= np.uint16(1 << (q - code.data_bits))
    st = code.decode(xs, cws)[2]
    return {
        "vectors": vectors,
        "single_flips": vectors * nb,
        "single_fail": fails,
        "double_samples": doubles,
        "double_detected": int((st == DETECTED).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--keys", type=int, default=4, help="random keys per config for triples")
    ap.add_argument("--e-bits", type=int, default=None, help="wider e-label space")
    ap.add_argument("--out", default="results/secded_exhaustive.json")
    args = ap.parse_args()

    rng = np.random.default_rng(1234)
    report = []
    for splits in args.splits:
        n_elem = args.head_dim // splits
        keys = [None] + [f"key-{i}".encode() for i in range(args.keys)]
        for key in keys:
            t0 = time.time()
            code = VecCode(n_elem=n_elem, key=key, e_bits_override=args.e_bits)
            cols = code.columns()
            row = {
                "splits": splits,
                "n_elem": n_elem,
                "key": "identity" if key is None else key.decode(),
                "e_bits": code.e_bits,
                "b_bits": code.b_bits,
                "check_bits": code.check_bits,
                "codeword_bits": code.codeword_bits,
                "overhead_vs_data": splits * 16 / (args.head_dim * 16),
                "single": check_singles(code, cols),
                "double": check_doubles(code, cols),
                "triple": count_triples(code, cols),
            }
            if key is not None and key.endswith(b"0"):
                row["end_to_end"] = end_to_end(code, rng, vectors=3, doubles=2000)
            row["seconds"] = round(time.time() - t0, 1)
            report.append(row)
            print(json.dumps(row), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    ok = all(r["single"]["all_correct"] and r["double"]["all_detected"] for r in report)
    print("ALL SINGLE CORRECTED, ALL DOUBLE DETECTED:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
