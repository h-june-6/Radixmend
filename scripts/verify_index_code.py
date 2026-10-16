"""Exhaustive verification of the module B slot-index SECDED code.

* every index value 0 .. 2**D-1 round-trips through encode/decode as CLEAN
* every single, double and triple flip pattern over the whole word (code bits,
  unused high bits and the sign bit) on a sample of index values:
  - single flip in code bits  -> corrected to the right index
  - single flip outside       -> cleared, right index
  - double flip in code bits  -> detected, never a wrong index reported as fine
  - triples                   -> counted; a wrong index with a non-DETECTED
                                 status is a silent miscorrection

Usage: python scripts/verify_index_code.py [--data-bits 23] [--word-bits 32]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radixmend.secded_index import CLEAN, DETECTED, IndexCode  # noqa: E402


def run(data_bits: int, word_bits: int, samples: int, rng) -> dict:
    code = IndexCode(data_bits, word_bits)
    out = {"data_bits": data_bits, "word_bits": word_bits, "r": code.r, "code_bits": code.code_bits}

    # all values round trip (chunked for 2**25)
    bad = 0
    step = 1 << 22
    for lo in range(0, 1 << data_bits, step):
        v = np.arange(lo, min(lo + step, 1 << data_bits), dtype=np.int64)
        idx, _, st, _ = code.decode(code.encode(v))
        bad += int(((idx != v) | (st != CLEAN)).sum())
    out["roundtrip_values"] = 1 << data_bits
    out["roundtrip_bad"] = bad

    vals = rng.integers(0, 1 << data_bits, samples, dtype=np.int64)
    words = code.encode(vals)
    sign_mask = np.int64((1 << word_bits) - 1)
    for k in (1, 2, 3):
        res = {"patterns": 0, "right_index": 0, "detected": 0, "silent_wrong": 0, "in_code_detect_fail": 0}
        for bits in itertools.combinations(range(word_bits), k):
            m = 0
            for b in bits:
                m |= 1 << b
            w = (words ^ np.int64(m)) & sign_mask
            idx, _, st, _ = code.decode(w)
            in_code = sum(1 for b in bits if b < code.code_bits)
            res["patterns"] += samples
            right = idx == vals
            res["right_index"] += int(right.sum())
            res["detected"] += int((st == DETECTED).sum())
            res["silent_wrong"] += int(((~right) & (st != DETECTED)).sum())
            if in_code == 2:  # a true double inside the code must always be caught
                res["in_code_detect_fail"] += int((st != DETECTED).sum())
            if in_code <= 1:  # effective single (or none) must give the right index
                res["in_code_detect_fail"] += int((~right).sum())
        out[f"flips_{k}"] = res
    out["ok"] = bool(
        bad == 0
        and out["flips_1"]["right_index"] == out["flips_1"]["patterns"]
        and out["flips_2"]["in_code_detect_fail"] == 0
        and out["flips_2"]["silent_wrong"] == 0
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["23:32", "25:32", "40:64"])
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--out", default="results/index_code_verify.json")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    report = []
    for cfg in args.configs:
        d, w = map(int, cfg.split(":"))
        if d > 30:  # do not enumerate 2**40 values; sample instead
            code = IndexCode(d, w)
            v = rng.integers(0, 1 << d, 1 << 20, dtype=np.int64)
            idx, _, st, _ = code.decode(code.encode(v))
            r = {"data_bits": d, "word_bits": w, "r": code.r, "code_bits": code.code_bits,
                 "roundtrip_sampled": int(len(v)), "roundtrip_bad": int(((idx != v) | (st != CLEAN)).sum())}
            samples = 200
            vals = rng.integers(0, 1 << d, samples, dtype=np.int64)
            words = code.encode(vals)
            for k in (1, 2):
                fails = 0
                for bits in itertools.combinations(range(w - 1), k):  # keep int64 sign out
                    m = sum(1 << b for b in bits)
                    idx, _, st, _ = code.decode(words ^ np.int64(m))
                    in_code = sum(1 for b in bits if b < code.code_bits)
                    if in_code <= 1:
                        fails += int((idx != vals).sum())
                    else:
                        fails += int((st != DETECTED).sum())
                r[f"flips_{k}_fail"] = fails
            r["ok"] = r["roundtrip_bad"] == 0 and r["flips_1_fail"] == 0 and r["flips_2_fail"] == 0
        else:
            r = run(d, w, args.samples, rng)
        print(json.dumps(r), flush=True)
        report.append(r)
    Path(args.out).write_text(json.dumps(report, indent=2))
    ok = all(r["ok"] for r in report)
    print("INDEX CODE OK:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
