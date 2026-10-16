"""Codeword length trade-off: k random flips inside one 256-byte head vector.

The head vector (128 bf16 + checkwords) is split into 1, 2 or 4 codewords.
Flips are uniform over all stored bits of the vector (data and checkwords).
Per trial the vector outcome is:
  corrected  - every codeword clean or corrected
  detected   - at least one codeword detected, none miscorrected
  silent     - at least one codeword miscorrected (wrong value passes as good)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radixmend.secded_vec import CLEAN, CORRECTED_CHECK, CORRECTED_DATA, DETECTED, VecCode  # noqa: E402
from radixmend.labels import bits_needed  # noqa: E402

rng = np.random.default_rng(42)
TRIALS = 50_000
rows = []
for splits in (1, 2, 4):
    n_elem = 128 // splits
    for widen in (False, True):
        e = bits_needed(n_elem) + (1 if widen else 0)
        code = VecCode(n_elem=n_elem, key=b"mc", e_bits_override=e)
        cols = code.columns()
        nb = code.codeword_bits
        total_bits = nb * splits
        for k in (1, 2, 3, 4):
            pos = rng.integers(0, total_bits, (TRIALS, k))
            while k > 1:  # resample rows that drew the same bit twice
                dup = (np.diff(np.sort(pos, 1), axis=1) == 0).any(1)
                if not dup.any():
                    break
                pos[dup] = rng.integers(0, total_bits, (int(dup.sum()), k))
            cw_id, local = pos // nb, pos % nb
            syn = np.zeros((TRIALS, splits), np.int64)
            for j in range(k):
                np.bitwise_xor.at(syn, (np.arange(TRIALS), cw_id[:, j]), cols[local[:, j]])
            st = code.lut_status[syn]
            # a corrected codeword is right only if it really had exactly one flip
            nflips = np.zeros((TRIALS, splits), np.int64)
            for j in range(k):
                np.add.at(nflips, (np.arange(TRIALS), cw_id[:, j]), 1)
            corrected = (st == CORRECTED_DATA) | (st == CORRECTED_CHECK)
            mis = (corrected & (nflips != 1)) | ((st == CLEAN) & (nflips > 0))
            silent = mis.any(1)
            detected = (~silent) & (st == DETECTED).any(1)
            ok = ~silent & ~detected
            row = {"splits": splits, "e_bits": e, "check_bits": code.check_bits,
                   "mem_overhead": splits * 2 / 256, "flips": k,
                   "corrected": float(ok.mean()), "detected": float(detected.mean()), "silent": float(silent.mean())}
            rows.append(row)
            print(json.dumps(row), flush=True)
Path("results/codeword_length_mc.json").write_text(json.dumps(rows, indent=2))
