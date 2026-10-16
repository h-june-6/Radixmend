"""Exact triple-error miscorrection rate vs label widths (checkword <= 16 bits).

For a random 3-bit error, the decoder miscorrects when the syndrome happens to
look like a single error. Wider label spaces leave more syndromes unused, so
more triples land on "detected". Prints mean over keys per (e_bits, b_bits).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radixmend.secded_vec import VecCode  # noqa: E402
from scripts.verify_secded_exhaustive import count_triples  # noqa: E402

rows = []
for n_elem in (128, 64):
    for e_extra, b_extra in ((0, 0), (1, 0), (0, 1)):
        rates = []
        for k in range(3):
            base = VecCode(n_elem=n_elem, key=None)
            code = VecCode(n_elem=n_elem, key=f"k{k}".encode(),
                           e_bits_override=base.e_bits + e_extra, b_bits_override=base.b_bits + b_extra)
            t = count_triples(code, code.columns())
            rates.append(t["miscorrection_rate"])
        row = {"n_elem": n_elem, "e_bits": code.e_bits, "b_bits": code.b_bits, "check_bits": code.check_bits,
               "triple_miscorrection_mean": float(np.mean(rates)), "per_key": rates}
        print(json.dumps(row), flush=True)
        rows.append(row)
Path("results/triple_miscorrection_sweep.json").write_text(json.dumps(rows, indent=2))
