"""Summarize a w14 final run: per (workload, config) medians over repeats of median/p99 TPOT and TTFT,
output throughput, change vs the baseline config, and peak GPU memory (gpu_mem.txt per config).

    python scripts/server/final_table.py results/w14/final [baseline_config]
"""

import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
configs = sorted(p.name for p in root.iterdir() if p.is_dir())
base = sys.argv[2] if len(sys.argv) > 2 else configs[0]
order = [base] + [c for c in configs if c != base]
rows = {}
for c in order:
    for f in sorted((root / c).glob("*.jsonl")):
        rs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if rs:
            rows[(f.stem, c)] = rs


def mem(c):
    p = root / c / "gpu_mem.txt"
    return p.read_text().strip() if p.exists() else ""


keys = ["median_tpot_ms", "p99_tpot_ms", "median_ttft_ms", "p99_ttft_ms", "output_throughput"]
print("workload,config,reps,completed," + ",".join(keys) + ",tpot_vs_base,p99_tpot_vs_base,ttft_vs_base,"
      "p99_ttft_vs_base,tput_vs_base,tpot_rep_spread,gpu_mem_mib(ready/peak)")
for wl in sorted({w for w, _ in rows}):
    b = rows.get((wl, base))
    for c in order:
        rs = rows.get((wl, c))
        if not rs:
            continue
        med = {k: st.median(r[k] for r in rs) for k in keys}
        bm = {k: st.median(r[k] for r in b) for k in keys} if b else None
        vs = lambda k: f"{100 * (med[k] / bm[k] - 1):+.1f}%" if bm else ""
        spread = (max(r["median_tpot_ms"] for r in rs) - min(r["median_tpot_ms"] for r in rs)) / med["median_tpot_ms"]
        print(f"{wl},{c},{len(rs)},{rs[0]['completed']}," + ",".join(f"{med[k]:.2f}" for k in keys) + ","
              + ",".join(vs(k) for k in ["median_tpot_ms", "p99_tpot_ms", "median_ttft_ms", "p99_ttft_ms",
                                         "output_throughput"])
              + f",{100 * spread:.1f}%,{mem(c)}")
