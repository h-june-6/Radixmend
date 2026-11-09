"""Summarize bench_tpot.sh output: median over repeats per (config, workload), change vs the first config.

    python scripts/server/tpot_table.py results/rq3/tpot_1109 [baseline_config]
"""

import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
configs = sorted(p.name for p in root.iterdir() if p.is_dir())
base = sys.argv[2] if len(sys.argv) > 2 else ("nosecded" if "nosecded" in configs else configs[0])
order = [base] + [c for c in configs if c != base]
rows = {}
for c in order:
    for f in sorted((root / c).glob("*.jsonl")):
        rs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if rs:
            rows[(f.stem, c)] = rs
print("workload,config,reps,completed,median_ttft_ms,median_tpot_ms,output_tok_s,tpot_vs_base,ttft_vs_base,tpot_rep_spread")
for wl in sorted({w for w, _ in rows}):
    b = rows.get((wl, base))
    for c in order:
        rs = rows.get((wl, c))
        if not rs:
            continue
        med = lambda k, rs=rs: st.median(r[k] for r in rs)
        spread = (max(r["median_tpot_ms"] for r in rs) - min(r["median_tpot_ms"] for r in rs)) / med("median_tpot_ms")
        vs = lambda k: f"{100 * (med(k) / st.median(r[k] for r in b) - 1):+.1f}%" if b else ""
        print(f"{wl},{c},{len(rs)},{rs[0]['completed']},{med('median_ttft_ms'):.1f},{med('median_tpot_ms'):.2f},"
              f"{med('output_throughput'):.1f},{vs('median_tpot_ms')},{vs('median_ttft_ms')},{100 * spread:.1f}%")
