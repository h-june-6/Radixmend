"""RQ3 pre-measurement: standalone cost of module A kernels on the GPU.

The kernels are not fused into attention yet, so this is an upper bound on
the extra work, compared against the same memory traffic without checking:

* decode: check_load (gather + syndrome + correct) vs a plain gather
  (index_select) of the same slots, for one K or V buffer of one layer,
  then scaled to all layers x {K, V} per decode step.
* store: encode of freshly written rows (prefill chunk / decode tokens).

    python scripts/bench_secded_kernels.py --out results/rq3/kernel_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `radixmend`

import torch
import triton.testing as tt

from radixmend.kernels.secded_triton import TritonVecCode
from radixmend.secded_vec import VecCode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--pool", type=int, default=180_040)
    ap.add_argument("--tpot-ms", type=float, nargs="*", default=[24.1, 27.8, 40.9],
                    help="baseline TPOT at concurrency 1/8/32 for the overhead estimate")
    a = ap.parse_args()
    dev = "cuda"
    code = TritonVecCode(VecCode(n_elem=a.head_dim), device=dev)
    buf = torch.randn(a.pool, a.heads, a.head_dim, dtype=torch.bfloat16, device=dev)
    cw = code.encode(buf)
    rows = []
    # decode: (batch, context) -> slots read per layer per buffer
    for batch, ctx in [(1, 1024), (8, 1024), (8, 2048), (32, 1024), (32, 2048)]:
        n = batch * ctx
        loc = torch.randint(0, a.pool, (n,), device=dev, dtype=torch.int64)
        t_check = tt.do_bench(lambda: code.check_load(buf, cw, loc), warmup=25, rep=100)
        t_gather = tt.do_bench(lambda: buf.index_select(0, loc), warmup=25, rep=100)
        per_step_extra = (t_check - t_gather) * a.layers * 2
        rows.append({"op": "decode_check", "batch": batch, "ctx": ctx, "slots": n,
                     "check_ms": t_check, "gather_ms": t_gather, "ratio": t_check / t_gather,
                     "extra_ms_per_step_all_layers": per_step_extra})
    for n in [8, 32, 512, 2048]:
        rows_new = torch.randn(n, a.heads, a.head_dim, dtype=torch.bfloat16, device=dev)
        t_enc = tt.do_bench(lambda: code.encode(rows_new), warmup=25, rep=100)
        rows.append({"op": "encode", "tokens": n, "encode_ms": t_enc,
                     "ms_all_layers_kv": t_enc * a.layers * 2})
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "args": vars(a)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    for r in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    print("note: kernels are standalone (separate launch + own gather); a fused attention kernel "
          "reuses the attention gather, so the real overhead should be below extra_ms_per_step.")


if __name__ == "__main__":
    main()
