"""RQ3 kernel-level bench: SGLang decode stage 1 vs the module A fused variants.

Same shapes as Qwen3-8B on one layer (32 q heads, 8 kv heads, head_dim 128,
bf16, pool 180,040 slots), random slot indices. Times one stage-1 launch per
variant with triton.testing.do_bench and scales it to 36 layers per decode
step. No SGLang server needed, but SGLang must be importable (baseline kernel).

    python scripts/bench_fused_decode.py --out results/rq3/fused_kernel.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton.testing as tt

from radixmend import sglang_secded as ss

VARIANTS = {
    "copy_nocheck": dict(check=False),
    "full": dict(mode="full", popc=False),
    "full_popc": dict(mode="full", popc=True),
    "gate": dict(mode="gate", popc=False),
    "gate_popc": dict(mode="gate", popc=True),
    "full_popc_s4": dict(mode="full", popc=True, sample=4),
    "gate_popc_s4": dict(mode="gate", popc=True, sample=4),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--pool", type=int, default=180_040)
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--shapes", default="1x1024,8x1024,8x2304,32x650,32x2304",
                    help="batch x context length")
    a = ap.parse_args()
    from sglang.kernels.ops.attention import decode_attention as da

    dev = "cuda"
    D = a.head_dim
    k = torch.randn(a.pool, a.kv_heads, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(a.pool, a.kv_heads, D, dtype=torch.bfloat16, device=dev)
    ss._tables(k.device)
    kcw = torch.zeros(a.pool, a.kv_heads, dtype=torch.int16, device=dev)
    vcw = torch.zeros_like(kcw)
    ss.encode_rows(k, kcw, torch.arange(a.pool, device=dev))
    ss.encode_rows(v, vcw, torch.arange(a.pool, device=dev))
    rows = []
    for shape in a.shapes.split(","):
        batch, ctx = map(int, shape.split("x"))
        q = torch.randn(batch, a.q_heads, D, dtype=torch.bfloat16, device=dev)
        idx = torch.randint(0, a.pool, (batch * ctx,), device=dev, dtype=torch.int64)
        indptr = torch.arange(0, batch * ctx + 1, ctx, device=dev, dtype=torch.int32)
        nsp = torch.full((batch,), a.splits, dtype=torch.int32, device=dev)
        out = torch.empty(batch, a.q_heads, a.splits, D, dtype=torch.float32, device=dev)
        lse = torch.empty(batch, a.q_heads, a.splits, dtype=torch.float32, device=dev)
        sm = D ** -0.5

        def base():
            da._decode_grouped_att_m_fwd(q, k, v, out, lse, indptr, idx, nsp, a.splits, sm, 0.0)

        base()
        ref = out.clone()
        t_base = tt.do_bench(base, warmup=25, rep=200)
        row = {"batch": batch, "ctx": ctx, "sglang_ms": t_base}
        for name, kw in VARIANTS.items():
            kw = dict(kw)
            check = kw.pop("check", True)

            def fn():
                ss.launch_stage1(q, k, v, kcw, vcw, out, lse, indptr, idx, nsp, a.splits, sm, 0.0, check,
                                 block_h=da._GROUPED_BLOCK_H, min_block_kv=da._MIN_BLOCK_KV, **kw)

            fn()
            same = bool(torch.equal(out, ref))  # no faults: must match SGLang bitwise
            t = tt.do_bench(fn, warmup=25, rep=200)
            row[name] = {"ms": t, "ratio": t / t_base, "extra_ms_per_step": (t - t_base) * a.layers,
                         "bitwise_equal": same}
        rows.append(row)
        print(json.dumps({"batch": batch, "ctx": ctx, "sglang_ms": round(t_base, 4),
                          **{n: round(row[n]["ratio"], 3) for n in VARIANTS},
                          "all_equal": all(row[n]["bitwise_equal"] for n in VARIANTS)}), flush=True)
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "args": vars(a)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print("ratio = variant time / SGLang stage-1 time (one layer). extra_ms_per_step in the json is"
          " scaled to all layers; compare it with TPOT (27.6 ms at c8, 34.6 ms at c32).")


if __name__ == "__main__":
    main()
