"""Pre-extend check (module A check_prefix) time vs RADIXMEND_SECDED_EXT_BLOCK, clean data.

Qwen3-8B shapes: 36 layers, K and V per layer, 8 KV heads x 128 bf16. One prefill that reuses a
cached prefix of --prefix tokens (--batch such prefills at once). Time is the sum over 36 layers
of check_fix_inplace(K) + check_fix_inplace(V), measured with CUDA events, median of --iters.

    python scripts/bench_ext_check.py --blocks 0,8,16,32 --prefix 4096 --out results/w15/extblock/kernel.json
"""

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from radixmend import sglang_secded as ss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0,8,16,32")
    ap.add_argument("--prefix", type=int, default=4096)
    ap.add_argument("--batch", default="1,8")
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda"
    ss._tables(dev)
    batches = [int(x) for x in a.batch.split(",")]
    pool = a.prefix * max(batches) + 1024
    g = torch.Generator(device=dev).manual_seed(0)
    bufs, cws = [], []
    for _ in range(a.layers * 2):
        b = torch.randn(pool, a.heads, 128, generator=g, device=dev).to(torch.bfloat16)
        c = torch.zeros(pool, a.heads, dtype=torch.int16, device=dev)
        ss.encode_rows(b, c, torch.arange(pool, device=dev))
        bufs.append(b)
        cws.append(c)
    res = []
    for nb in batches:
        perm = torch.randperm(pool - 1, device=dev)[: a.prefix * nb].to(torch.int64)
        indptr = torch.arange(0, a.prefix * nb + 1, a.prefix, dtype=torch.int32, device=dev)
        for blk in [int(x) for x in a.blocks.split(",")]:
            def once():
                for i in range(a.layers * 2):
                    ss.check_fix_inplace(bufs[i], cws[i], perm, indptr, 4 if i % 2 == 0 else 6, block=blk)
            once()
            torch.cuda.synchronize()
            ts = []
            for _ in range(a.iters):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                once()
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            errs = ss.errors()
            r = {"batch": nb, "prefix": a.prefix, "block": blk, "median_ms": round(st.median(ts), 3),
                 "min_ms": round(min(ts), 3), "errors": errs}
            print(json.dumps(r), flush=True)
            res.append(r)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
