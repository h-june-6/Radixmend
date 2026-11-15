"""RQ4: blast radius of one uncorrectable (2-bit) error in a shared prefix under concurrency.

N requests that share the system prompt are in flight together. One 2-bit error
(module A detects, cannot correct) is put into one vector of the shared prefix,
either before the requests arrive (``--when hit``) or once they are decoding
(``--when decode``). Per trial it records which requests' outputs changed versus
a fault-free golden run of the same N requests, end-to-end latency, and the
module A / module C counters (how many requests module C retracted and recomputed,
how many tokens it wrote back).

Run against a server in deterministic mode (outputs must not depend on batching)
with ``--disable-overlap-schedule``:
- module A + C: RADIXMEND_SECDED=1 RADIXMEND_RECOVER=1 -> blast radius = recomputed requests;
- module A only: RADIXMEND_SECDED=1 -> blast radius = requests whose output changed.

    python scripts/rq4_blast.py --out results/rq4/c_n8 --n 8 --when decode --reps 20
    python scripts/rq4_blast.py --out results/rq4/c_n8 --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq1_derived as dv  # noqa: E402
import rq1_kv_sweep as kv  # noqa: E402

STAT_OPS = ("secded_stats", "recover_stats")


def build(tok, n: int):
    system = (kv.DATA / "w1_system_prompt.txt").read_text().strip()
    qs = [q for q in (kv.DATA / "w1_questions.txt").read_text().splitlines() if q.strip()]
    sys_msgs = [{"role": "system", "content": system}]
    warm = dv.chat_ids(tok, sys_msgs + [{"role": "user", "content": qs[0]}])
    prompts = []
    for i in range(n):
        q = qs[1 + i % (len(qs) - 1)]
        if i >= len(qs) - 1:
            q = f"{q} (request {i})"
        prompts.append(dv.chat_ids(tok, sys_msgs + [{"role": "user", "content": q}]))
    shared = min(len(warm), *(len(p) for p in prompts))
    for p in prompts:
        k = 0
        while k < shared and p[k] == warm[k]:
            k += 1
        shared = k
    return warm, prompts, shared


def burst(url: str, prompts, max_new: int):
    """All prompts at once; returns (outputs, per-request seconds, wall seconds)."""
    def one(p):
        t = time.monotonic()
        o = kv.generate(url, p, max_new)
        return o, time.monotonic() - t

    t0 = time.monotonic()
    with ThreadPoolExecutor(len(prompts)) as ex:
        res = list(ex.map(one, prompts))
    return [r[0] for r in res], [r[1] for r in res], time.monotonic() - t0


def trial(a, cmd_dir: Path, warm, prompts, shared, inject):
    kv.flush(a.url)
    kv.generate(a.url, warm, a.warm_new)
    for op in STAT_OPS:
        kv.hook(cmd_dir, {"op": op, "reset": True})
    op = {"op": "inject_kv", **inject, "pos_lo": 0, "pos_hi": shared}
    try:
        if a.when == "decode":
            op["after_tokens"] = a.after_tokens
            cid = dv.send_cmd(cmd_dir, op)
            outs, secs, wall = burst(a.url, prompts, a.max_new)
            inj = dv.wait_done(cmd_dir, cid, 5.0)
            if inj is None:
                kv.hook(cmd_dir, {"op": "cancel", "target_id": cid})
                inj = {"flips": [], "not_triggered": True}
            elif not inj.get("ok", True):
                raise RuntimeError(inj.get("error"))
        else:
            inj = kv.hook(cmd_dir, op)
            outs, secs, wall = burst(a.url, prompts, a.max_new)
    finally:
        kv.hook(cmd_dir, {"op": "revert"})
    stats = {op: kv.hook(cmd_dir, {"op": op}) for op in STAT_OPS}
    return outs, secs, wall, inj, stats


def summarize(tpath: Path) -> None:
    rows = [json.loads(l) for l in tpath.read_text().splitlines() if l.strip()]
    gold = [r for r in rows if r.get("golden")]
    rs = [r for r in rows if not r.get("golden") and not r.get("error")]
    print("kind,bits,n,trials,not_triggered,detected,reqs_changed(mean/max),reqs_retracted(mean/max),"
          "tokens_repaired(mean),wall_s(median),golden_wall_s(median),req_s(median),golden_req_s(median)")
    gw = statistics.median(g["wall"] for g in gold) if gold else float("nan")
    gq = statistics.median(s for g in gold for s in g["secs"]) if gold else float("nan")
    for key in sorted({(r["kind"], r["bits"], r["n"]) for r in rs}):
        g = [r for r in rs if (r["kind"], r["bits"], r["n"]) == key]
        ch = [r["n_changed"] for r in g]
        rt = [r["stats"]["recover_stats"].get("reqs_retracted", 0) for r in g]
        tr = [r["stats"]["recover_stats"].get("tokens_repaired", 0) for r in g]
        det = sum(1 for r in g if any(r["stats"]["secded_stats"].get(f"{p}_{k}_uncorrectable", 0)
                                      for p in ("decode", "extend") for k in ("k", "v")))
        print(f"{key[0]},{key[1]},{key[2]},{len(g)},{sum(1 for r in g if r['inject'].get('not_triggered'))},"
              f"{det},{statistics.mean(ch):.2f}/{max(ch)},{statistics.mean(rt):.2f}/{max(rt)},"
              f"{statistics.mean(tr):.0f},{statistics.median(r['wall'] for r in g):.3f},{gw:.3f},"
              f"{statistics.median(s for r in g for s in r['secs']):.3f},{gq:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--cmd-dir", default=f"/tmp/{os.environ.get('USER', 'user')}/rm_cmd")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="concurrent requests sharing the system prompt")
    ap.add_argument("--kinds", default="v")
    ap.add_argument("--bits-each", default="14:7,7:3")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--golden-reps", type=int, default=3)
    ap.add_argument("--warm-new", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--when", choices=["hit", "decode"], default="decode")
    ap.add_argument("--after-tokens", type=int, default=4)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    tpath = out / "trials.jsonl"
    if a.summarize:
        summarize(tpath)
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd_dir = Path(a.cmd_dir)
    cmd_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    warm, prompts, shared = build(tok, a.n)
    dry = {"seed": 0, "kinds": ["v"], "bits_each": [14, 7], "dry_run": True}

    golden = None
    for g in range(a.golden_reps):
        outs, secs, wall, _, _ = trial(a, cmd_dir, warm, prompts, shared, dry)
        if golden is None:
            golden = outs
        elif outs != golden:
            bad = [i for i, (x, y) in enumerate(zip(outs, golden)) if x != y]
            print(f"golden not reproducible under concurrency (requests {bad}); "
                  "is the server in deterministic mode?")
            sys.exit(2)
        with tpath.open("a") as fh:
            fh.write(json.dumps({"trial": f"golden-n{a.n}-{a.when}-g{g}", "golden": True, "n": a.n,
                                 "secs": [round(s, 4) for s in secs], "wall": round(wall, 4)}) + "\n")
    print(f"golden ok: n={a.n}, shared {shared} tokens, when={a.when}", flush=True)

    done = {json.loads(l)["trial"] for l in tpath.read_text().splitlines() if l.strip()}
    pairs = [p.strip() for p in a.bits_each.split(",") if p.strip()]
    todo = sorted(((k, b, r) for k in a.kinds.split(",") for b in pairs for r in range(a.reps)),
                  key=lambda t: (t[2], t[0], t[1]))
    for kind, bits, rep in todo:
        pa, pb = (int(x) for x in bits.split(":"))
        tid = f"blast-n{a.n}-{a.when}-{kind}-each{pa:02d}x{pb:02d}-r{rep:03d}"
        if tid in done:
            continue
        seed = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16)
        inject = {"seed": seed, "kinds": ["v" if kind == "none" else kind], "bits_each": [pa, pb],
                  "dry_run": kind == "none"}
        rec = {"trial": tid, "kind": kind, "bits": bits, "rep": rep, "n": a.n, "when": a.when}
        try:
            outs, secs, wall, inj, stats = trial(a, cmd_dir, warm, prompts, shared, inject)
            changed = [i for i, (x, y) in enumerate(zip(outs, golden)) if x != y]
            rec.update(n_changed=len(changed), changed=changed, secs=[round(s, 4) for s in secs],
                       wall=round(wall, 4), inject={k: v for k, v in inj.items() if k != "flips"},
                       flips=inj.get("flips", []), stats=stats)
        except Exception as e:
            rec["error"] = repr(e)
            print(f"{tid}: ERROR {e!r}", flush=True)
        with tpath.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if "error" not in rec:
            rc = stats["recover_stats"]
            print(f"{tid}: changed={rec['n_changed']}/{a.n} retracted={rc.get('reqs_retracted', 0)} "
                  f"repaired={rc.get('tokens_repaired', 0)} wall={wall:.2f}s", flush=True)


if __name__ == "__main__":
    main()
