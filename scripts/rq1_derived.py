"""RQ1-6 derived contamination: does a child node stay wrong after its parent is repaired?

One trial (no protection):
  1. flush, warm-up request (system + question 0) -> parent node = shared system prompt
  2. flip one bit in the parent's K/V
  3. turn 1: system + question 1 is prefilled on top of the corrupted parent;
     its prompt tokens and answer become a child node
  4. revert the flip: the parent is clean again
  5. compare the child's K/V (question 1 prompt tokens) with a golden snapshot
  6. turn 2: system + question 1 + golden answer + follow-up reuses parent and child
The golden run takes the same path with dry_run. A trial shows derived
contamination when the child K/V differs from golden after step 4, and the
damage persists when turn 2 differs too, even though the parent is clean.
``--kinds none`` runs controls (no flip) that must show zero difference.

    python scripts/rq1_derived.py --out results/rq1/derived --kinds v --bits 14,7 --reps 30
    python scripts/rq1_derived.py --out results/rq1/derived --kinds none --bits 0 --reps 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq1_kv_sweep as kv  # noqa: E402

FOLLOW_UP = "Thanks. Can you summarize that in one sentence?"


def chat_ids(tok, msgs):
    try:
        return list(tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_dict=False))
    except TypeError:
        return list(tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False))


def flow(a, cmd_dir, tok, system, q0, q1, inject, golden=None):
    """Returns (turn1_out, turn2_out, kv_compare or None, flips)."""
    sys_msgs = [{"role": "system", "content": system}]
    p0 = chat_ids(tok, sys_msgs + [{"role": "user", "content": q0}])
    p1 = chat_ids(tok, sys_msgs + [{"role": "user", "content": q1}])
    shared = 0
    while shared < min(len(p0), len(p1)) and p0[shared] == p1[shared]:
        shared += 1
    kv.flush(a.url)
    kv.generate(a.url, p0, a.warm_new)
    inj = kv.hook(cmd_dir, {"op": "inject_kv", **inject, "pos_lo": 0, "pos_hi": shared})
    try:
        out1 = kv.generate(a.url, p1, a.max_new)
    finally:
        kv.hook(cmd_dir, {"op": "revert"})
    child = {"tokens": p1, "pos_lo": shared, "pos_hi": len(p1)}
    if golden is None:
        kv.hook(cmd_dir, {"op": "snapshot_kv", **child, "name": "derived_golden"})
        cmp = None
        answer = tok.decode(out1, skip_special_tokens=True)
    else:
        cmp = kv.hook(cmd_dir, {"op": "compare_kv", **child, "name": "derived_golden"})
        answer = golden["answer"]
    p2 = chat_ids(tok, sys_msgs + [{"role": "user", "content": q1}, {"role": "assistant", "content": answer},
                                   {"role": "user", "content": FOLLOW_UP}])
    out2 = kv.generate(a.url, p2, a.max_new)
    return out1, out2, cmp, inj["flips"], shared, len(p1), answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--cmd-dir", default=f"/tmp/{os.environ.get('USER', 'user')}/rm_cmd")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="v")
    ap.add_argument("--bits", default="14")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warm-new", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    tpath = out / "trials.jsonl"
    if a.summarize:
        rows = [json.loads(l) for l in tpath.read_text().splitlines() if l.strip()]
        keys = sorted({(r["kind"], r["bit"]) for r in rows})
        print("kind,bit,n,child_kv_differs,turn1_changed,turn2_changed,turn2_changed_given_kv_differs,median_max_abs")
        for k, b in keys:
            rs = [r for r in rows if r["kind"] == k and r["bit"] == b and not r.get("error")]
            kvd = [r for r in rs if r["child_kv_differs"]]
            mx = sorted(r["kv"]["max_abs"] for r in rs)
            print(f"{k},{b},{len(rs)},{len(kvd)},{sum(r['turn1_changed'] for r in rs)},"
                  f"{sum(r['turn2_changed'] for r in rs)},{sum(r['turn2_changed'] for r in kvd)},"
                  f"{mx[len(mx)//2] if mx else 0:.3g}")
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd_dir = Path(a.cmd_dir)
    cmd_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    system = (kv.DATA / "w1_system_prompt.txt").read_text().strip()
    qs = [q for q in (kv.DATA / "w1_questions.txt").read_text().splitlines() if q.strip()]
    q0, q1 = qs[0], qs[1]

    dry = {"seed": 0, "kinds": ["v"], "bits": [0], "dry_run": True}
    g = flow(a, cmd_dir, tok, system, q0, q1, dry)
    golden = {"out1": g[0], "out2": g[1], "answer": g[6]}
    # second golden must match exactly, KV included
    g2 = flow(a, cmd_dir, tok, system, q0, q1, dry, golden)
    if g2[0] != g[0] or g2[1] != g[1] or g2[2]["n_diff"]:
        print("golden not reproducible:", g2[2])
        sys.exit(2)
    print(f"golden ok: shared {g[4]}, child prompt tokens {g[5] - g[4]}", flush=True)

    done = set()
    if tpath.exists():
        done = {json.loads(l)["trial"] for l in tpath.read_text().splitlines() if l.strip()}
    todo = [(k, b, r) for k in a.kinds.split(",") for b in kv.parse_bits(a.bits) for r in range(a.reps)]
    todo.sort(key=lambda t: (t[2], t[0], t[1]))
    for kind, bit, rep in todo:
        tid = f"derived-{kind}-b{bit:02d}-r{rep:03d}"
        if tid in done:
            continue
        seed = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16)
        inject = {"seed": seed, "kinds": ["v" if kind == "none" else kind], "bits": [bit], "dry_run": kind == "none"}
        rec = {"trial": tid, "kind": kind, "bit": bit, "rep": rep}
        t0 = time.monotonic()
        try:
            o1, o2, cmp, flips, *_ = flow(a, cmd_dir, tok, system, q0, q1, inject, golden)
            rec.update(flips=flips, kv=cmp, child_kv_differs=cmp["n_diff"] > 0,
                       turn1_changed=o1 != golden["out1"], turn2_changed=o2 != golden["out2"],
                       turn1_first_diff=kv.first_diff(o1, golden["out1"]),
                       turn2_first_diff=kv.first_diff(o2, golden["out2"]))
        except Exception as e:
            rec["error"] = repr(e)
            print(f"{tid}: ERROR {e!r}", flush=True)
        rec["seconds"] = round(time.monotonic() - t0, 2)
        with tpath.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if "error" not in rec:
            print(f"{tid}: child_kv_differs={rec['child_kv_differs']} max_abs={cmp['max_abs']:.3g} "
                  f"turn1={rec['turn1_changed']} turn2={rec['turn2_changed']}", flush=True)


if __name__ == "__main__":
    main()
