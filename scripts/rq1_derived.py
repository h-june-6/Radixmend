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

``--when decode`` moves the flip to after the prefix hit: it is injected while
turn 1 is decoding (``--after-tokens``), so a hit-time hash (server started
with RADIXMEND_HASH=1) has already passed the parent. The child range then
also covers turn 1's generated tokens, whose KV was computed from the flipped
parent. Hash and SECDED counters are recorded per trial when the server has
them.

    python scripts/rq1_derived.py --out results/rq1/derived --kinds v --bits 14,7 --reps 30
    python scripts/rq1_derived.py --out results/rq1/derived --kinds none --bits 0 --reps 10
    # hash baseline (server with RADIXMEND_HASH=1): flip before the hit, then after it
    python scripts/rq1_derived.py --out results/rq1/derived_hash --kinds v --bits 14,7 --reps 30
    python scripts/rq1_derived.py --out results/rq1/derived_hash_toctou --when decode --kinds v --bits 14,7 --reps 30
    # module C (server with RADIXMEND_SECDED=1 RADIXMEND_RECOVER=1 --disable-overlap-schedule):
    # 2-bit error in one vector of the parent, which module A detects but cannot correct
    python scripts/rq1_derived.py --out results/c/derived_2bit --kinds v --bits-each 14:7,7:3 --reps 30 --turn2 own
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

STAT_OPS = ("hash_stats", "secded_stats", "recover_stats", "replica_stats")
FOLLOW_UP = "Thanks. Can you summarize that in one sentence?"


def chat_ids(tok, msgs):
    try:
        return list(tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_dict=False))
    except TypeError:
        return list(tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False))


def flow(a, cmd_dir, tok, system, q0, q1, inject, golden=None):
    """Returns (turn1_out, turn2_out, kv_compare or None, flips, shared, len(p1), answer, stats)."""
    sys_msgs = [{"role": "system", "content": system}]
    p0 = chat_ids(tok, sys_msgs + [{"role": "user", "content": q0}])
    p1 = chat_ids(tok, sys_msgs + [{"role": "user", "content": q1}])
    shared = 0
    while shared < min(len(p0), len(p1)) and p0[shared] == p1[shared]:
        shared += 1
    kv.flush(a.url)
    kv.generate(a.url, p0, a.warm_new)
    stats = {op: kv.hook(cmd_dir, {"op": op, "reset": True}) for op in STAT_OPS}
    op = {"op": "inject_kv", **inject, "pos_lo": 0, "pos_hi": shared}
    try:
        if a.when == "decode":
            op["after_tokens"] = a.after_tokens
            cid = send_cmd(cmd_dir, op)
            t1 = time.monotonic()
            out1 = kv.generate(a.url, p1, a.max_new)
            t1 = time.monotonic() - t1
            inj = wait_done(cmd_dir, cid, 5.0)
            if inj is None:
                kv.hook(cmd_dir, {"op": "cancel", "target_id": cid})
                inj = {"flips": [], "not_triggered": True}
            elif not inj.get("ok", True):
                raise RuntimeError(inj.get("error"))
        else:
            inj = kv.hook(cmd_dir, op)
            t1 = time.monotonic()
            out1 = kv.generate(a.url, p1, a.max_new)
            t1 = time.monotonic() - t1
    finally:
        kv.hook(cmd_dir, {"op": "revert"})
    stats = {op: kv.hook(cmd_dir, {"op": op}) for op in STAT_OPS}
    # child = turn 1's own tokens; with a decode-time flip also its generated tokens
    # (the last output token has no KV yet)
    gen = out1 if a.when == "decode" else []
    child = {"tokens": p1 + gen, "pos_lo": shared, "pos_hi": len(p1) + max(len(gen) - 1, 0)}
    if golden is None:
        kv.hook(cmd_dir, {"op": "snapshot_kv", **child, "name": "derived_golden"})
        cmp = None
        answer = tok.decode(out1, skip_special_tokens=True)
    else:
        if out1 == golden["out1"]:
            cmp = kv.hook(cmd_dir, {"op": "compare_kv", **child, "name": "derived_golden"})
        else:  # turn 1 tokens diverged: the child is a different sequence
            cmp = {"diverged": True, "n_diff": -1, "max_abs": float("nan")}
        answer = golden["answer"]
    if a.turn2 == "own":
        # p1 + turn 1 tokens + the template tail for the next user turn: same tokens as the
        # cached child, so turn 2 reuses the child's output KV (derived from the flipped parent)
        tail = tok.encode("<|im_end|>\n<|im_start|>user\n" + FOLLOW_UP + "<|im_end|>\n<|im_start|>assistant\n"
                          "<think>\n\n</think>\n\n", add_special_tokens=False)
        body = out1[:-1] if out1 and out1[-1] == tok.convert_tokens_to_ids("<|im_end|>") else out1
        p2 = p1 + body + tail
    else:
        p2 = chat_ids(tok, sys_msgs + [{"role": "user", "content": q1}, {"role": "assistant", "content": answer},
                                       {"role": "user", "content": FOLLOW_UP}])
    out2 = kv.generate(a.url, p2, a.max_new)
    stats2 = {op: kv.hook(cmd_dir, {"op": op}) for op in STAT_OPS}
    return out1, out2, cmp, inj.get("flips", []), shared, len(p1), answer, {"turn1": stats, "after_turn2": stats2,
                                                                      "turn1_seconds": round(t1, 4)}


def send_cmd(cmd_dir: Path, op: dict) -> str:
    cid = f"{time.time_ns()}"
    tmp = cmd_dir / f"{cid}.tmp"
    tmp.write_text(json.dumps(op))
    os.replace(tmp, cmd_dir / f"{cid}.cmd.json")
    return cid


def wait_done(cmd_dir: Path, cid: str, timeout: float):
    done = cmd_dir / f"{cid}.done.json"
    t0 = time.monotonic()
    while not done.exists():
        if time.monotonic() - t0 > timeout:
            return None
        time.sleep(0.005)
    res = json.loads(done.read_text())
    done.unlink()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--cmd-dir", default=f"/tmp/{os.environ.get('USER', 'user')}/rm_cmd")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="v")
    ap.add_argument("--bits", default="14")
    ap.add_argument("--bits-each", default="",
                    help='comma-separated "a:b" pairs: flip bit a of one element and bit b of another in '
                         'the same vector (a 2-bit error that module A detects but cannot correct)')
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warm-new", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--when", choices=["hit", "decode"], default="hit",
                    help="flip before turn 1 (hit) or while turn 1 decodes, after the hit check (decode)")
    ap.add_argument("--after-tokens", type=int, default=4)
    ap.add_argument("--turn2", choices=["golden", "own"], default="golden",
                    help="turn 2 repeats turn 1 as the golden answer text (golden) or reuses turn 1's own "
                         "generated tokens, so it hits the child's cached output KV (own)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    tpath = out / "trials.jsonl"
    if a.summarize:
        rows = [json.loads(l) for l in tpath.read_text().splitlines() if l.strip()]
        keys = sorted({(r["kind"], r["bit"]) for r in rows})
        print("kind,bit,n,child_kv_differs(compared),turn1_diverged(no_compare),turn1_changed,turn2_changed,"
              "turn2_changed_given_kv_differs,turn2_changed_given_turn1_same,median_max_abs")
        for k, b in keys:
            rs = [r for r in rows if r["kind"] == k and r["bit"] == b and not r.get("error")]
            div = [r for r in rs if r["kv"].get("diverged")]
            kvd = [r for r in rs if r["child_kv_differs"] and not r["kv"].get("diverged")]
            same1 = [r for r in rs if not r["turn1_changed"]]
            mx = sorted(r["kv"]["max_abs"] for r in rs if r["kv"]["max_abs"] == r["kv"]["max_abs"])
            print(f"{k},{b},{len(rs)},{len(kvd)},{len(div)},{sum(r['turn1_changed'] for r in rs)},"
                  f"{sum(r['turn2_changed'] for r in rs)},{sum(r['turn2_changed'] for r in kvd)},"
                  f"{sum(r['turn2_changed'] for r in same1)}/{len(same1)},"
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
    print(f"golden ok: shared {g[4]}, child prompt tokens {g[5] - g[4]}, when={a.when}, "
          f"hash={g[7]['turn1']['hash_stats'].get('enabled')}, secded={g[7]['turn1']['secded_stats'].get('enabled')}",
          flush=True)

    done = set()
    if tpath.exists():
        done = {json.loads(l)["trial"] for l in tpath.read_text().splitlines() if l.strip()}
    groups = [p.strip() for p in a.bits_each.split(",") if p.strip()] if a.bits_each else kv.parse_bits(a.bits)
    todo = [(k, b, r) for k in a.kinds.split(",") for b in groups for r in range(a.reps)]
    todo.sort(key=lambda t: (t[2], t[0], t[1]))
    for kind, bit, rep in todo:
        if isinstance(bit, str):
            pa, pb = (int(x) for x in bit.split(":"))
            tid = f"derived-{kind}-each{pa:02d}x{pb:02d}-r{rep:03d}"
        else:
            tid = f"derived-{kind}-b{bit:02d}-r{rep:03d}"
        if tid in done:
            continue
        seed = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16)
        inject = {"seed": seed, "kinds": ["v" if kind == "none" else kind], "dry_run": kind == "none"}
        if isinstance(bit, str):
            inject["bits_each"] = [pa, pb]
        else:
            inject["bits"] = [bit]
        rec = {"trial": tid, "kind": kind, "bit": bit, "rep": rep}
        t0 = time.monotonic()
        try:
            o1, o2, cmp, flips, _, _, _, st = flow(a, cmd_dir, tok, system, q0, q1, inject, golden)
            rec.update(flips=flips, kv=cmp, child_kv_differs=cmp["n_diff"] != 0, when=a.when, stats=st,
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
