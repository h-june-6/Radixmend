"""RQ1 KV bit-flip sweep against a running SGLang server (RQ1-2 / RQ1-3 / RQ1-4).

One trial:
  1. flush the radix cache
  2. warm-up request (system prompt + question 0) puts the shared prefix in the tree
  3. flip bit(s) in K/V of one slot inside the shared prefix (via radixmend.sglang_hook)
  4. follow-up requests (system prompt + question i) reuse the corrupted prefix
  5. revert the flips
The follow-up outputs are compared token by token with a golden run that goes
through the identical flow with ``dry_run`` (no flip), so the cache state of
golden and faulty runs is the same (see experiment_spec 7장 5번).

Results are appended to <out>/trials.jsonl, one line per trial; a rerun skips
trials already there, so a sweep resumes across 3-day server rentals.

Server (separate tmux window):
    PYTHONPATH=~/radixmend:~/radixmend/radixmend/sglang_site RADIXMEND_CMD_DIR=/tmp/$USER/rm_cmd \\
    python -m sglang.launch_server --model-path Qwen/Qwen3-8B --attention-backend triton \\
        --page-size 1 --port 30000 --mem-fraction-static 0.85 --kv-canary none
Driver:
    python scripts/rq1_kv_sweep.py --out results/rq1/v_bits --kinds v --bits 0-15 --reps 50
    python scripts/rq1_kv_sweep.py --out results/rq1/v_bits --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


# ---- server I/O -------------------------------------------------------------------------


def http(url: str, payload=None, timeout: float = 600.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def flush(base: str) -> None:
    for _ in range(100):
        try:
            http(base + "/flush_cache", {})
            return
        except urllib.error.HTTPError:
            time.sleep(0.1)  # not idle yet
    raise RuntimeError("flush_cache kept failing")


def generate(base: str, ids: list[int], max_new: int) -> list[int]:
    out = http(base + "/generate", {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new, "ignore_eos": False},
    })
    if "output_ids" in out:
        return out["output_ids"]
    raise RuntimeError(f"no output_ids in response keys {list(out)}")


def hook(cmd_dir: Path, op: dict, timeout: float = 60.0) -> dict:
    cid = f"{time.time_ns()}"
    tmp = cmd_dir / f"{cid}.tmp"
    tmp.write_text(json.dumps(op))
    os.replace(tmp, cmd_dir / f"{cid}.cmd.json")
    done = cmd_dir / f"{cid}.done.json"
    t0 = time.monotonic()
    while not done.exists():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"hook did not answer {op['op']} (is RADIXMEND_CMD_DIR set on the server?)")
        time.sleep(0.005)
    res = json.loads(done.read_text())
    done.unlink()
    if not res.get("ok"):
        raise RuntimeError(f"hook {op['op']} failed: {res.get('error')}\n{res.get('trace')}")
    return res


# ---- workload ---------------------------------------------------------------------------


def build_prompts(tokenizer_name: str, n_followups: int):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    system = (DATA / "w1_system_prompt.txt").read_text().strip()
    questions = [q for q in (DATA / "w1_questions.txt").read_text().splitlines() if q.strip()]
    prompts = []
    for q in questions[: n_followups + 1]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        prompts.append(list(ids))
    shared = 0
    while all(len(p) > shared and p[shared] == prompts[0][shared] for p in prompts):
        shared += 1
    return prompts, shared


def positions(spec: str, shared: int) -> tuple[int, int]:
    """'prefix' = whole shared prefix, 'sink' = first 4 tokens, 'body' = rest, 'a:b' = explicit."""
    if spec == "prefix":
        return 0, shared
    if spec == "sink":
        return 0, 4
    if spec == "body":
        return 4, shared
    a, b = spec.split(":")
    return int(a), int(b)


def run_flow(base, cmd_dir, prompts, inject: dict, warm_new: int, max_new: int):
    t0 = time.monotonic()
    flush(base)
    generate(base, prompts[0], warm_new)
    info = hook(cmd_dir, {"op": "tree_info"})
    inj = hook(cmd_dir, {"op": "inject_kv", **inject})
    try:
        outs = [generate(base, p, max_new) for p in prompts[1:]]
    finally:
        hook(cmd_dir, {"op": "revert"})
    return outs, inj, info, time.monotonic() - t0


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def parse_bits(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# ---- summary ----------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(out: Path) -> None:
    rows = [json.loads(l) for l in (out / "trials.jsonl").read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        key = (r["kind"], r["pos"], r["bit"])
        by.setdefault(key, []).append(r)
    lines = ["kind,pos,bit,n,any_changed,rate,ci_lo,ci_hi,mean_changed_followups,errors"]
    for (kind, pos, bit), rs in sorted(by.items()):
        ok = [r for r in rs if not r.get("error")]
        k = sum(r["any_changed"] for r in ok)
        lo, hi = wilson(k, len(ok))
        mean = sum(r["n_changed"] for r in ok) / len(ok) if ok else 0
        lines.append(f"{kind},{pos},{bit},{len(ok)},{k},{k/len(ok) if ok else 0:.3f},{lo:.3f},{hi:.3f},{mean:.2f},{len(rs)-len(ok)}")
    (out / "summary.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- main -------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--cmd-dir", default=f"/tmp/{os.environ.get('USER', 'user')}/rm_cmd")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="v", help="v, k or k,v")
    ap.add_argument("--bits", default="0-15")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--pos", default="prefix", help="prefix | sink | body | a:b")
    ap.add_argument("--followups", type=int, default=4)
    ap.add_argument("--warm-new", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--flips-per-vector", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=0, help="stop starting new trials after this long")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if a.summarize:
        summarize(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd_dir = Path(a.cmd_dir)
    cmd_dir.mkdir(parents=True, exist_ok=True)

    prompts, shared = build_prompts(a.model, a.followups)
    lo, hi = positions(a.pos, shared)
    print(f"shared prefix {shared} tokens, injecting positions [{lo},{hi})", flush=True)

    # Golden twice through the same code path; abort if the server is not reproducible.
    dry = {"seed": 0, "kinds": a.kinds.split(","), "bits": [0], "pos_lo": lo, "pos_hi": hi, "dry_run": True}
    g1, *_ = run_flow(a.url, cmd_dir, prompts, dry, a.warm_new, a.max_new)
    g2, _, info, secs = run_flow(a.url, cmd_dir, prompts, dry, a.warm_new, a.max_new)
    if g1 != g2:
        print("golden runs differ; fix determinism before sweeping", [first_diff(x, y) for x, y in zip(g1, g2)])
        sys.exit(2)
    gid = hashlib.sha256(json.dumps(g1).encode()).hexdigest()[:12]
    session = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "golden_id": gid, "shared": shared,
               "tree_info": info, "flow_seconds": round(secs, 2), "args": vars(a)}
    gpath = out / "golden.json"
    if gpath.exists() and json.loads(gpath.read_text())["golden_id"] != gid:
        print(f"WARNING: golden differs from earlier sessions ({gid}); trials keep their golden_id")
    gpath.write_text(json.dumps({"golden_id": gid, "outputs": g1}))
    with (out / "sessions.jsonl").open("a") as fh:
        fh.write(json.dumps(session) + "\n")
    print(f"golden ok ({gid}), one flow {secs:.1f}s", flush=True)

    tpath = out / "trials.jsonl"
    done = set()
    if tpath.exists():
        done = {json.loads(l)["trial"] for l in tpath.read_text().splitlines() if l.strip()}
    todo = [(k, b, r) for k in a.kinds.split(",") for b in parse_bits(a.bits) for r in range(a.reps)]
    # interleave bits so a partial sweep still covers every bit evenly
    todo.sort(key=lambda t: (t[2], t[0], t[1]))
    t_start = time.monotonic()
    n_new = 0
    for kind, bit, rep in todo:
        tid = f"{kind}-{a.pos}-f{a.flips_per_vector}-b{bit:02d}-r{rep:03d}"
        if tid in done:
            continue
        if a.max_hours and time.monotonic() - t_start > a.max_hours * 3600:
            print("max-hours reached, stopping")
            break
        seed = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16)
        inject = {"seed": seed, "kinds": [kind], "bits": [bit], "pos_lo": lo, "pos_hi": hi,
                  "flips_per_vector": a.flips_per_vector}
        rec = {"trial": tid, "kind": kind, "bit": bit, "rep": rep, "pos": a.pos, "golden_id": gid}
        try:
            outs, inj, _, secs = run_flow(a.url, cmd_dir, prompts, inject, a.warm_new, a.max_new)
            diffs = [first_diff(o, g) for o, g in zip(outs, g1)]
            rec.update(flips=inj["flips"], first_diff=diffs, n_changed=sum(d is not None for d in diffs),
                       any_changed=any(d is not None for d in diffs), seconds=round(secs, 2), outputs=outs)
        except Exception as e:
            rec.update(error=repr(e))
            print(f"{tid}: ERROR {e!r}", flush=True)
            # a dead server should not burn through the trial list
            try:
                http(a.url + "/health", timeout=10)
            except Exception:
                with tpath.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print("server is down; stopping")
                sys.exit(3)
        with tpath.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        n_new += 1
        if n_new % 20 == 0:
            el = time.monotonic() - t_start
            print(f"{n_new} trials in {el/60:.1f} min ({el/n_new:.1f}s each)", flush=True)
    summarize(out)


if __name__ == "__main__":
    main()
