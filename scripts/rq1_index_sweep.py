"""RQ1-5 index-fault sweep: one bit flip in a slot-index tensor per trial.

Targets (experiment_spec 3장): node_value (radix tree slot list, int64),
free_pages (allocator free list, int64), req_to_token (int32 row of a running
request) and seq_lens (running batch, int64). Each target is tried with a low
and a high bit group. Trial flow is the same as rq1_kv_sweep (flush, warm-up,
inject, follow-ups), except that req_to_token / seq_lens are injected while the
first follow-up is decoding.

Outcome per trial: ``crash`` (server died; the supervisor restarts it),
``error`` (request failed but the server lives), ``changed``, ``unchanged``,
``not_triggered`` (a deferred flip never found a running request).

    python scripts/rq1_index_sweep.py --out results/rq1/index --reps 30
    python scripts/rq1_index_sweep.py --out results/rq1/index --summarize
Run it under scripts/server/run_supervised.sh so that crashes restart the server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq1_kv_sweep as kv  # noqa: E402

BIT_GROUPS = {
    # int32 req_to_token: bit 31 is the sign
    "req_to_token": {"low": list(range(0, 8)), "high": list(range(18, 32))},
    "node_value": {"low": list(range(0, 8)), "high": list(range(18, 64))},
    "free_pages": {"low": list(range(0, 8)), "high": list(range(18, 64))},
    "seq_lens": {"low": list(range(0, 8)), "high": list(range(18, 64))},
}
DEFERRED = {"req_to_token", "seq_lens"}


def send_cmd(cmd_dir: Path, op: dict) -> str:
    cid = f"{time.time_ns()}"
    tmp = cmd_dir / f"{cid}.tmp"
    tmp.write_text(json.dumps(op))
    os.replace(tmp, cmd_dir / f"{cid}.cmd.json")
    return cid


def wait_done(cmd_dir: Path, cid: str, timeout: float) -> dict | None:
    done = cmd_dir / f"{cid}.done.json"
    t0 = time.monotonic()
    while not done.exists():
        if time.monotonic() - t0 > timeout:
            return None
        time.sleep(0.005)
    res = json.loads(done.read_text())
    done.unlink()
    return res


def server_alive(base: str) -> bool:
    try:
        kv.http(base + "/health", timeout=10)
        return True
    except Exception:
        return False


def trial(a, cmd_dir, prompts, shared, golden, target, group, seed):
    kv.flush(a.url)
    kv.generate(a.url, prompts[0], a.warm_new)
    op = {"op": "inject_index", "target": target, "seed": seed, "bits": BIT_GROUPS[target][group],
          "pos_lo": 0, "pos_hi": shared, "after_tokens": a.after_tokens, "head": 64}
    outs, err = [], None
    if target in DEFERRED:
        cid = send_cmd(cmd_dir, op)
        res_box = {}

        def first():
            try:
                res_box["out"] = kv.generate(a.url, prompts[1], a.max_new)
            except Exception as e:
                res_box["err"] = repr(e)

        th = threading.Thread(target=first)
        th.start()
        th.join(timeout=a.req_timeout)
        inj = wait_done(cmd_dir, cid, 1.0)
        if inj is None:
            kv.hook(cmd_dir, {"op": "cancel", "target_id": cid})
            inj = {"ok": True, "not_triggered": True}
        if th.is_alive():
            err = "first follow-up timed out"
        elif "err" in res_box:
            err = res_box["err"]
        else:
            outs.append(res_box["out"])
        rest = prompts[2:]
    else:
        inj = kv.hook(cmd_dir, op)
        rest = prompts[1:]
    if err is None:
        try:
            outs += [kv.generate(a.url, p, a.max_new) for p in rest]
        except Exception as e:
            err = repr(e)
    return inj, outs, err


def summarize(out: Path) -> None:
    rows = [json.loads(l) for l in (out / "trials.jsonl").read_text().splitlines() if l.strip()]
    keys = sorted({(r["target"], r["group"]) for r in rows})
    cols = ["crash", "error", "changed", "unchanged", "not_triggered"]
    lines = ["target,group,n," + ",".join(cols) + ",new_owner_counts"]
    for t, g in keys:
        rs = [r for r in rows if r["target"] == t and r["group"] == g]
        cnt = {c: sum(r["outcome"] == c for r in rs) for c in cols}
        owners = {}
        for r in rs:
            o = (r.get("inject") or {}).get("new_owner")
            if o:
                owners[o] = owners.get(o, 0) + 1
        lines.append(f"{t},{g},{len(rs)}," + ",".join(str(cnt[c]) for c in cols) + "," +
                     ";".join(f"{k}={v}" for k, v in sorted(owners.items())))
    (out / "summary.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--cmd-dir", default=f"/tmp/{os.environ.get('USER', 'user')}/rm_cmd")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="node_value,free_pages,req_to_token,seq_lens")
    ap.add_argument("--groups", default="low,high")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--followups", type=int, default=4)
    ap.add_argument("--warm-new", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--after-tokens", type=int, default=8)
    ap.add_argument("--req-timeout", type=float, default=120)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.summarize:
        summarize(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd_dir = Path(a.cmd_dir)
    cmd_dir.mkdir(parents=True, exist_ok=True)
    for p in cmd_dir.glob("*.cmd.json"):  # stale commands from a crashed run
        p.unlink()

    prompts, shared = kv.build_prompts(a.model, a.followups)
    dry = {"seed": 0, "kinds": ["v"], "bits": [0], "pos_lo": 0, "pos_hi": shared, "dry_run": True}
    g1, *_ = kv.run_flow(a.url, cmd_dir, prompts, dry, a.warm_new, a.max_new)
    g2, *_ = kv.run_flow(a.url, cmd_dir, prompts, dry, a.warm_new, a.max_new)
    if g1 != g2:
        print("golden runs differ; fix determinism first")
        sys.exit(2)
    gid = hashlib.sha256(json.dumps(g1).encode()).hexdigest()[:12]
    print(f"shared {shared}, golden ok ({gid})", flush=True)

    tpath = out / "trials.jsonl"
    done = set()
    if tpath.exists():
        done = {json.loads(l)["trial"] for l in tpath.read_text().splitlines() if l.strip()}
    todo = [(t, g, r) for t in a.targets.split(",") for g in a.groups.split(",") for r in range(a.reps)]
    todo.sort(key=lambda x: (x[2], x[0], x[1]))
    for target, group, rep in todo:
        tid = f"{target}-{group}-r{rep:03d}"
        if tid in done:
            continue
        seed = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16)
        rec = {"trial": tid, "target": target, "group": group, "rep": rep, "golden_id": gid}
        # mark the trial as started first: if the server dies mid-trial the supervisor's
        # rerun sees it as done and the crash is kept on record
        with tpath.open("a") as fh:
            fh.write(json.dumps({**rec, "outcome": "crash", "provisional": True}) + "\n")
        t0 = time.monotonic()
        try:
            inj, outs, err = trial(a, cmd_dir, prompts, shared, g1, target, group, seed)
        except Exception as e:
            inj, outs, err = None, [], repr(e)
        alive = server_alive(a.url)
        if not alive:
            outcome = "crash"
        elif inj and inj.get("not_triggered"):
            outcome = "not_triggered"
        elif err:
            outcome = "error"
        else:
            diffs = [kv.first_diff(o, g) for o, g in zip(outs, g1)]
            rec.update(first_diff=diffs, n_changed=sum(d is not None for d in diffs))
            outcome = "changed" if rec["n_changed"] else "unchanged"
        rec.update(outcome=outcome, inject=inj, error=err, seconds=round(time.monotonic() - t0, 2))
        # replace the provisional line
        lines = [l for l in tpath.read_text().splitlines() if l.strip()]
        lines[-1] = json.dumps(rec)
        tpath.write_text("\n".join(lines) + "\n")
        print(f"{tid}: {outcome}" + (f" ({err[:120]})" if err else ""), flush=True)
        if not alive:
            print("server is down; exiting for restart")
            sys.exit(3)
    summarize(out)


if __name__ == "__main__":
    main()
