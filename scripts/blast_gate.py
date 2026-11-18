"""Correctness gate for module C blast runs (rq4_blast.py output).

A trial passes when every request it changed is also changed by some fake control, i.e. by a
recompute without any corrupted data. Controls are one-step (--kinds fake) and two-step
(--kinds fake --fake-steps 2) runs: a request recomputed twice can differ from one recomputed
once, so the allowed set is the union of both. Any trial whose already-committed tokens differ
from golden (committed_bad, recorded since 42bff8a) fails regardless.

    python scripts/blast_gate.py --runs results/x/blast_ac_n32 --controls results/x/fake1 results/x/fake2
prints "ok" or "fail [...]" and exits 0 / 1.
"""

import argparse
import json
import sys
from pathlib import Path


def trials(d):
    return [t for t in (json.loads(l) for l in (Path(d) / "trials.jsonl").read_text().splitlines() if l.strip())
            if not t.get("golden")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--controls", nargs="+", required=True)
    a = ap.parse_args()
    allowed = set()
    for d in a.controls:
        for t in trials(d):
            allowed.update(t.get("changed") or [])
    bad = []
    for d in a.runs:
        for t in trials(d):
            extra = sorted(set(t.get("changed") or []) - allowed)
            if t.get("error") or extra or t.get("committed_bad"):
                bad.append((Path(d).name, t.get("trial"), t.get("error"), extra, t.get("committed_bad")))
    print("ok" if not bad else "fail " + str(bad[:5]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
