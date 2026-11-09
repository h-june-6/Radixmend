"""Send fixed prompts to /generate (greedy, 128 new tokens) and save the outputs.

    python scripts/smoke_client.py out.json
    python scripts/smoke_client.py --compare a.json b.json   # exit 1 if any output differs

Used by scripts/server/day1.sh to check that a server change is bit-neutral
when no fault is injected.
"""

import json
import sys
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Explain in two sentences why the sky appears blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "List three common causes of memory bit flips in computer hardware:",
    "대한민국의 수도와 그 도시의 특징을 설명하면",
]


def run(out_path: str, url: str = "http://127.0.0.1:30000") -> None:
    results = []
    for p in PROMPTS:
        body = json.dumps({"text": p, "sampling_params": {"temperature": 0, "max_new_tokens": 128}}).encode()
        req = urllib.request.Request(url + "/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        m = d.get("meta_info", {})
        results.append({"prompt": p, "text": d.get("text"), "output_ids": d.get("output_ids"),
                        "cached_tokens": m.get("cached_tokens"), "e2e_latency": m.get("e2e_latency")})
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out_path)


def compare(a_path: str, b_path: str) -> int:
    a, b = json.load(open(a_path, encoding="utf-8")), json.load(open(b_path, encoding="utf-8"))
    bad = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x["output_ids"] == y["output_ids"]:
            continue
        bad += 1
        pos = next((k for k, (s, t) in enumerate(zip(x["output_ids"], y["output_ids"])) if s != t), None)
        print(f"prompt {i}: differs at token {pos}")
    print(f"{len(a) - bad}/{len(a)} identical")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    run(sys.argv[1])
