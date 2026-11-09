# RQ3 module A fused prototype: correction and TPOT (2026-11-08~09)

Code: `radixmend/sglang_secded.py`.
- Decode check fused into SGLang triton grouped decode: commit 6cfe590.
- Pre-extend prefix check with in-place fix: commit d43acb3.

Server: RTX A6000, SGLang 0.5.22, Qwen3-8B, triton backend, page size 1, mem-fraction 0.85.
Raw data is on the PC at `C:\Users\kimhojun\radixmend_results\20261109-125614\results\rq3\`.

## Correctness

GPU tests:
- Before the extend check: 38/38 passed.
- After d43acb3: 22/22 passed (4 files).

With no fault, the 5 greedy prompts gave bitwise-identical output with and without SECDED, and the sweep golden stayed `94c11b0adbc5`.

The checkword buffers add 200 MiB of GPU memory. The KV pool keeps the same size (180,040 tokens).

Fault sweeps used one V bit flip in the shared prefix, then 4 follow-up requests (same flow as RQ1-2), 10 reps per bit.

| V bit | No protection (RQ1-2, n=150) | Decode check only | Decode + extend check |
|---|---|---|---|
| 14 | 95.3% | 9/10 | **0/10** (95% CI 0–27.8%) |
| 7 | 68.7% | 6/10 | **0/10** (95% CI 0–27.8%) |

### Decode check only

The decode counters rose: 2,451 for bit 14 and 2,198 for bit 7. They count reads, not events.

Output still changed. Every changed trial differed from the first token of each follow-up. That token is produced while prefilling the follow-up, and the prefill reads the corrupted shared prefix through extend attention, which was not checked yet.

### Decode + extend check

| Counter | Bit 14 sweep | Bit 7 sweep |
|---|---|---|
| Extend V corrected | 10 | 10 |
| Extend V uncorrectable | 0 | 0 |
| Decode K/V corrected or uncorrectable | 0 | 0 |

The extend check found each flip exactly once and fixed it in place, so decode had nothing left to fix.

Ten reps per bit is too few to claim 0%. More reps are needed.

## TPOT / TTFT

Measurement conditions:
- Non-deterministic mode, 2 repeats; the tables show the median.
- Shared-prefix (gsp): 128 requests at c1, 256 requests at c8 and c32.
- Random: 500 requests.
- Repeat-to-repeat spread was ≤1% for TPOT and mostly ≤5% for TTFT. Exceptions: c gsp c1 at 13% and d gsp c32 at 15%.

Configurations:
- a: no protection
- b: encode only (`RADIXMEND_SECDED_CHECK=0`)
- c: encode + decode check (6cfe590)
- d: c + extend check (d43acb3)

### TPOT in ms (change vs a)

| Workload | a | b | c | d |
|---|---|---|---|---|
| gsp c1 | 24.12 | 24.25 (+0.5%) | 25.93 (+7.5%) | 25.93 (+7.5%) |
| gsp c8 | 27.56 | 27.69 (+0.5%) | 32.92 (+19.4%) | 32.94 (+19.5%) |
| gsp c32 | 34.62 | 34.71 (+0.3%) | 62.56 (+80.7%) | 62.67 (+81.0%) |
| random c8 | 28.91 | 29.11 (+0.7%) | 30.95 (+7.1%) | 30.98 (+7.2%) |
| random c32 | 49.44 | 49.70 (+0.5%) | 54.77 (+10.8%) | 54.87 (+11.0%) |

### TTFT in ms (change vs a)

| Workload | a | b | c | d |
|---|---|---|---|---|
| gsp c1 | 64.9 | 63.4 (−2.3%) | 69.3 (+6.8%) | 69.2 (+6.7%) |
| gsp c8 | 245.6 | 246.4 (+0.3%) | 268.4 (+9.3%) | 265.1 (+7.9%) |
| gsp c32 | 793.1 | 798.9 (+0.7%) | 871.9 (+9.9%) | 911.8 (+15.0%) |
| random c8 | 142.2 | 142.4 (+0.1%) | 146.1 (+2.7%) | 146.2 (+2.8%) |
| random c32 | 182.8 | 183.9 (+0.6%) | 192.3 (+5.2%) | 192.5 (+5.3%) |

### Notes on the measurement

- Today's gsp c8 TTFT for a (245.6 ms) is lower than RQ1-1 (324.7 ms) because the request count differs (256 vs 1,024). TPOT matches: 27.56 vs 27.8 ms.

## Findings

- **Encode at store is nearly free.** TPOT rises 0.3–0.7%, which meets the 3% target.
- **The cost is in the decode check.** It grows with the KV volume read per decode step:
  - gsp, about 2,300 tokens per request: +81% at c32;
  - random, about 650 tokens: +7–11%.

  This follows the trend of the standalone kernel pre-measurement.
- **The extend check is cheap and essential.** It adds at most 0.3 points of TPOT. Without it, faults in a shared prefix still reach the output, because prefill reads them.
- **Next step for RQ3:** make the decode check cheaper. Candidates:
  - a fast path for clean tiles (overall parity only, with the full syndrome only when it fails);
  - a cheaper S_e reduction (warp-level, or pack two bf16 per 32-bit word);
  - shorter codewords or check placement (e.g. check once per slot at first read and scrub, then skip until rewritten).
