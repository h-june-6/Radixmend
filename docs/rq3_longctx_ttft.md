# RQ3: 긴 문맥 재측정과 TTFT 원인 (2026-11-16~17)

launch 설정 (4 warps, 1 stage, block 32) 기본값(커밋 7387217) 기준. 일반 모드, 보호 없음 대비. 결과: PC `radixmend_results\20261117-002306\results\rq3\{longctx, ttft_prof}`.

## 1. 긴 공유 프롬프트 (2회 반복)

| 워크로드 | TPOT 이전 설정 (4,2,32) | TPOT 모듈 A | TPOT A+D | TTFT 모듈 A | TTFT A+D |
|---|---|---|---|---|---|
| gsp 4096, c8 | +4.3% | +1.4% | +1.5% | +17.8% | +24.7% |
| gsp 4096, c32 | +19.0% | **+4.0%** | +4.7% | +15.2% | +16.8% |
| gsp 8192, c8 | +5.5% | +1.4% | +1.7% | +25.9% | +29.2% |

decode 비용은 4096 c32를 빼면 목표(3%) 안이다. 4096 c32는 +19%에서 +4.0%로 줄었지만 목표를 조금 넘는다.
남은 큰 비용은 TTFT이고, 프롬프트가 길수록 커진다.

## 2. TTFT 프로파일 (gsp 4096 c8, 32요청, torch profiler)

| 항목 | 보호 없음 | 모듈 A |
|---|---|---|
| TTFT 중앙값 | 626 ms | 672 ms (+7%) |
| `_check_fix_inplace_kernel` (extend 사전 검사) | | +189 ms / 720회 (prefill당 약 9.5 ms) |
| `_encode_scatter_kernel` (저장 시 encode) | | +149 ms |
| GEMM 호출 | 900회, 1,353 ms | 1,008회, 1,544 ms (+191 ms) |

ncu는 서버 정책(RmProfilingAdminOnly=1)으로 막혀 있고 nsys 2023.2는 CUDA 13 커널을 못 봐서 torch profiler를 썼다.
짧은 실행(32요청)이라 TTFT 증가가 본 측정(256요청, +17.8%)보다 작다. 대기열이 길수록 커지는 것으로 본다.

GEMM 호출 증가는 보호 모듈이 GEMM을 건드리지 않으므로 prefill 묶음(chunk, batch 구성)이 달라진 결과로 추정한다(미확인).

## 3. 줄일 후보

1. extend 사전 검사: 같은 prefix를 prefill마다 전부 다시 검사한다. 최근 검사한 노드를 건너뛰면 줄지만 검사와 사용 사이 틈이 생긴다.
   그 틈은 decode 검사(sig)가 같은 KV를 다시 읽으며 메운다는 점을 근거로 들 수 있다. 단 prefill 자체가 깨진 값을 읽을 위험은 남는다.
2. encode를 저장 커널에 합치기 (WBS 개정 때 뺀 항목).
3. GEMM 호출 증가 원인 확인.
