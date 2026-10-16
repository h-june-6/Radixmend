# [RadixMend 1주차] SGLang 코드를 뜯어보고 실험 환경을 정하다

> RadixMend는 SGLang의 radix tree KV 캐시에서 생기는 비트플립을, attention 커널이 KV를 읽는 순간 교정하는 프로젝트다. 16주 계획 중 1주차 기록이다.

## 이번 주 목표 (WBS)

- SGLang 버전 확정, radix tree·KV 풀·attention 커널 코드 분석
- attention 백엔드 결정
- 워크로드 선정, 실험 환경·재현 스크립트 구성
- 결함 주입 도구 개발 착수

## 1. 버전 고정: SGLang v0.5.22

분석 시점의 최신 릴리스인 v0.5.22(commit `baa9befa3`)로 고정했다. 기간 중에는 올리지 않는다. 최근 SGLang은 캐시 구조를 빠르게 바꾸고 있어서, 중간에 버전을 올리면 패치를 처음부터 다시 맞춰야 할 수 있기 때문이다.

이 버전의 의존성은 torch 2.14.1, flashinfer 0.7.1(cu13)이다. CUDA 13 기반이라 GPU 서버 드라이버가 CUDA 13을 지원하는지부터 확인해야 한다.

## 2. 코드 분석에서 나온 것: 설계 가정과 다른 점

기술서를 쓸 때는 `mem_cache/radix_cache.py`의 `RadixCache`를 기준으로 생각했다. 실제 코드를 따라가 보니 몇 가지가 달랐다.

**기본 radix tree는 Rust 구현이다.** 기본 factory는 `UnifiedRadixCache`를 만들고, 그 안의 트리 코어는 기본값이 Rust다(`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = "rust"`). Rust 코어는 Python 노드 객체를 내주지 않는다. 노드에 poisoned 플래그나 epoch를 붙이려면 환경 변수로 Python 코어를 강제해야 한다.

```bash
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
```

**`lock_ref`는 노드 필드가 아니다.** 노드마다 component(FULL/SWA/MAMBA)별 데이터가 있고, KV slot과 참조 카운트는 `node.component_data[FULL].value`와 `.lock_ref`에 있다.

**root는 KV를 갖지 않는다.** 이건 설계 가정과 맞았다. 그래서 "가장 많이 공유되는 노드"는 root의 자식, 즉 첫 레벨 노드다.

**KV를 저장하는 커널은 Triton이 아니라 JIT CUDA C++다.** `store_cache` 커널은 slot 번호 범위를 device assert로 검사하고 slot 0(padding)은 건너뛴다. encode-at-store를 넣으려면 이 커널을 고치거나, 같은 자리에 Triton store+encode 커널을 끼워야 한다. 프로토타입은 후자가 빠르다.

**보호 대상 텐서의 실제 dtype**

| 텐서 | 위치 | dtype | 비고 |
|---|---|---|---|
| `req_to_token` | GPU | int32 | 행 0은 padding |
| `node.value` | GPU | int64 | `req_to_token` 행을 복사한 것 |
| KV free list (`free_pages`) | GPU | int64 | slot 0 제외 |
| `seq_lens` | GPU + CPU 미러 | int64 | 산술 소비자가 많음 |
| `out_cache_loc` | GPU | int64 | |

`seq_lens`는 GPU에서 `+1`, `cumsum` 같은 연산에 계속 쓰인다. 상위 비트에 ECC를 넣으면 모든 소비자에 디코드를 넣어야 한다. 그래서 CPU 미러와 대조하는 방식이 더 싸다는 결론을 냈다. 이건 5주차 설계 확정 때 반영할 예정이다.

**모듈 C의 순서 가정이 깨진다.** overlap scheduler에서는 chunked prefill 결과가 트리에 들어가는 시점이, 그 chunk의 오류 플래그를 처리하는 시점보다 앞선다. "플래그 처리 전에는 트리에 넣지 않는다"를 지키려면 이 순서를 바꿔야 한다. 그래서 모듈 C 초기 구현은 `--disable-overlap-schedule`로 시작하기로 했다.

## 3. attention 백엔드: Triton

| 백엔드 | 기본 사용 | 커널 위치 | RadixMend 패치 |
|---|---|---|---|
| FlashInfer | Ampere(A6000) 기본 | 외부 라이브러리(JIT CUDA) | fork 필요, 어려움 |
| FlashAttention 3 | Hopper 기본 | sgl-kernel(CUTLASS) | 재빌드 필요, 사실상 불가 |
| **Triton** | 명시해야 사용 | SGLang 저장소 안 Python/Triton | **타일 load 직후에 코드 삽입 가능** |

Triton을 고른 이유는 세 가지다.

1. decode, extend, `kv_indices` 생성 커널이 모두 저장소 안의 Triton 코드다.
2. decode 커널(`_fwd_grouped_kernel_stage1`)은 head_dim 128일 때 K 타일의 열 하나가 정확히 (토큰, KV 헤드) 벡터 하나다. 그리고 그 벡터를 스텝·레이어마다 한 번만 읽는다. load 직후에 신드롬을 계산하고 레지스터에서 고치면 되고, 검사가 중복되지 않는다.
3. extend 커널은 같은 prefix 벡터를 여러 번 읽는다. 그래서 오류 로그에는 중복 제거가 필요하다. 메모리 write-back을 별도 scrub 커널로 빼는 기술서의 설계가 맞다는 근거이기도 하다.

대가는 A6000에서 FlashInfer보다 느릴 수 있다는 점이다. 그래서 기준선은 "패치하지 않은 Triton"으로 잡고, FlashInfer 수치는 참고로만 보고하기로 했다.

## 4. 실험 환경과 워크로드 명세

| 항목 | 값 |
|---|---|
| GPU | RTX A6000 48GB (GDDR6, 하드웨어 ECC 꺼짐) |
| 모델 | Qwen3-8B(주), Llama-3-8B-Instruct(보조), bf16 |
| 서버 옵션 | `--attention-backend triton --page-size 1 --tp-size 1 --kv-canary none`, `--max-total-tokens` 고정 |
| 샘플링 | greedy, 고정 seed (golden 출력과 비교하기 위해) |

Qwen3-8B를 고른 이유는 선행 연구(Yamamoto & Matsuura, 2026)가 이 모델로 실험해서 결과를 직접 비교할 수 있어서다.

워크로드는 공유 prefix가 있는 것 위주로 골랐다.

| 이름 | 구성 | 보려는 것 |
|---|---|---|
| W1 | 공통 시스템 프롬프트 + ShareGPT 질문 | 첫 레벨 노드·sink 피해 |
| W2 | GSM8K 8-shot, MMLU 5-shot | 정답률로 보는 의미적 피해 |
| W3 | 다중 턴 대화 | 파생 오염 |
| W4 | 분기 트리 | 자손 노드 수와 계보 무효화 |
| W5 | W1 + `SGLANG_RADIX_FORCE_MISS=1` | 공유 효과를 뺀 대조군 |

## 5. 결함 주입 도구 착수

GPU 없이도 돌아가도록 torch 텐서만 다루게 만들었다. CPU에서 테스트하고 GPU에서는 같은 코드를 그대로 쓴다.

```python
from radixmend import fault_inject as fi

# 레이어 3, K 텐서, slot 1024, 헤드 2, 원소 77의 비트 9를 뒤집는다
recs = fi.inject_kv(pool.k_buffer, pool.v_buffer,
                    [fi.KVFault(layer=3, kind="k", slot=1024, head=2, elem=77, bit=9)])
fi.revert(recs, k_buffer=pool.k_buffer, v_buffer=pool.v_buffer)  # 시행 후 원상복구
```

트리에서 노리는 위치도 고를 수 있다. 첫 레벨 노드(`first_level_nodes`), sink 토큰 slot(`sink_slots`), 공유도 순 노드(`nodes_by_sharing`)를 지원한다. SGLang 기본 트리의 `component_data` 구조와 예전 `TreeNode` 구조를 둘 다 읽는다.

## 남은 일

- GPU 서버에서 드라이버 확인, Triton 백엔드로 모델 띄우기
- Python 트리 코어가 Rust 대비 얼마나 느린지 측정
- 재현 스크립트(워크로드 생성, 서버 실행)는 GPU에서 처음 돌릴 때 완성

## 다음 주

결함 주입기를 완성하고, SECDED 참조 구현을 만들어 모든 단일·이중 오류에 대해 전수 검증한다.
