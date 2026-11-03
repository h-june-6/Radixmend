# SGLang v0.5.19로 기준 버전 변경: v0.5.22 분석과 달라진 점

작성: 2026-11-03. 비교 대상은 v0.5.19(`0bcd82237`)와 v0.5.22(`baa9befa3`)다. 경로는 `python/sglang/` 기준이다.

## 왜 바꿨나

대학교에서 빌릴 수 있는 서버(RTX A6000 × 7)의 CUDA가 12.2다. v0.5.20부터 SGLang은 CUDA 13 빌드만 배포한다. CUDA 12 빌드(cu129)가 있는 마지막 버전은 v0.5.19다(공식 설치 문서). v0.5.22를 쓰려면 드라이버를 580 이상으로 올려야 하는데, 공용 서버라 바로 올리기 어렵다. 그래서 기준 버전을 v0.5.19로 낮췄다.

설치 (v0.5.19 공식 문서의 CUDA 12 절차):

```bash
uv pip install --prerelease=allow sglang==0.5.19
uv pip install --force-reinstall torch==2.13.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/
```

cu129 빌드가 드라이버 535 계열(CUDA 12.2)에서 도는지는 CUDA 12 minor version compatibility에 기대는 것이라 **[GPU 확인]**. flashinfer는 cu13 extra로 고정돼 있어 import 단계에서 문제가 생길 수 있다. 우리는 Triton backend를 쓰므로 flashinfer가 실제로 호출되지는 않는다 **[GPU 확인]**.

## 그대로인 것 (v0.5.19에서 다시 확인함)

| 항목 | v0.5.19 위치 |
|---|---|
| 기본 tree는 `UnifiedRadixCache` | `srt/mem_cache/registry.py:187` |
| `node.value`는 `req_to_token` 행을 int64로 복사한 GPU tensor | `srt/mem_cache/unified_radix_cache.py:892,987` |
| KV free list `free_pages`는 GPU int64 tensor | `srt/mem_cache/allocator/token.py:44` |
| `req_to_token`은 int32 | `srt/mem_cache/memory_pool.py:280` |
| KV 저장은 JIT CUDA `store_cache` | `srt/mem_cache/memory_pool.py:157`, `kernels/jit/csrc/elementwise/kvcache.cuh` |
| `_set_kv_buffer_impl` (모듈 A encode 삽입 후보) | `srt/mem_cache/memory_pool.py:141` |
| `TritonAttnBackend` | `srt/layers/attention/triton_backend.py:147` |
| 다음 decode 입력 token은 GPU `FutureMap.output_tokens_buf`에서 가져옴 | `srt/managers/overlap_utils.py:98-102` |
| `seq_lens_cpu` CPU 미러 | `srt/managers/schedule_batch.py:2290` |
| 체크워드 크기를 더할 `_compute_cell_size` | `srt/model_executor/pool_configurator.py:248` |
| sm86 기본 backend는 flashinfer, Triton은 명시해야 함 | `srt/arg_groups/model_override_base.py:296-329` |
| CUDA graph에 BREAKABLE(prefill) 모드 있음 | `srt/model_executor/cuda_graph_config.py:44-53` |
| `srt/kv_canary/` 인프라 | 있음 |

## 달라진 것과 그에 따른 조치

| 항목 | v0.5.22 | v0.5.19 | 조치 |
|---|---|---|---|
| tree core 기본값 | Rust (`environ.py:688`) | **Python** (`environ.py:643`) | 노드 접근이 기본으로 된다. 그래도 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`은 실행 스크립트에 명시해 둔다. 실험 명세의 "Rust 대비 처리량 비교" 항목은 뺀다 |
| tree 삽입 API | `checkpoint_into_tree`, `checkpoint_kv_cache` | **`cache_finished_req`(:838), `cache_unfinished_req`(:925)** | hash baseline의 digest 계산, 모듈 C의 삽입 게이팅 위치를 이 두 함수로 옮긴다. chunked prefill은 `stash_chunked_request` → `maybe_cache_unfinished_req`(`scheduler.py:3245-3246`) 경로다 |
| Triton kv_indices 생성 | `KVIndexTranslator.pack_read_stream`을 거침 | `_fill_kv_indptr_and_indices`(`triton_backend.py:460`)에서 `create_flashinfer_kv_indices_triton`을 **바로 호출** | 모듈 B decode의 choke point가 한 단계 짧아진다. 이 함수 한 곳에 decode를 넣으면 decode와 CUDA graph 경로(`:508-515`)를 같이 덮는다 |
| hash baseline용 GPU 해시 | `kernels/ops/memory/adler32.py`(Adler-32) | **없음**. `kernels/ops/memory/gpu_tensor_hash.py`(`gpu_tensor_hash` :141)만 있음 | 빠른 해시 버전은 `gpu_tensor_hash`로 바꾼다. SHA-256(CPU) 버전은 그대로 |
| component 파일 | `components/base.py`, `full.py` | `components/tree_component.py`(`lock_ref` :50), `full_component.py`(`FullComponent` :43) | 경로만 바꾼다 |
| `UnifiedTreeCore` | `unified_tree_core.py:416` | `unified_tree_core.py:386` | 줄 번호만 바꾼다 |
| 의존성 | torch 2.14.1, flashinfer 0.7.1, sglang-kernel 0.4.9 | torch 2.13.0, flashinfer 0.6.18, sglang-kernel 0.4.6.post1 | 실험 명세와 보고서의 환경 표를 고친다 |

`docs/sglang_code_analysis.md`의 줄 번호는 v0.5.22 기준으로 남겨 둔다. 패치할 때는 이 문서의 v0.5.19 위치를 먼저 본다. 위 표에 없는 세부 줄 번호는 패치하는 주차에 그 파일을 열어 다시 맞춘다.
