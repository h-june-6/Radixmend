# SGLang v0.5.22 코드 분석: RadixMend 패치 지점

대상 체크아웃: `/home/claude/sglang`, tag `v0.5.22`, commit `baa9befa3`.
모든 경로는 `python/sglang/` 기준 상대경로(`srt/...`, `kernels/...`)이며, 줄 번호는 이 체크아웃 기준이다.
이 노트는 RadixMend 모듈 A–D, fault injector, hit-time hash baseline을 붙일 위치를 정리한다. 설계 문서와 다른 사실은 **[설계 가정과 다름]**으로 표시했다.

---

## 0. 핵심 요약: 설계 가정과 다른 사실

1. **기본 radix tree는 `mem_cache/radix_cache.py`의 `RadixCache`가 아니다.** 기본 factory는 항상 `UnifiedRadixCache`를 만들고(`srt/mem_cache/registry.py:83-146`, 인스턴스 생성은 `:191`), 그 안의 TreeCore는 기본값이 **Rust 구현**이다(`srt/environ.py:688` `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = EnvStr("rust")`). Rust core는 Python 노드 객체를 내주지 않는다(`node_by_id`와 `root_node`가 `NotImplementedError`, `srt/mem_cache/rust_tree_core/adapter.py:471-478`). 따라서 **`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`을 강제하고** `srt/mem_cache/unified_cache/unified_tree_core.py`(Python core)를 패치해야 한다. `radix_cache.RadixCache`는 FlexKV 경로에서만 쓰이고(`srt/mem_cache/storage/flexkv/flexkv_radix_cache.py:44`), 이 파일에서는 `RadixKey`만 공용으로 쓰인다.
2. `cache_finished_req` / `cache_unfinished_req`는 **없다.** 대신 `BasePrefixCache.checkpoint()`(`srt/mem_cache/base_prefix_cache.py:476-485`), `checkpoint_kv_cache()`(`srt/mem_cache/common.py:253`), `release_kv_cache(..., checkpoint=True|False)`(`srt/mem_cache/common.py:378`)가 그 역할을 한다.
3. `lock_ref`는 노드 필드가 아니라 **component별** 필드다: `node.component_data[ComponentType.FULL].lock_ref`(`srt/mem_cache/unified_cache/components/base.py:68-75`). `last_access_time`은 wall-clock이 아니라 **논리 카운터**다(`components/base.py:160-164`). cold-node scrub에 쓸 시간 정보는 직접 추가해야 한다.
4. `node.value`는 **GPU의 int64 tensor**이고 `req_to_token` row slice를 복사한 것이다(`srt/mem_cache/unified_radix_cache.py:1094,1150`). root의 value는 Python `[]`로, **root는 KV를 갖지 않는다**(`unified_tree_core.py:497-503`). 이 부분은 설계 가정과 일치한다.
5. KV free list는 설계 가정대로 **GPU int64 tensor**다(`free_pages`, `srt/mem_cache/allocator/token.py:44-46`). 반면 `ReqToTokenPool`의 row free list(`free_slots`)는 Python list다(`srt/mem_cache/memory_pool.py:408`). Module B 대상은 앞의 것이다.
6. dtype: `req_to_token`은 **int32**(`memory_pool.py:405`)다. `node.value`, `out_cache_loc`, `free_pages`, `seq_lens`는 int64다. Triton backend의 `kv_indices`는 int64(`srt/layers/attention/triton_backend.py:861`), FlashInfer의 `kv_indices`는 int32(`srt/layers/attention/flashinfer_backend.py:1758`)다.
7. **`seq_lens`에는 CPU 미러 `seq_lens_cpu`(int64)가 있다**(`srt/managers/schedule_batch.py:2759-2770`). 또 GPU `seq_lens`는 곳곳에서 산술에 쓰인다(`+1`, `cumsum`). 그래서 ECC보다 CPU 미러와 대조하는 편이 싸다(4장 참조).
8. **다음 decode step의 입력 token은 CPU가 아니라 GPU `FutureMap.output_tokens_buf`에서 gather한다.** overlap 모드와 non-overlap 모드 모두 그렇다(`srt/managers/overlap_utils.py:87-123`, `srt/managers/scheduler.py:4830-4855`). sampled token을 버리려면 CPU 쪽 처리만으로는 부족하고 이 버퍼도 무효화해야 한다.
9. **chunked prefill chunk의 tree insert는 다음 `get_next_batch_to_run`에서 일어난다**(`scheduler.py:3786-3793` → `stash_chunked_request` `:3644-3648` → `checkpoint_kv_cache`). overlap 루프에서는 이 호출이 해당 chunk 결과의 `process_batch_result`보다 **먼저** 실행된다(`scheduler.py:1996-2036`). 결국 "flag 처리 전에는 insert하지 않는다"는 설계 가정이 지금 코드에서는 깨진다.
10. **이미 비슷한 인프라가 있다.**
    - `srt/kv_canary/`: slot별 canary와 fingerprint, sweep, 그리고 KV byte와 `req_to_token`을 뒤집는 perturbation injector.
    - PD disaggregation의 Adler-32 KV checksum(`srt/disaggregation/checksum.py:82-117`).
    - `TreeNode.hash_value`: token-id SHA-256이며 KV 내용 hash가 아니다(`srt/mem_cache/utils.py:166-176`).
11. KV store kernel `store_cache`는 Triton이 아니라 **in-repo JIT CUDA C++**이다(`kernels/jit/csrc/elementwise/kvcache.cuh:44-82`). 수정은 가능하지만 CUDA 코드를 고쳐야 한다. 이 kernel은 `index < size_limit`를 device assert로 검사하고 slot 0 쓰기는 건너뛴다.
12. 기본 attention backend는 Ampere(sm86)에서 **flashinfer**, Hopper에서 **fa3**이다(`srt/arg_groups/model_override_base.py:296-324`). 둘 다 외부 컴파일 kernel이다. Triton backend는 명시적으로 지정해야 쓰인다.

---

## 1. 대상 버전과 범위

- 버전: v0.5.22 (`git describe` = `v0.5.22`). 의존성 고정 버전(`python/pyproject.toml:6,42,81`): `torch==2.14.1`, `flashinfer_python[cu13]==0.7.1`, `sglang-kernel==0.4.9`. 이 wheel들은 CUDA 13 대상이므로 드라이버 버전을 먼저 확인해야 한다(9장).
- 가정한 모델과 경로: 표준 MHA/GQA decoder, 즉 Llama-3-8B(32 layer, 32 q-head, 8 kv-head, head_dim 128)와 Qwen3-8B(36 layer, 8 kv-head, head_dim 128). bf16 KV(`--kv-cache-dtype auto`), TP=1, speculative decoding 없음, HiCache/PD 없음.
  - CUDA에서 Llama·Qwen3는 fused rope+store를 쓰지 않는다. `enable_fused_set_kv_buffer`(`srt/models/utils.py:286-303`)는 gpt_oss, qwen3_moe, bailing 등에서만 호출된다. Qwen3의 fused KV write 경로는 ROCm/aiter 전용이다(`srt/models/qwen3.py:206-288`). 따라서 KV 쓰기는 항상 attention backend → `pool.set_kv_buffer` 경로를 탄다.
- KV pool 클래스: `MHATokenToKVPool`(`srt/mem_cache/memory_pool.py:2198`). 생성 위치는 `srt/mem_cache/kv_cache_configurator.py:2425-2475`.
- Allocator: `page_size==1`이고 DCP가 아니면 `TokenToKVPoolAllocator`를 쓰고, 그 밖에는 `PagedTokenToKVPoolAllocator`를 쓴다(`kv_cache_configurator.py:2593-2615`).
- `page_size` 기본값: CUDA에서는 1(`srt/arg_groups/overrides.py:1303-1326`). trtllm_mha, flashmla 등 일부 backend는 64로 강제한다(`overrides.py:1085-1149`).
- CUDA graph 기본값: decode는 `FULL`, prefill은 CUDA에서 `BREAKABLE`(`srt/model_executor/cuda_graph_config.py:119-123,150-153`). 즉 **extend 경로도 기본적으로 graph capture된다.**
- **page_size>1일 때 달라지는 것**
  - free list가 page id 단위가 된다(`allocator/paged.py:395-399`).
  - free는 `torch.unique(free_index // page_size)` 또는 `free_index[::ps] // ps`로 page를 얻는다(`paged.py:328-357`).
  - alloc은 `alloc_extend` / `alloc_decode` kernel을 쓰고 `last_loc`가 필요하며, prefix가 없으면 sentinel `-1`을 쓴다(`srt/mem_cache/allocation.py:427-430`, `:120-148`).
  - tree key와 value가 page 정렬되고, 마지막 partial page는 tree가 아니라 req row에만 남는다(`radix_cache.py:533-537` 주석, `unified_radix_cache.py:1148-1150`).
  - Triton kernel은 `PAGE_SIZE` constexpr로 page-aware 주소를 계산한다(`kernels/ops/attention/decode_attention.py:648-657`).
  - Module B 인코딩은 page id 산술(`// ps`, `% ps`) 전에 decode해야 한다.
  - hash baseline은 page 단위 digest가 자연스럽다.

---

## 2. Radix tree

### 2.1 클래스 계층과 선택

| 계층 | 파일:줄 | 비고 |
|---|---|---|
| `BasePrefixCache` (ABC) | `srt/mem_cache/base_prefix_cache.py:383` | `checkpoint`(:476), `free_kv_row`(:487), `lock`/`unlock`(:528-534), `path_device_indices`(:536), `prefix_device_indices`(:541) |
| `UnifiedRadixCache` | `srt/mem_cache/unified_radix_cache.py:175` | 실제 기본 tree. component(FULL/SWA/MAMBA)를 조립하고 HiCache와 action을 실행한다 |
| `UnifiedTreeCore` (Python) | `srt/mem_cache/unified_cache/unified_tree_core.py:416` | 트리 구조, LRU, evictable leaf set, match/insert/split/evict |
| `RustUnifiedTreeCore` | `srt/mem_cache/rust_tree_core/adapter.py:324` | 기본값이지만 Python 노드 접근 불가(:471-478) |
| `FullComponent` | `srt/mem_cache/unified_cache/components/full.py:45` | FULL KV의 lock, split, evict, match-validator |
| `RadixCache` (레거시) | `srt/mem_cache/radix_cache.py:281` | 기본 경로에서 쓰이지 않음 |

- 선택 흐름: `create_tree_cache`(`registry.py:229`) → `default_radix_cache_factory`(`:83`) → `create_unified_radix_cache`(`:149`) → `UnifiedRadixCache(params)`(`:191`).
- TreeCore 선택: `select_tree_core_backend`(`unified_cache/tree_core_registry.py:124-131`)가 env를 읽고, `_rust_fallback_reason`(`:44-106`)이 Python core로 폴백할 조건을 정한다. 폴백 조건은 toolchain 없음, non-linux, 지원 범위 밖 torch, session radix cache 등이다. 이 체크아웃에는 빌드된 Rust `.so`가 없다. 하지만 cargo가 있으면 source build를 시도하므로(`SGLANG_RUST_BUILD_MODE="auto"`, `environ.py:1905`) env를 **명시적으로 python으로 고정**해야 한다.
- TP 정합성: `match_prefix`, `insert`, `evict`에 `@rank_consensus` decorator가 붙어 있다(`unified_radix_cache.py:576-579,614-617,641,645`). poison이나 epoch 결정은 모든 rank에서 같아야 한다.

### 2.2 `UnifiedTreeNode` 필드 (`unified_tree_core.py:112-157`)

| 필드 | 의미 |
|---|---|
| `children: dict` | key는 `RadixKey.child_key(page_size)`, 즉 첫 token(page_size=1) 또는 첫 page tuple이다(`radix_cache.py:222-247`). `extra_key`와 `cache_salt`가 있으면 tuple로 namespace가 붙는다 |
| `parent`, `key: RadixKey` | `RadixKey.token_ids`는 `array('q')`이고 CPU에 있다(`radix_cache.py:59-82`) |
| `component_data[ct]: ComponentData` | `value`(device index tensor), `lock_ref`, `host_value`, `host_lock_ref`, `session_ref`, `metadata`(`components/base.py:68-75`) |
| `last_access_time`, `creation_time` | 논리 카운터 `get_and_increase_time_counter()` |
| `hash_value`, `event_hash_value` | token-id 기반 SHA-256 page hash. storage나 linker가 켜졌을 때만 계산된다(`unified_tree_core.py:1534-1535`) |
| `hit_count`, `priority`, `id`(=NodeId) | `id`는 전역 counter. 외부에는 NodeId(int)만 노출된다(`unified_tree_core_interface.py:17`) |
| `write_through_pending_id`, `load_back_pending_id`, `rotation_base`, `lru_prev/next`, `_tlru_*` | HiCache, sharding, LRU 관련 |
| `evicted` property | `parent is not None and FULL.value is None`(:167-172). root는 evicted로 보지 않는다 |

- `node.value` 위치: `checkpoint_into_tree`에서 `self.req_to_token_pool.req_to_token[req_pool_idx, :len]`(GPU int32)를 `.to(torch.int64, copy=True)`로 복사한다(`unified_radix_cache.py:1094-1096,1150`). insert 시 `value.clone()`(`unified_tree_core.py:1529`), split 시 `child_cd.value[:split_len].clone()`(`components/full.py:160-162`)로 다시 복사된다. 따라서 **노드마다 별도 GPU tensor**를 갖고, `req_to_token`과 메모리를 공유하지 않는다. cross-recovery 원천으로 쓸 수 있다.
- root: `reset()`에서 `_new_node()`를 만들고 `FULL.value = []`(Python list), 모든 component의 `lock_ref = 1`로 둔다(`unified_tree_core.py:497-503`). **root는 KV를 갖지 않는다.** 단일 root가 모든 namespace를 처리한다(`root_node_handle`, `:664-666`).
- 노드 arena: `self._node_arena: dict[NodeId, UnifiedTreeNode]`(`:490`), `node_by_id`(`:543`). fault injector는 lock_ref와 descendant 기준으로 노드를 찾을 때 이 arena를 순회하면 된다. first-level 노드는 `tree_core.root_node.children.values()`다.

### 2.3 주요 연산

**match_prefix**: `UnifiedRadixCache.match_prefix`(`unified_radix_cache.py:580`) → `_match_tree`(`:586`) → `UnifiedTreeCore.match_prefix`(`unified_tree_core.py:867`) → `_match_prefix_helper`(`:894-982`).
- 루프는 `:951-975`에 있다. 노드 중간에서 갈라지면 `_split_node`를 호출한다(`:960`, **match도 트리 구조를 바꾼다**).
- `_update_best_if_valid`(`:935-949`)는 component validator(`FullComponent.create_match_validator`, `full.py:107-118`)를 통과한 노드만 best로 갱신한다. 그런데 **validator가 False여도 루프는 계속 내려간다.** 그래서 poisoned 노드를 validator에서만 거르면 그 아래 descendant가 다시 best가 될 수 있다. **"poisoned 앞에서 match 중단"은 루프 안 `:955`의 `if child.evicted and not child.backuped: break` 옆에 `if child.poisoned: break`로 넣어야 한다.**
- 후처리 `_match_post_processor`(`:1010-1062`)는 경로 전체의 `last_access_time`을 갱신하고, `MatchResult`를 NodeId로 바꿔서 반환한다.
- 호출부: `match_kv_cache`(`srt/mem_cache/common.py:203-251`). 이 함수는 `schedule_policy.py:225,352`, `schedule_batch.py:1667`에서 불린다. 대기 중인 요청은 `touch_prefix`도 호출한다(`common.py:197`).

**insert**: `UnifiedRadixCache.insert`(`unified_radix_cache.py:618-640`)가 resumable walk를 돌린다: `begin_insert`(`unified_tree_core.py:1124`) → `_insert_walk_step`(`:1275-1358`) → `_insert_commit_step`(`:1360-1398`) → `_insert_tail_step`(`:1428`).
- 기존 노드와 겹치는 구간에서 요청 쪽 중복 slot은 `FreeDeviceKV` action으로 해제된다(`:1338-1352`). 실제 해제는 `_apply_cache_action`이 `free_segment`를 호출해서 한다(`unified_radix_cache.py:1320-1323`).
- 새 leaf는 `_add_new_node`(`unified_tree_core.py:1510-1540`)가 만든다. **epoch 기록(`born_epoch`)과 hash digest 등록 위치로 적당하다.** 부모가 root면 first-level 노드이므로 Module D 사본 생성 trigger 지점이 된다.
- evicted 노드에 다시 insert하면 `_unevict_node_on_insert`(`:1542-1568`)로 value가 새 slot으로 복원된다. epoch와 digest를 리셋해야 한다.

**checkpoint (기존 cache_(un)finished_req)**: `UnifiedRadixCache.checkpoint_into_tree`(`unified_radix_cache.py:1080-1254`).
1. `req_to_token` row를 읽는다(:1094).
2. `insert`(:1154). prompt 경계로 split하기 위한 두 번째 insert가 있다(:1173-1195).
3. `_match_tree`로 새 경로를 다시 찾는다(:1203).
4. `req_to_token_pool.write`로 row를 **tree의 slot으로 다시 가리키게 한다**(:1213-1216). 중복 slot을 해제했기 때문이다.
5. lock을 옮긴다: `unlock(req.lock)`, `req.lock = self.lock(new_last_node)`(:1218-1233).
6. `req.kv.cache_protected_len`과 `req.last_node`를 갱신한다(:1236-1237).
- 호출 경로:
  - 실행 중 요청(prefill 완료, chunk): `checkpoint_kv_cache`(`common.py:253-262`).
  - 종료한 요청: `release_kv_cache(checkpoint=True)`(`common.py:378-417`). checkpoint 뒤 `free_kv_row`로 `[cache_protected_len, owned_kv_len)`를 해제하고 `unlock`한 다음 `req_to_token_pool.free(req)`를 호출한다.
  - retract나 abort는 `checkpoint=False`로 insert 없이 해제한다.

**evict**: `UnifiedRadixCache.evict`(`:642`) → `_evict`(`:741`) → `_evict_components`(`:872-974`) → `tree_core.evict_device_start` / `evict_device_next_node` / `evict_device_leaf`.
- victim은 `FullComponent._evict_device_next_node`(`full.py:215-239`)가 heap에서 고른다. 대상은 `evictable_device_leaves`이고, 조건은 lock_ref==0이고 device에 child가 없는 leaf다(`_is_device_leaf`, `unified_tree_core.py:2266`).
- backup이 없는 leaf는 `_delete_unbacked_device_leaf`(`:1867-1883`)가 처리한다. value 해제는 `_free_values` → `FreeDeviceKV`(`unified_radix_cache.py:774-784,1320`) 순서로 일어난다.
- 서브트리를 통째로 지우는 기존 함수 `drop_subtree_no_host`(`unified_tree_core.py:1793-1842`)는 lock이 걸린 노드가 있으면 거절한다. "lock_ref==0이 되면 poisoned 서브트리 해제" 구현에 재사용할 수 있다.
- allocator가 부족할 때 eviction은 `alloc_token_slots` → `evict_from_tree_cache` → `allocator.evict_to_free_tokens`로 이어진다(`srt/mem_cache/allocation.py:152-158`, `common.py:265-270`).

**inc/dec_lock_ref**: `UnifiedRadixCache.inc_lock_ref`/`dec_lock_ref`(`:993-1008`) → `UnifiedTreeCore.inc_lock_ref`(`unified_tree_core.py:713-727`) → `FullComponent.acquire_component_lock`(`full.py:274-321`). leaf에서 root까지 올라가며 `cd.lock_ref += 1`을 하고 evictable/protected size를 옮긴다. 해제는 `release_component_lock`(`full.py:323-370`)이 한다.
- 요청의 lock은 `TreeLock` receipt(`base_prefix_cache.py:243`)로 관리한다. 획득은 `schedule_policy.py:1591`(prefill admission) 등에서 한다.
- 노드를 공유하는 요청 수 = 해당 노드의 `FULL.lock_ref`다. root에서 노드까지 lock이 합산되므로 상위 노드일수록 크다.

### 2.4 RadixMend 필드와 hook 추가 위치

| 기능 | 위치 |
|---|---|
| `poisoned: bool`, `born_epoch`, `clean_epoch`, `last_scrub_ts`, digest 목록 | `UnifiedTreeNode.__init__`(`unified_tree_core.py:115-157`) |
| split 시 상속 | `_split_node`(`:1446-1508`). new_node가 child의 prefix를 받으므로 poisoned와 epoch를 복사하고, page digest는 `split_node_hash_value`(`mem_cache/utils.py:217`)처럼 리스트를 분할한다 |
| match 중단 | `_match_prefix_helper` 루프 `:951-975`(위 설명). `match_full_device_prefix`(`:984-1008`)에도 같은 처리가 필요하다 |
| insert 시 poisoned 노드 재사용 금지 | `_insert_walk_step`(`:1275`). poisoned 노드와 겹치면 기존 slot을 adopt하지 말고 분기해야 한다. 가장 단순한 방법은 poison 즉시 노드를 tree에서 분리(detach)해서 이후 match와 insert에 보이지 않게 하는 것이다 |
| lock_ref==0이 되면 해제 | `FullComponent.release_component_lock`(`full.py:362-369`)에서 `cd.lock_ref == 0`이 되는 순간 |
| 신규 노드 epoch, digest, Module D trigger | `_add_new_node`(`:1510-1540`), `_unevict_node_on_insert`(`:1542`) |
| hit-time hash 검증 | `match_kv_cache` 반환 직후(`common.py:226-251`) 또는 lock 획득 직전(`schedule_policy.py:1591`). 검증할 범위는 `prefix_device_indices(req)`(`unified_radix_cache.py:3816-3824`)다 |
| digest 계산 시점 | `checkpoint_into_tree`의 insert 직후(`unified_radix_cache.py:1154-1160`). page/block이 완성되는 시점이다. GPU 계산 primitive로 `kernels/ops/memory/adler32.py:97`(`adler32_strided_checksum`, PD checksum이 쓰는 것)이나 `kernels/ops/memory/gpu_tensor_hash.py:141`을 재사용할 수 있다. SHA-256 자체는 GPU kernel이 없다 |
| cold-node scrub 대상 열거 | `evictable_device_leaves`(`:515`) 또는 `walk_for_kv_canary(unlocked_only=True)`(`:3293-3336`). 후자는 slot, position, prev-slot을 flatten하는 기존 walker다 |

참고로 기존 `hash_value`는 storage 키용 **token-id hash**다. `compute_node_hash_values` → `get_storage_hash_str(node.key, parent_hash)`(`mem_cache/utils.py:166-176`). KV 내용은 들어가지 않으므로 baseline 용도로 쓸 수 없다. 별도 필드를 추가해야 한다.

---

## 3. KV pool (`MHATokenToKVPool`)

### 3.1 버퍼 레이아웃
- 생성자: `memory_pool.py:2201-2324`. `head_num`은 TP와 DCP 적용 뒤 rank-local kv head 수다(`kv_cache_configurator.py:2462-2464`).
- `_kv_buffer_shapes`(`memory_pool.py:2518-2529`):
  - NHD(기본): `k_buffer[l]` = `(size + page_size, head_num, head_dim)`, `v_buffer[l]`도 같은 형태다. **layer마다 별도 tensor**(list)이고 `torch.zeros`로 할당한다(`:2573-2582`).
  - HND(`SGLANG_USE_HND_KVCACHE`): `(num_pages, head_num, page_size, head_dim)`.
  - vectorized_5d: ROCm/AITER 전용(`:2541-2572`).
- dtype: `store_dtype`. bf16이면 bf16 그대로 저장하고 fp8은 uint8로 저장한다(`:2052-2056`).
- padding: 첫 행(page)이 **slot 0 = dummy/padding 쓰기 전용**이다(`:2536` 주석). allocator는 slot 1부터 배분한다(`allocator/token.py:43-46`). store kernel은 `reserved_skip_index=0`인 쓰기를 건너뛴다(`kernels/ops/kvcache/kvcache.py:79,104-105`).
- (token, kv_head) 벡터는 `k_buffer[l][slot, h, :]`로 contiguous 256B(bf16×128)다. 한 slot의 한 layer row는 `head_num*256B`이고, Llama-8B TP1이면 2048B다.
- `k_data_ptrs`, `v_data_ptrs`, `data_ptrs`, `data_strides`(`:2496-2516`): layer별 포인터 테이블. 전 layer copy kernel과 PD 전송이 이 테이블을 쓴다.

### 3.2 store 경로 (encode-at-store 융합 지점)
```
TritonAttnBackend.forward_extend/forward_decode   (srt/layers/attention/triton_backend.py:1647-1675, 2236-2263)
  └ _set_kv_buffer                                (triton_backend.py:1563-1598)
     └ MHATokenToKVPool.set_kv_buffer             (memory_pool.py:2825-2919)
        └ _store_kv_layer                         (memory_pool.py:2921-2968)
           └ _set_kv_buffer_impl                  (memory_pool.py:157-217)
              └ store_cache (JIT CUDA)            (kernels/ops/kvcache/kvcache.py:70-118)
                 └ store_kvcache_kernel           (kernels/jit/csrc/elementwise/kvcache.cuh:44-82)
```
- `store_cache`는 `can_use_store_cache(row_bytes, v_row_bytes)`일 때 쓰이고(`memory_pool.py:173-183`), CUDA/HIP 기본 경로다. 그렇지 않으면 `k_cache[indices] = k`로 폴백한다(`:206-217`).
- kernel 구조: warp 하나가 row 하나의 일부(`kNumSplit` 분할)를 맡는다. load → `assert(index >= 0 && index < size_limit)` → slot 0이 아니면 store(`kvcache.cuh:63-81`). row는 전 head를 이어 붙인 것(`H*D*2` bytes)이다. split 경계가 head 경계(256B)와 맞도록 `num_split`을 정해야 head별 checkword를 warp 안에서 끝낼 수 있다.
- **융합 방식 두 가지**
  - (a) `kvcache.cuh`에 checkword 출력 포인터(`int16 [rows, H]`)를 추가한다. JIT 템플릿이라 `_jit_kvcache_module` 인자(`kvcache.py:25-48`)를 늘리면 된다.
  - (b) `_set_kv_buffer_impl`의 CUDA 분기를 RadixMend Triton store+encode kernel로 바꾼다. 같은 함수에 이미 Triton masked store(`masked_set_kv_buffer_kernel`, DCP용, `memory_pool.py:2872-2891`)가 있으므로 형식 참고가 된다.
  - 프로토타입은 (b)가 빠르고 최종 오버헤드 측정은 (a)가 적절하다.
- 다른 KV writer(모두 checkword를 갱신하거나 재인코딩해야 함)
  - `move_kv_cache`(`memory_pool.py:3357-3427`, `copy_all_layer_kv_cache_func`). speculative decoding(`srt/speculative/spec_utils.py:848`)과 unified pool compaction이 쓴다.
  - `load_cpu_copy`(`:2747`). retraction cpu_tensor 백업 복원 경로다.
  - HiCache load-back(`srt/mem_cache/pool_host/mha.py:329-` `transfer_kv_per_layer*`).
  - PD 수신(transfer engine이 `get_contiguous_buf_infos` 포인터로 RDMA write, `memory_pool.py:2709-2723`).
  - `fused_qk_norm_mrope_3d_cache_*`(ROCm), `fused_set_kv_buffer`(일부 모델).
- OOB 진단: `maybe_detect_oob`(`memory_pool.py:2836`, `srt/utils/async_probe.py:131-143`, `SGLANG_ENABLE_ASYNC_ASSERT=1`).

### 3.3 checkword 버퍼 할당 위치
- `_create_buffers_normal`(`memory_pool.py:2531-2582`) 끝에서 layer별 `k_check[l]`, `v_check[l]`를 할당한다. 형태는 `torch.zeros((size+page_size, head_num), int16)`이다. zeros로 두면 **0 벡터의 checkword도 0이 되어(선형 코드)** 초기 상태와 slot 0이 일관된다.
- 할당은 `memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE)` 안에서 해야 한다(`:2532`). `_init_data_ptrs_and_strides`(`:2496`)처럼 포인터 테이블도 만들면 decode kernel에 layer별로 넘기기 쉽다.
- 함께 고쳐야 할 곳
  - `get_kv_size_bytes`(`:2687-2701`): 로그와 메모리 계정.
  - `_build_kv_buffer_descs`(`:2591`) / `_pd_registerable_tensors`(`:2704`): PD를 지원하려는 경우.
  - `_clear_buffers`(`:2662`).
  - `get_cpu_copy`/`load_cpu_copy`(`:2725-2768`).
- 오버헤드 계산: token당 `2(K,V) × head_num × layers × 2B`. Llama-8B는 1024 B/token으로 KV(131072 B)의 0.78%, Qwen3-8B는 1152 B/token이다. 비율은 항상 2/256이다.

### 3.4 메모리 산정 (max tokens profile)
- 흐름: `KVCacheConfigurator._resolve_memory_pool_config`(`kv_cache_configurator.py:2855-2870`) → `_profile_available_bytes`(`:2671`) → `config_from_budget`(`:2872-2900`) → `DefaultPoolConfigurator.calculate_pool_sizes`(`srt/model_executor/pool_configurator.py:614-624`). 여기서 `max_total_num_tokens = available_bytes // self._cell_size`가 계산된다.
- MHA `cell_size`: `_compute_cell_size`(`pool_configurator.py:334-509`)의 일반 분기 `:467-471`, `n * (head_dim + v_head_dim) * layers * kv_size`. **checkword 바이트(`n * 2 * layers * 2`)를 여기에 더해야** pool 크기와 실제 할당이 맞는다. 맞추지 않으면 OOM이 나거나 비교 실험의 용량이 달라진다.
- 실험에서 용량을 고정하려면 `--max-total-tokens`를 쓴다(`_apply_token_constraints`, `kv_cache_configurator.py:2892`).

---

## 4. ReqToTokenPool, allocator, seq_lens, out_cache_loc

### 4.1 `ReqToTokenPool` (`memory_pool.py:374-500`)
- `req_to_token = torch.zeros((size+1, max_context_len), dtype=torch.int32, device)`(`:403-406`). **행 0은 padding row**로, CUDA graph padding batch가 `req_pool_indices=0`을 쓴다(`:399-401`). 미사용 칸은 0이거나 이전 요청 값이 남아 있다.
- row free list는 Python list `free_slots`(`:408`)이고, `alloc_rows`/`free_rows`(`:435-458`)가 관리한다. `req_generation`은 CPU int64 tensor다(`:409`).
- 쓰기 경로(Module B encode 지점)
  - extend: `write_cache_indices`(`allocation.py:56-105`). Triton backend 계열이면 `write_req_to_token_pool_triton`(`allocation.py:79-88`)을 쓴다. prefix(`node.value`)와 새 `out_cache_loc`를 함께 쓴다.
  - decode: `alloc_for_decode_default`(`allocation.py:656-658`) `req_to_token_pool.write((req_pool_indices, locs), out_cache_loc.to(torch.int32))`.
  - checkpoint 뒤 tree slot으로 재지정: `unified_radix_cache.py:1213-1216`.
  - speculative: `assign_req_to_token_pool`(`allocation.py:685-719`).
- 읽기 경로(decode 지점)
  - `create_flashinfer_kv_indices_triton`(`kernels/ops/kvcache/kv_indices.py`), 호출은 `KVIndexTranslator.pack_read_stream`(`srt/mem_cache/kv_index_translator.py:458-518`, launch `:507-517`).
  - FA3는 `req_to_token[req_pool_indices, :max_len]`을 그대로 page_table로 쓴다(`flashattention_backend.py:878,892,912,955,1030,1141`). CUDA graph에서는 `copy_page_table`(`:764-768`)을 쓴다. **FA3에서는 decode용 복사 kernel이 따로 없어 high bit가 kernel까지 그대로 전달된다.**
  - 그 밖의 읽기: `get_last_loc`(page>1, `allocation.py:108-148`), `checkpoint_into_tree`(`unified_radix_cache.py:1094`), `free_kv_row`(`base_prefix_cache.py:487-503`), PD checksum(`disaggregation/checksum.py:38-43`), kv_canary.

### 4.2 Token allocator
- `TokenToKVPoolAllocator`(`allocator/token.py:28-96`)
  - `free_pages = torch.arange(1, size+1, int64, device)`(`:44-46`). **GPU tensor이고 slot 0은 제외된다.**
  - `release_pages`는 PD 모드(`need_sort`)에서만 쓴다(`:47`, `kv_cache_configurator.py:2488`).
  - `alloc(n)`: `free_pages[:n]` 슬라이스이고 kernel이 없다(`:54-63`).
  - `free(idx)`: `torch.cat((free_pages, idx))`(`:65-75`). `free_group` 모드에서는 `free_group_end`까지 미뤘다가 한 번에 처리한다(`allocator/base.py:202-214`). `process_batch_result_*`가 이 group으로 감싼다(`batch_result_processor.py:264,509`).
  - `free_segment`(`base.py:273-282`) → `free`.
- `PagedTokenToKVPoolAllocator`(`allocator/paged.py`): `free_pages`가 page id다(`:395-399`). `alloc_extend`(`:204-277`)와 `alloc_decode`(`:279-326`)는 Triton kernel로 `out_cache_loc`를 만들고, free는 `:328-380`이다.
- 슬롯 획득 경로: `alloc_for_extend`(`allocation.py:367-481`) → `alloc_token_slots`(`:152-172`) 또는 `alloc_paged_token_slots_extend`(`:175-248`). decode는 `alloc_for_decode_default`(`:618-681`).
- 해제 경로
  - 요청 종료와 retract: `release_kv_cache` → `free_kv_row` → `free_kv_row_segments`(`common.py:134`) → `allocator.free_segment`.
  - over-allocation: `_release_overallocated_kv_indices`(`common.py:425-454`).
  - tree evict: `FreeDeviceKV`(`unified_radix_cache.py:1320-1323`).
  - insert 중복 해제(`unified_tree_core.py:1350-1352`).
- **quarantine 구현 위치**: `free`/`free_page_ids` 진입부에서 quarantine set에 속한 slot을 걸러낸다. 이 경우 메모리 누수 검사 `InvariantChecker._check_full_pool`(`srt/managers/scheduler_components/invariant_checker.py:97-`)의 보존식(available + evictable + protected = total)이 깨지므로 quarantine 크기를 항에 추가해야 한다.

### 4.3 `seq_lens`, `out_cache_loc` 생성 (`schedule_batch.py`)
- extend: `prepare_for_extend`(`schedule_batch.py:2730`)
  - `seq_lens_tensor = torch.tensor(seq_lens, int64).to(device)`(`:2759`), `seq_lens_cpu`(int64, `:2762`), `orig_seq_lens`(int32, `:2763`).
  - `out_cache_loc, ... = alloc_for_extend(self)`(`:2795`), `self.out_cache_loc = out_cache_loc`(`:2948`).
- decode: `prepare_for_decode`(`:3563`)
  - `self.out_cache_loc = alloc_for_decode(self, token_per_req=1)`(`:3605`).
  - `self.seq_lens = self.seq_lens + 1`(`:3612`). **산술 갱신**이고 CPU 미러도 함께 갱신된다.
- `ForwardBatch`는 `batch.seq_lens`를 그대로 받는다(`srt/model_executor/forward_batch_info.py:1015-1019`). overlap에서는 next-iter seq_lens를 `FutureMap.new_seq_lens_buf`로 relay한다(`overlap_utils.py:602-612`).
- CUDA graph 정적 버퍼(`srt/model_executor/runner/base_runner.py:95-99`)
  - `req_pool_indices` zeros int64.
  - `seq_lens`는 `seq_len_fill_value`로 채운 int64(Triton은 1, `triton_backend.py:1371`).
  - `out_cache_loc`는 zeros이고 dtype은 `cache_loc_dtype`.
  - padding lane은 req row 0, slot 0, seq_len 1을 가리킨다.

### 4.4 Module B 인코딩에 영향을 주는 sentinel과 dtype
| 값 | 위치 | 영향 |
|---|---|---|
| slot 0 (padding) | `token.py:44`, `kvcache.py:79`, `base_runner.py:99` | 선형 SECDED에서 0의 codeword는 0이다. zeros 버퍼, padding lane, `req_to_token` 미사용 칸이 모두 그대로 유효하다 |
| `-1` | `get_last_loc_torch`(`allocation.py:145`), paged `last_loc`(`:429`), `token_mapping`(`kv_indices.py` docstring, "preserving -1"), FutureMap debug poison(`overlap_utils.py:280-286`), mamba track(`schedule_batch.py:2239`) | all-ones라 sign bit가 켜진다. `radixmend/secded_index.py` 규약(sign bit 0)에서는 "sentinel"로 구분할 수 있다. decode 함수는 `v < 0`을 그대로 통과시켜야 한다 |
| int32 `req_to_token` | `memory_pool.py:405` | 31bit 안에 data와 check를 넣어야 한다. `secded_index.py` 기준 최대 25 data bit다. Llama-8B를 A6000(48GB)에 올리면 pool이 대략 2×10^5 slot으로 18 bit라 여유가 충분하다 |
| int64↔int32 cast | `allocation.py:657`(`.to(int32)`), `unified_radix_cache.py:1150`(`.to(int64)`) | 인코딩된 워드가 31 bit 이하면 cast 후에도 값이 유지된다. **같은 코드를 `out_cache_loc`, `req_to_token`, `node.value`, `free_pages`에 공통으로 쓰면 구조 사이를 오갈 때 재인코딩이 필요 없다.** decode는 "소비 지점"에서만 하면 된다: store kernel, `kv_indices`/page_table 생성, `// page_size` 산술, torch 인덱싱 |
| store kernel assert | `kvcache.cuh:75` | decode를 빠뜨린 경로에서 high bit가 남아 있으면 범위 밖 index가 되어 **device assert로 프로세스가 중단된다.** 은닉 오류가 되지 않으니 실험상 안전한 편이다 |

**seq_lens 권고**: GPU `seq_lens`는 `+1`(`schedule_batch.py:3612`), `cumsum`(`triton_backend.py:518`, `flashinfer_backend.py:1750`), FutureMap publish(`scheduler.py:4581`) 등 산술 소비자가 많다. high-bit ECC를 넣으면 모든 소비자에 decode를 넣어야 한다. 대안으로 choke point(`init_forward_metadata` 진입)에서 GPU `seq_lens`와 `seq_lens_cpu`를 비교(또는 checksum)하고, 불일치하면 CPU 값으로 덮어쓰는 방식을 권한다. 단, overlap+spec 경로에서는 `seq_lens_cpu`가 비동기로 갱신된다(`overlap_utils.py:544`).

---

## 5. Attention backend (decode / extend)

### 5.1 기본 backend 선택
`get_default_attn_backend`(`srt/arg_groups/model_override_base.py:268-341`)는 `_attention_backend_default`(`srt/arg_groups/overrides.py:1066-1080`)에서 호출된다. MHA 기준 선택은 다음과 같다.
- Hopper이고 CUDA≥12.3, spec이 없거나 topk==1이면 `fa3`(`:298-302`).
- SM100이면 `trtllm_mha`(`:315`).
- 그 밖(**Ampere sm86 포함**)에서 flashinfer가 설치되어 있고 attention sink가 없으면 `flashinfer`, 아니면 `triton`(`:320-324`).
- fa3는 sm80도 허용한다(`srt/layers/attention/attention_registry.py:243-256`). 다만 A6000에서 기본으로 선택되지는 않는다.

### 5.2 Triton backend (패치 가능, 권장)
- 클래스: `TritonAttnBackend`(`srt/layers/attention/triton_backend.py:145`). kernel import는 `:162-191`이고, `decode_attention_fwd`와 `extend_attention_fwd`는 `torch.compiler.disable`로 감싸져 있다.
- **kv_indices 생성(Module B decode choke point)**: `_fill_kv_indptr_and_indices`(`:508-527`) → `kv_index_translator.pack_read_stream`(`srt/mem_cache/kv_index_translator.py:458`) → `create_flashinfer_kv_indices_triton[(bs,)]`(`:507-517`, kernel은 `kernels/ops/kvcache/kv_indices.py`).
  - decode: `init_forward_metadata` decode 분기(`triton_backend.py:845-868`, `kv_indices`는 int64 `torch.empty(seq_lens_sum)`).
  - extend: prefix 길이(`extend_prefix_lens`)만큼 만든다(`:1013-1028`). extend 토큰 자체는 indices로 읽지 않는다.
  - CUDA graph: `_update_decode_kv_buffers`(`:529-577`)가 정적 `self.cuda_graph_kv_indices`(`:1164`)에 쓴다.
  - plain pool에서는 `KVIndexTable.ids = req_to_token`, `row_ids = req_pool_indices`이고 번역은 없다(`kv_index_translator.py:449-456`).
- **decode kernel**(`kernels/ops/attention/decode_attention.py`)
  - 진입: `decode_attention_fwd`(`:1158`). GQA(group>1)는 `decode_attention_fwd_grouped`(`:1098`) → `_decode_grouped_att_m_fwd`(`:770`) → `_fwd_grouped_kernel_stage1`(`:512`)로 간다. MHA(group==1)는 `_fwd_kernel_stage1`(`:226`)을 쓴다. stage2 reduce는 `_fwd_kernel_stage2`(`:906`)다.
  - 루프 `:640-761`: `kv_loc = tl.load(kv_indices + ...)`(`:642-646`), **K tile `[BLOCK_DMODEL, BLOCK_N]` load `:658-662`**, **V tile `[BLOCK_N, BLOCK_DV]` load `:718-722`**. MHA kernel은 K `:330-334`, V `:375-379`.
  - head_dim 128 = `BLOCK_DMODEL`이므로 tile 열(K) 또는 행(V) 하나가 정확히 (token, kv_head) 벡터 하나다. **SECDED 검사와 교정을 tile load 직후 in-register로 할 수 있다.** checkword는 `kv_loc * head_num + cur_kv_head` 주소로 int16 하나를 load하면 된다.
  - grid는 `(batch, head_tiles, kv_splits)`(`:840`)이고 `head_tiles = cdiv(q_heads, min(16, group))`(`:114-116`)이다. Llama-8B(group 4)에서는 한 program이 한 kv head를 맡는다. **decode에서는 각 (token, kv_head) 벡터를 step·layer마다 정확히 한 번 load하므로 검사 중복이 없다.**
  - ROCm Lean kernel(`:1391`)은 HIP 전용이라 무시해도 된다.
- **extend kernel**(`kernels/ops/attention/extend_attention.py`)
  - `extend_attention_fwd`(`:915`) → `_fwd_kernel`(`:377`). prefix 단계에서 `kv_indices` load는 `:578-583`, **K_Buffer load `:604-608`**, **V_Buffer load `:697-701`**이다. 현재 chunk 단계는 `K_Extend`/`V_Extend`(`:791,855`)로, cache가 아닌 새로 계산한 K/V를 직접 읽는다.
  - grid는 `(batch, q_head, cdiv(max_extend, BLOCK_M))`(`:1074`)이다. 같은 prefix 벡터를 group 크기 × M-block 수만큼 **중복 load(=중복 검사)**한다. 오류 로그는 dedup(slot, layer, head 키)이 필요하고, write-back은 별도 scrub kernel로 처리하는 설계가 맞다.
  - deterministic 모드에서는 `_fwd_kernel_unified`(`:1208`, K `:1422`, V `:1510`)가 쓰이므로(`triton_backend.py:1728-1739`) 이 kernel도 패치해야 한다.
  - target-verify(spec)용 `verify_splitkv` / `verify_shared_kv`는 gfx95, MLA 전용이라 범위 밖이다.
- store: decode `:2236-2263`, extend `:1647-1675`. 두 곳 모두 attention kernel **전에** `set_kv_buffer`를 호출한다. decode의 `kv_indices`에는 방금 쓴 현재 token도 포함되므로 store→load 왕복도 검사 대상이 된다.

### 5.3 FlashInfer backend (sm86 기본, 외부 라이브러리)
- `flashinfer.BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper`를 import한다(`flashinfer_backend.py:77-83`).
- K/V를 읽는 곳: `decode_wrapper.forward(q, kv_cache, ...)`(`:1526`), `prefill_wrapper_paged.forward(_return_lse)`(`:1376,1451`). 모두 **flashinfer 패키지의 JIT/AOT CUDA kernel**이다.
- `kv_indices`는 int32이고 `pack_read_stream`으로 만든다(decode `:1761`, prefill `:2216`). Module B 측면에서는 Triton과 같은 choke point를 공유한다.
- ragged 경로에서는 KV를 attention **후에** 저장한다(`:1464-1477`).
- RadixMend 관점: in-kernel 검사를 하려면 flashinfer를 fork하고 JIT 소스를 수정해야 한다. 어렵다.

### 5.4 FlashAttention v3 backend (Hopper 기본, 외부 라이브러리)
- `sgl_kernel.flash_attn.flash_attn_with_kvcache` / `flash_attn_varlen_func`(`flashattention_backend.py:281-286`; wrapper는 `kernels/ops/attention/flash_attention_v3.py:72`). 컴파일된 CUDA(CUTLASS) kernel이다.
- `page_table`은 `req_to_token` row를 그대로 쓴다(4.1 참조). CUDA graph에서는 `copy_page_table`을 쓴다. Module B decode를 넣으려면 page_table 생성 경로마다 decode 복사를 추가해야 한다.
- in-kernel 검사는 사실상 불가능하다. 바꾸려면 sgl-kernel을 재빌드해야 한다.

### 5.5 권고: **Triton backend를 1차 대상으로 한다**
1. decode, extend, kv_indices kernel이 모두 in-repo Python/Triton이라 tile load 직후에 코드를 넣을 수 있다(5.2의 줄 번호).
2. decode grouped kernel은 벡터당 1회 load하므로 검사 오버헤드 측정이 깔끔하다.
3. sm86과 Hopper에서 모두 돌아가므로 이식성 문제가 없다. bf16/head_dim 128 경로가 이미 기본 지원된다.
4. FlashInfer와 `kv_indices` choke point를 공유하므로 Module B 결과를 그대로 옮길 수 있다.
5. 대가: sm86에서는 flashinfer보다 느릴 수 있다. baseline은 "Triton 무패치"로 잡고, 필요하면 "flashinfer 무패치"를 참고 수치로 함께 보고한다.

---

## 6. Scheduler 경로 (Module C)

### 6.1 event loop
- non-overlap: `event_loop_normal`(`srt/managers/scheduler.py:1940-1972`). `get_next_batch_to_run` → `run_batch` → `process_batch_result` 순서로 동기 실행한다.
- overlap: `event_loop_overlap`(`:1974-2051`). 반복 i에서 다음 순서로 실행된다.
  1. `get_next_batch_to_run`(:1997). batch i-1의 chunked req stash를 포함한다.
  2. `run_batch(batch_i)`(:2023). GPU에 enqueue한다.
  3. `result_queue.append((batch.copy(), result))`(:2026).
  4. `pop_and_process()`로 batch i-1의 결과를 처리한다(:2032-2034).
  5. `launch_batch_sample_if_needed`(:2042).
  - 연속 prefill의 overlap은 `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`로만 끈다(`:2053-2087`).

### 6.2 next_token_ids GPU→CPU 복사
- `GenerationBatchResult.copy_to_cpu`(`srt/managers/utils.py:162-226`): `self.next_token_ids = _async_d2h(self.next_token_ids)`(`:196`), 이어서 `copy_done.record()`(`:226`)가 실행된다. **이 함수에 error-flag tensor의 `_async_d2h`를 추가하면 같은 이벤트에 piggyback할 수 있다.** 다른 방법은 `next_token_ids`에 음수 sentinel을 덮어쓰는 것이다(token id는 항상 ≥0).
- overlap(non-spec): `run_batch`에서 `copy_stream`으로 `copy_to_cpu`를 실행한다(`scheduler.py:4594-4618`). delay-sample(structured output)일 때는 `launch_batch_sample_if_needed`가 실행한다(`:4883-4919`).
- non-overlap(non-spec): `copy_to_cpu`가 없다. `process_batch_result_*`에서 `next_token_ids.tolist()`로 동기 D2H한다(`batch_result_processor.py:293`, decode는 `_normalize_decode_outputs` 경유 `:775/894`).
- 소비: `copy_done.synchronize()`(`batch_result_processor.py:267-268`, `:947-948`).

### 6.3 다음 step 입력 token의 출처 **[설계 가정과 다름]**
- `_relay_forward_payload`(`scheduler.py:4830-4855`)가 `future_map.stash(req_pool_indices, RelayPayload(bonus_tokens=next_token_ids))`를 실행한다. 다음 iteration의 `resolve_forward_inputs`(`overlap_utils.py:87-123`)는 `output_tokens_buf[req_pool_indices]`에서 input_ids를 gather한다. **이 동작은 non-overlap에서도 같다**(`scheduler.py:4740-4750`, "Non-spec: relay via future_map").
- 따라서 오염된 step의 sampled token을 버리려면 (a) `output_tokens_buf[req_pool_idx]`를 무효화하거나 재설정하고, (b) CPU의 `req.output_ids` append를 막아야 한다(`batch_result_processor.py:1032`, prefill은 `:364`).
- overlap에서는 flag를 CPU에서 읽는 시점(`process(i-1)`)에 step i가 이미 `output_tokens_buf`를 소비했고 KV도 썼다. 설계대로 "다음 step까지 폐기"가 필요하다.

### 6.4 tree insert 시점과 flag 처리 순서
| 경로 | insert 호출 | overlap에서 flag 처리와의 순서 |
|---|---|---|
| prefill 완료 (미종료 요청) | `checkpoint_kv_cache`(`batch_result_processor.py:393-394`) | `process(i-1)` 안에서 실행되므로 **flag 확인 후 insert하도록 고칠 수 있다** |
| chunked prefill 중간 chunk | `get_next_batch_to_run` → `stash_chunked_request`(`scheduler.py:3786-3793`, `:3644-3648`) | **flag 처리 전에 insert된다(위반).** stash를 `process_batch_result_prefill`의 chunked 분기(`batch_result_processor.py:434-450`)로 옮기거나, 직전 chunk의 flag가 처리될 때까지 stash를 미뤄야 한다 |
| 요청 종료 | `_handle_finish_state_updated_req` → `release_kv_cache(checkpoint=True)`(`batch_result_processor.py:1367-1456`, `:1452`), prefill 즉시 종료(`:387-391`) | process 안에서 실행된다. flag가 있으면 `checkpoint=False`로 전환한다 |
| PD prefill | `release_kv_cache(checkpoint=True)`(`disaggregation/prefill.py:1185,1262`) | 범위 밖 |

### 6.5 retract와 re-queue
- decode 메모리 부족: `update_running_batch`(`scheduler.py:4335-4420`) → `batch.retract_decode()`(`schedule_batch.py:3298-3388`) → `release_req`(`schedule_batch.py:2242-2275`)로 이어진다. `release_req`는 `release_kv_cache(req, tree_cache, checkpoint=False)`로 **tree insert 없이 해제**하고, `evict_from_tree_cache` 후 `req.reset_for_retract()`(`:1955-1994`)를 호출한다. 이후 `_add_request_to_queue(req, is_retracted=True)`(`scheduler.py:4412-4413`)로 대기열에 다시 넣는다.
- 재개 시 `origin_input_ids + output_ids` 전체를 다시 prefill한다. `match_kv_cache`가 tree prefix를 재사용하므로, **poison(노드 분리)을 retract보다 먼저 해야** 오염된 prefix를 다시 쓰지 않는다.
- RadixMend가 오염 요청을 retract하려면 `retract_decode`의 메모리 조건 루프와 별개로 "지정 index 집합 retract" 함수가 필요하다. 구성은 `release_req` + `filter_batch(keep_indices)`(`schedule_batch.py:3378`)로 충분하다. 테스트용 강제 retract 훅 `TEST_RETRACT`(`scheduler.py:4355-4357`)도 참고할 수 있다.
- extend/chunked 요청 재대기: 현재 chunked 요청 abort 경로 `process_pending_chunked_abort`(`scheduler.py:3650-3700`)가 `release_kv_cache(checkpoint=False)`와 `chunked_req=None`을 수행한다. 이 경로를 본떠 abort 대신 `reset_for_retract` + re-queue로 바꾸면 된다. overlap에서는 이미 launch된 다음 chunk 결과를 `inflight_middle_chunks`로 drain해야 한다(`batch_result_processor.py:436`).
- overlap에서 retract한 요청은 다음 결과 처리에서 `req.is_retracted`로 건너뛴다(`batch_result_processor.py:1012-1017`). 따라서 "다음 step 결과 폐기"는 retract만으로 자동 처리된다.
- TP>1이면 rank마다 자기 kv head만 검사하므로 flag를 all-reduce(OR)해야 한다. 그러지 않으면 `rank_consensus`(2.1) 위반이나 deadlock이 생긴다.

---

## 7. 기타 KV 소비자와 작성자

| 대상 | 위치 | RadixMend 영향 |
|---|---|---|
| **kv_canary** (기존 KV 무결성 감시) | 설치: `srt/kv_canary/api.py:51-109`, model_runner hook은 `srt/model_executor/model_runner.py:985`. 설정은 `srt/kv_canary/config.py`. 버퍼 부착: `pool_patcher/adapters/mha.py:12-48`. perturbation: `perturb/real_kv_used.py`(사용 중 slot의 첫 byte 반전), `perturb/req_to_token.py`(`req_to_token` 포인터를 다른 slot으로 교체), `perturb/real_kv_unused_cache.py`, `perturb/slot_picker.py`. tree walker: `kv_canary/radix_cache_walker.py` → `unified_tree_core.py:3293`. kernel은 `kernels/ops/kv_canary/`와 `kernels/jit/csrc/kv_canary/` | fault injector와 scrub 열거 코드의 출발점으로 쓸 수 있다. 단 real-KV fingerprint는 **layer 0만** 읽는다(`mha.py:33-38` `pool.k_buffer[0]`). 실험 중에는 `--kv-canary none`으로 둔다(model.forward를 monkey-patch함) |
| PD disaggregation checksum | `disaggregation/checksum.py:82-117`(`adler32_strided_checksum`). prefill이 마지막 chunk에서 계산하고(`disaggregation/prefill.py:1455-1471`) decode가 수신 후 검증한다(`disaggregation/decode.py:3175,3184-3212`). flag는 `--disaggregation-enable-kv-checksum` | 요청 단위, prompt 전체 대상, 1회 검증 구조다. hash baseline의 GPU primitive로 재사용할 수 있다. PD 수신 KV는 store kernel을 거치지 않으므로 checkword를 재인코딩해야 한다 |
| HiCache | host pool `srt/mem_cache/pool_host/mha.py:125-`. device→host backup과 host→device `load_to_device_per_layer`(`:329-`)는 `sgl_kernel.kvcacheio`의 `transfer_kv_*`를 쓴다. tree 쪽은 `unified_radix_cache.py:1599-1674`(`_execute_and_commit_kv_backup`), `:1731`(`load_back`) | `k_buffer`를 직접 읽고 쓴다. checkword를 함께 전송하거나 load 후 재인코딩해야 한다. 1차 범위에서 제외 |
| speculative decoding | `move_kv_cache`(`memory_pool.py:3357`), 호출은 `speculative/spec_utils.py:848`. 위 경우 `enable_kv_cache_copy`(`kv_cache_configurator.py:2470`). draft pool은 별도 | 제외 |
| retraction CPU 백업 | `get_cpu_copy`/`load_cpu_copy`(`memory_pool.py:2725-2768`), `backup_kv_cache`(`common.py:309-338`, PD decode 전용) | 제외 |
| `forward_extend`의 `k is None` 경로 | `triton_backend.py:1628-1640`, `k_buffer[cache_loc]` 직접 인덱싱 | 표준 모델에서는 쓰이지 않는다 |
| CUDA graph | decode는 `FULL`, prefill은 `BREAKABLE`이 기본(1장). Triton 정적 버퍼는 `init_cuda_graph_state`(`triton_backend.py:1100-1214`)에 있고 `cuda_graph_kv_indices`는 `:1164`. decode buffer는 `base_runner.py:95-99`, capture는 `decode_cuda_graph_runner.py:1117-1148` | error-log 버퍼, 카운터, checkword 포인터 테이블은 **capture 전에 고정 주소로 할당**해야 한다(backend `__init__` 또는 pool). 매 step 0으로 초기화하는 것도 graph 안의 kernel로 넣거나, `copy_to_cpu` 뒤에 graph 밖에서 수행한다. kv_canary가 graph 안에서 동작하는 방식(`api.py:110-137`, pre/post ops)이 참고가 된다 |
| 메모리 누수 검사 | `invariant_checker.py:97-130`, `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY`(`scheduler.py:1970-1971`) | quarantine slot이나 Module D 사본 slot을 회계에 반영해야 한다 |

---

## 8. 패치 지점 요약 표

| 모듈/기능 | 파일:함수 | 난이도/위험 |
|---|---|---|
| 전제: Python TreeCore 강제 | env `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` (`srt/environ.py:688`) | 하 / Rust와 성능 차이가 있으니 baseline도 같은 설정으로 측정해야 한다 |
| A: checkword 버퍼 | `memory_pool.py:MHATokenToKVPool._create_buffers_normal`(2531), `get_kv_size_bytes`(2687) | 하 |
| A: 메모리 산정 | `model_executor/pool_configurator.py:DefaultPoolConfigurator._compute_cell_size`(467-471) | 하 / 빠뜨리면 OOM이 난다 |
| A: encode-at-store | `memory_pool.py:_set_kv_buffer_impl`(157) 또는 `kernels/jit/csrc/elementwise/kvcache.cuh:store_kvcache_kernel`(44) | 중 (Triton 대체) ~ 상 (CUDA 융합). warp split이 head 경계와 맞아야 한다 |
| A: decode 검사·교정 | `kernels/ops/attention/decode_attention.py:_fwd_grouped_kernel_stage1`(K 658, V 718), `_fwd_kernel_stage1`(330/375), 인자 전달 `decode_attention_fwd`(1158) → `triton_backend.py:forward_decode`(2375) | 중 / Triton 레지스터 압박, BLOCK_N 튜닝 |
| A: extend 검사·교정 | `kernels/ops/attention/extend_attention.py:_fwd_kernel`(K 604, V 697), deterministic은 `_fwd_kernel_unified`(1422/1510) | 중 / 중복 load로 로그 폭증, dedup 필요 |
| A: 오류 로그 버퍼, scrub kernel | 신규. 할당은 `TritonAttnBackend.__init__`(154) 또는 pool, write-back은 별도 kernel | 중 / CUDA graph 고정 주소 |
| A: cold node 백그라운드 scrub | `unified_tree_core.py:evictable_device_leaves`(515) / `walk_for_kv_canary`(3293), scheduler `on_idle`(`scheduler.py:5072`) | 중 / 순회는 CPU 비용, 논리 시계만 있음 |
| B: 인코딩 (공통 코드) | `allocator/token.py:clear/alloc/free`(42-75), `allocation.py:alloc_for_decode_default`(656), `write_cache_indices`(56) / `write_req_to_token_pool_triton`, `unified_radix_cache.py:checkpoint_into_tree`(1150,1213) | 중 / 모든 소비자를 찾아 decode해야 한다 |
| B: decode choke point | `kernels/ops/kvcache/kv_indices.py:create_flashinfer_kv_indices_triton`(via `kv_index_translator.py:507`), `store_kvcache_kernel`(index load `kvcache.cuh:65`), FA3 page_table(`flashattention_backend.py:764,878…`), paged `free`(`paged.py:333`) | 중~상 / FA3 경로가 많다 |
| B: seq_lens | `init_forward_metadata`(`triton_backend.py:826`) 진입 시 `seq_lens_cpu`와 대조 | 하 (ECC 대신 권장) |
| B: cross-recovery req_to_token↔node.value | `req.last_node` 경로의 `path_device_indices`(`unified_radix_cache.py:3826`) vs `req_to_token[row, :cache_protected_len]` | 중 / tree 소유 구간에서만 성립 |
| C: flag piggyback | `managers/utils.py:GenerationBatchResult.copy_to_cpu`(196), non-overlap은 `.tolist()` 지점 | 하 |
| C: token과 KV 폐기 | `batch_result_processor.py:process_batch_result_decode`(942-1100), `_prefill`(258-), `overlap_utils.py:FutureMap.output_tokens_buf` | 중 / overlap 1-step 지연 |
| C: insert 지연 | `batch_result_processor.py:394,1452`, chunked는 `scheduler.py:3793` 이동 | 중 / chunk 순서 불변식 |
| C: poison, match 중단, 해제 | `unified_tree_core.py:UnifiedTreeNode`(115), `_match_prefix_helper`(951-975), `_split_node`(1446), `full.py:release_component_lock`(362), `drop_subtree_no_host`(1793) 응용 | 중 / TP consensus, HiCache 상호작용 |
| C: 사용 요청 retract와 re-queue | `schedule_batch.py:release_req`(2242), `filter_batch`, `scheduler.py:_add_request_to_queue`, chunked는 `process_pending_chunked_abort`(3650) 변형 | 중 |
| C: slot quarantine | allocator `free` 진입부, `invariant_checker.py:_check_full_pool` | 하~중 |
| C: born_epoch / clean_epoch | `_add_new_node`(1510), `_unevict_node_on_insert`(1542), scrub 완료 시 갱신 | 하 |
| D: first-level, sink 사본 | 트리거 `_add_new_node`(parent==root), 복사 `MHATokenToKVPool.move_kv_cache`(3357, `enable_kv_cache_copy` 필요 `kv_cache_configurator.py:2470`), 해제 `_delete_unbacked_device_leaf`(1867) | 중~상 / 사본 slot 회계, 사본에서 읽는 fallback 경로 |
| Injector: KV bit flip | `pool.k_buffer[l].view(torch.int16)[slot, head, elem] ^= 1<<bit` (NHD), kv_canary `perturb/utils.py` 참고 | 하 |
| Injector: 인덱스 flip | `req_to_token`(int32), `node.value`(`component_data[FULL].value`), `allocator.free_pages`, `batch.out_cache_loc`, `batch.seq_lens` | 하 / 주입 시점(stream 동기화) 주의 |
| Injector: 노드 탐색 | `tree_core.root_node.children`(first-level), `_node_arena`(490), `component_data[FULL].lock_ref`, sink = 각 경로 첫 4 token의 slot | 하 |
| Baseline: hit-time hash | digest는 `checkpoint_into_tree` insert 직후(1154), 검증은 `match_kv_cache`(`common.py:203`) 또는 `schedule_policy.py:1591`, 저장은 `UnifiedTreeNode` 신규 필드(page 리스트, split 동기화) | 중 / 전 layer KV 읽기 비용, SHA-256 GPU 구현 없음(Adler/xxhash류로 대체하거나 CPU SHA) |

---

## 9. 1주차 결정 사항 제안

1. **버전 고정**: `v0.5.22` / `baa9befa3`로 고정하고 기간 중 업그레이드하지 않는다. 이 버전은 tree가 Unified/Rust로 재구성된 상태라 상위 버전에서도 구조가 계속 바뀔 가능성이 크다. 의존성: torch 2.14.1, flashinfer 0.7.1(cu13), sglang-kernel 0.4.9. 이 wheel들은 CUDA 13 기반이므로 A6000 서버의 드라이버가 CUDA 13을 지원하는지 **첫날 확인**해야 한다.
2. **필수 env**:
   - `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`: Python 노드 접근을 위해 필요하다. baseline과 RadixMend 모두 동일하게 적용한다.
   - 디버그 시 `SGLANG_ENABLE_ASYNC_ASSERT=1`.
3. **attention backend**: `--attention-backend triton`. prefill과 decode 모두 Triton으로 맞추고 FlashInfer는 쓰지 않는다. 근거는 5.5. 참고 수치로만 flashinfer 무패치 처리량을 1회 측정한다.
4. **page size**: `--page-size 1`(CUDA 기본값과 같지만 명시). page_size>1은 4.4와 1장의 차이점 때문에 2차 범위로 미룬다.
5. **CUDA graph**:
   - 초기 프로토타입: `--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled`. `--disable-cuda-graph`는 deprecated alias다(`srt/server_args.py:410-414`).
   - 커널이 안정되면 decode graph(`full`)를 다시 켜서 오버헤드를 측정한다. prefill은 기본이 `breakable`이므로 최종 측정은 기본값으로 한다.
6. **overlap scheduler**:
   - Module A/B 성능 측정: 기본값(overlap on).
   - Module C 초기 구현과 디버깅: `--disable-overlap-schedule`로 1-step 지연을 제거한 뒤 시작한다.
   - 이후 overlap on에서 "다음 step 폐기"와 chunked stash 이동(6.4)을 구현하고, 두 모드를 ablation으로 보고한다.
7. **기타 고정 설정**:
   - `--tp-size 1`, speculative off, `--kv-cache-dtype auto`(bf16), HiCache/PD off, `--kv-canary none`.
   - `--mem-fraction-static` 또는 `--max-total-tokens`를 고정해 baseline과 RadixMend의 pool 용량을 같게 한다(checkword 오버헤드 0.78%가 용량 차이로 섞이지 않도록).
   - 재사용 효과 ablation에는 `SGLANG_RADIX_FORCE_MISS=1`(`srt/environ.py:669`, `common.py:222-225`)을 쓴다.
8. **정확성 비교 방법**: golden 실행과 fault 실행을 비교하려면 greedy(temperature 0)를 쓴다. `--enable-deterministic-inference`는 Triton extend를 `_fwd_kernel_unified`로 바꾸므로(5.2), 쓰려면 그 kernel도 패치 대상에 넣는다. 1차에서는 쓰지 않고 batch 구성을 고정하는 방식으로 대신하는 것을 권장한다.
9. **재사용할 기존 코드**:
   - fault injector 골격: `srt/kv_canary/perturb/*`.
   - 노드 열거: `walk_for_kv_canary`.
   - GPU checksum: `kernels/ops/memory/adler32.py`.
   - 위 코드는 SGLang을 수정하지 않고 RadixMend 패키지 쪽에서 import하거나 monkey-patch하는 방식을 우선 검토한다.
