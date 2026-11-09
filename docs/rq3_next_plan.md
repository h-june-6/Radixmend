# 다음 GPU 대여 측정 계획 (RQ3 decode 검사 경량화)

작성 2026-11-09.
- 지난 결과는 `docs/rq3_fused_results.md`에 있다.
- 서버 절차는 `docs/server_runbook.md`를 따른다.

## 왜 하는가

지난 측정 결과를 기준(보호 없음)과 비교하면 다음과 같다.

| 구성 | TPOT 증가 |
|---|---|
| 저장 시 encode | +0.3~0.7% |
| decode 검사 | +7%(c1) ~ +81%(gsp c32) |
| extend 검사 | 추가 비용 거의 없음 |

목표인 3%에 다가가려면 decode 검사를 줄여야 한다.

decode 커널은 메모리 대역폭에 묶여 있다. 검사의 정수 연산(원소마다 패리티, where, xor 리덕션)이 타일마다 K와 V에 더해지는 것이 비용이라고 본다.

## 준비한 변형

모두 `radixmend/sglang_secded.py`에 있다. 서버에서는 환경 변수로 고른다.

| 변수 | 값 | 뜻 |
|---|---|---|
| `RADIXMEND_SECDED_MODE` | `full` (기본) | 타일마다 전체 신드롬(S_e, S_b, p) |
| | `gate` | XOR 리덕션 하나로 p와 S_b를 먼저 구한다. 체크워드와 다른 벡터가 타일에 있을 때만 S_e를 계산한다. |
| `RADIXMEND_SECDED_POPC` | 비움 (GPU면 켬) / `0` / `1` | 원소 패리티를 shift-xor 4번 대신 `popc` 한 번으로 구한다. |
| `RADIXMEND_SECDED_SAMPLE` | `1` (기본) / P | decode 단계마다 타일의 1/P만 검사한다. 위상은 단계마다 돈다. |

extend(prefill) 검사는 모드와 관계없이 항상 full이고 제자리 교정한다.

### 변형별로 놓치는 것

- **gate**
  - 1비트 오류는 모두 잡는다. p가 반드시 바뀌기 때문이다.
  - 놓치는 경우는 한 벡터 안에서 같은 비트 위치가 두 원소에 걸쳐 뒤집힌 2비트 오류다. S_b가 상쇄되고 p도 그대로다.
  - 같은 원소 안의 2비트나 위치가 다른 2비트는 S_b가 바뀌어 잡힌다.
  - 테스트: `test_gate_misses_same_position_double_but_full_flags_it`.
- **sample P**
  - decode에서는 오류가 최대 P-1단계 동안 교정되지 않은 채 읽힐 수 있다. 메모리에는 고쳐 쓰지 않기 때문이다.
  - extend 검사는 그대로라 공유 프리픽스는 prefill 때 고쳐진다.
  - 출력이 바뀌는 비율로 영향을 따로 재야 한다.

### 로컬 확인

`TRITON_INTERPRET=1`에서 `tests/test_sglang_secded.py` 10개가 통과했다(fp16).

popc 경로는 libdevice가 필요해 GPU에서만 테스트된다(`test_variants_correct_single_flips[...popc=True]`).

## 대여 첫날 순서

시간은 지난 대여의 실측 기준이다.

### 1. 설치와 확인 (runbook 1~2장, 약 1시간)

`setup.sh` → `day1.sh` → `PROFILE=smoke bash scripts/server/bench_tpot.sh results/rq3/smoke nosecded check`

이 단계에서 새 테스트(popc 포함)가 GPU에서 통과해야 한다.

### 2. 커널 마이크로벤치 (약 5분)

`python scripts/bench_fused_decode.py --out results/rq3/fused_kernel.json`

- SGLang 원본 stage 1과 각 변형의 시간 비율을 비교한다.
- 결함이 없을 때 출력이 비트 단위로 같은지(`all_equal`)도 함께 확인한다.
- 이 결과로 서버 측정에 넣을 변형을 고른다.
  - 기본 후보: `full_popc`, `gate_popc`, `gate_popc_s4`.
  - `copy_nocheck`의 비율이 1에서 크게 벗어나면 복사본 커널 자체가 다른 것이다. 먼저 원인을 본다.

### 3. TPOT 측정 (구성당 약 55분, 6구성이면 약 5.5시간)

```
tmux new -d -s tpot "bash ~/radixmend/scripts/server/bench_tpot.sh results/rq3/tpot2 \
  nosecded check \
  popc:RADIXMEND_SECDED=1,RADIXMEND_SECDED_POPC=1 \
  gate:RADIXMEND_SECDED=1,RADIXMEND_SECDED_MODE=gate \
  s4:RADIXMEND_SECDED=1,RADIXMEND_SECDED_MODE=gate,RADIXMEND_SECDED_SAMPLE=4 \
  s16:RADIXMEND_SECDED=1,RADIXMEND_SECDED_MODE=gate,RADIXMEND_SECDED_SAMPLE=16"
```

- `check`는 지난번 c(full)와 같은지 보는 재현용이다. 단, 이번에는 GPU에서 popc가 기본으로 켜진다. 지난번과 같은 조건이 필요하면 `RADIXMEND_SECDED_POPC=0`을 준다.
- 마이크로벤치에서 효과가 없던 변형은 빼서 시간을 줄인다.

### 4. 교정 효과 확인 (시행당 약 6초)

변형마다 V와 K 비트 14, 7, 0을 각 30회 반복한다. 변형 4개면 약 1.2시간이다.

```
RADIXMEND_SECDED=1 RADIXMEND_SECDED_MODE=gate RADIXMEND_SECDED_SAMPLE=4 ...  # 서버 기동
python scripts/rq1_kv_sweep.py --out results/rq3/sweep_s4 --kinds v,k --bits 14,7,0 --reps 30
```

- 매 sweep 전후로 `secded_stats`(reset)를 읽는다.
- 지난번 10회로는 신뢰구간 상한이 27.8%였다. 30회로 늘리면 10% 안쪽으로 줄어든다.

### 5. 반납 전 결과 회수 (runbook 4장)

## 판단 기준

- TPOT 증가 3% 미만이면서 출력 변화 0인 변형이 있으면, 그것을 모듈 A의 기본값으로 정한다.
- 없으면 다음 후보로 넘어간다.
  - S_e 계산을 2원소씩 32비트로 묶어 연산 수를 반으로 줄인다.
  - 타일 크기와 num_warps를 다시 튜닝한다.
  - 검사를 attention과 다른 stream에서 겹쳐 돌린다(교정이 늦어지는 것을 감수).

## 남은 일 (GPU 없이 진행 가능)

- 모듈 C(계보 무효화): RQ1-6에서 부모를 고쳐도 자식이 오염된 채 남았다. 모듈 A만으로는 부족하다는 근거다.
- 모듈 B(인덱스 상위 비트 보호): RQ1-5에서 하위 비트가 조용한 변화의 주범이었다.
- MHA decode 경로와 page size > 1은 아직 범위 밖이다.
