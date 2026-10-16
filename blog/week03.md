# [RadixMend 3주차·선행] GPU 없이 Triton 커널 검증하기

> 3주차 계획에는 "신드롬 계산만 넣은 커널 프로토타입으로 decode·extend 오버헤드 선측정"이 있다. 오버헤드는 GPU가 있어야 잴 수 있지만, 커널이 계산을 맞게 하는지는 CPU에서도 확인할 수 있다. 그래서 그 부분만 먼저 했다. 여기서 만든 커널들은 6~7주차 모듈 A 구현의 뼈대가 된다.

## 이번 주에 하기로 한 것 (WBS)

- K·V 텐서 비트 위치별 영향 측정 → GPU 필요, 아직 못 함
- 커널 검사 프로토타입과 decode·extend 경로별 오버헤드 선측정 → **커널 작성과 정확성 검증은 끝, 오버헤드 측정은 GPU 필요**

## 왜 독립 커널부터 만들었나

최종 목표는 SGLang Triton 백엔드의 decode/extend attention 커널 안, K/V 타일을 읽은 바로 다음에 검사를 넣는 거다. 그런데 처음부터 attention 커널을 고치면 attention 계산 버그와 SECDED 계산 버그가 뒤섞여서 어디가 틀렸는지 찾기 어렵다. 그래서 같은 계산만 하는 커널 세 개를 따로 만들고, 2주차에 검증한 numpy 구현과 비트 단위로 맞췄다.

| 커널 | 하는 일 | 나중에 들어갈 곳 |
|---|---|---|
| `encode_kernel` | 새로 쓴 K/V 행의 체크워드 계산 | KV 저장 커널 (encode-at-store) |
| `check_load_kernel` | slot 번호로 벡터를 모아 오면서 신드롬 계산, 1비트는 레지스터에서 교정, 오류는 로그에 기록 | decode/extend attention의 타일 읽기 직후 |
| `scrub_kernel` | 로그에 남은 벡터를 다시 검사하고 메모리에 고친 값 쓰기 | 별도 커널 (스텝 사이나 쉬는 시간) |

## 커널에서 하는 계산

원소마다 하는 일은 패리티 계산과 XOR뿐이다.

```python
@triton.jit
def _par16(v):           # 16비트 값의 패리티
    v = v ^ (v >> 8)
    v = v ^ (v >> 4)
    v = v ^ (v >> 2)
    v = v ^ (v >> 1)
    return v & 1

@triton.jit
def _data_terms(x, e_lab, bmask_ptr, B: tl.constexpr):
    par = _par16(x)                                         # 원소별 패리티
    p = tl.sum(par, axis=0) & 1                             # P
    se = tl.reduce(tl.where(par == 1, e_lab, 0), 0, _xor)  # S_e
    sb = p * 0
    for k in tl.static_range(B):                            # S_b: 비트마다 마스크 하나
        m = tl.load(bmask_ptr + k)
        sb = sb | ((tl.sum(_par16(x & m), axis=0) & 1) << k)
    return p, se, sb
```

교정은 분기 없이 한다. "데이터 오류"로 판정됐고 원소 번호가 맞는 lane만 그 비트를 XOR한다.

```python
fix = tl.where((status == 1) & (d == elem), 1 << bit, 0)
x = x ^ fix    # attention 계산은 고친 값으로 하고, 메모리는 손대지 않는다
```

커널 안에서 메모리를 바로 고치지 않는 이유는 기술서에 쓴 그대로다. 같은 벡터를 여러 CTA가 동시에 읽을 수 있고(prefix 공유, split-KV, GQA), 3비트 이상 오류를 잘못 고친 값이 메모리에 박히면 안 된다. 1주차에 extend 커널이 같은 prefix 벡터를 여러 번 읽는다는 걸 확인해서 이 분리가 더 중요해졌다.

## 결과

```bash
TRITON_INTERPRET=1 python -m pytest tests/test_triton_kernels.py
```

| 테스트 | 결과 |
|---|---|
| encode 결과 vs numpy 구현 | 비트 단위로 같음 |
| 벡터 하나의 단일 오류 2,063개 (데이터 2048 + 체크비트 15) | 전부 교정 |
| 데이터 오류, 체크워드 오류, 이중 오류를 섞은 배치 | 각각 교정 / 체크워드만 교정 / 탐지. 메모리는 그대로 |
| check_load 다음 scrub | 메모리 값과 체크워드 복원. 같은 벡터가 로그에 두 번 있어도 결과 같음. 이중 오류는 안 건드림 |
| scrub 뒤에 다시 check_load | 이중 오류 벡터만 로그에 남음 |

## 트러블슈팅

### 1. GPU가 없는데 Triton 커널을 어떻게 돌리나

**문제.** Triton 커널은 보통 CUDA GPU에서만 돈다. 개발 환경엔 GPU가 없었다.

**해결.** `TRITON_INTERPRET=1`을 켜면 Triton이 커널을 numpy로 해석해서 CPU에서 실행해 준다. 엄청 느리지만 결과는 같아서 정확성 검증엔 충분하다. 테스트 파일은 GPU가 있으면 GPU에서, 없으면 인터프리터 모드일 때만 돌고 그 외엔 건너뛰게 했다. 나중에 GPU 서버에서 같은 테스트를 그대로 돌리면 된다.

### 2. popcount와 XOR 리덕션을 인터프리터에서 쓰기

**문제.** 패리티는 보통 popcount로 구하는데, Triton에서 popcount는 CUDA libdevice 함수(`popc`)를 불러야 하고, 이건 인터프리터에서 쓸 수 없다. 또 S_e는 원소 128개를 XOR로 합쳐야 하는데, `tl.sum` 같은 내장 XOR 리덕션은 없다.

**해결.** 패리티는 시프트하고 XOR하는 걸 네 번 반복하는 방식(XOR 접기)으로 바꿨다. GPU에서도 정수 연산 몇 개라 느리지 않다. XOR 리덕션은 `tl.reduce`에 XOR 결합 함수를 직접 넘겼다. 본 커널을 쓰기 전에 원소 128개를 XOR로 합치는 작은 커널부터 만들어서 numpy 결과와 같은지 먼저 확인했다.

```python
@triton.jit
def _xor(a, b):
    return a ^ b

se = tl.reduce(values, 0, _xor)
```

### 3. 같은 벡터가 로그에 여러 번 찍힌다

**문제.** attention 커널은 같은 벡터를 여러 번 읽을 수 있어서, 오류 하나가 로그에 여러 번 기록된다. scrub 커널이 로그대로 무작정 고치면 같은 비트를 두 번 뒤집어서 원래대로 되돌려 버릴 수 있다.

**해결.** scrub 커널은 로그를 그대로 믿지 않고, 메모리에서 벡터를 다시 읽어 신드롬을 새로 계산한다. 이미 고쳐진 벡터면 "정상"으로 나와서 아무것도 안 한다. 그 사이에 오류가 더 쌓여서 이제 단일 오류가 아니면 역시 쓰지 않는다. 이렇게 하면 같은 항목을 몇 번 처리해도 결과가 같다. 같은 벡터를 두 번 기록해 놓고 결과가 같은지 테스트로 확인했다.

### 4. 테스트가 틀렸던 경우

**문제.** scrub 후에 다시 읽으면 이중 오류 벡터가 로그에 두 번 남을 거라고 테스트를 썼는데 실제로는 한 번이었다.

**원인.** 커널이 아니라 테스트가 틀렸다. 그 테스트에서 이중 오류 벡터(slot 20)는 읽기 목록에 한 번만 들어 있었다. 두 번 읽은 건 단일 오류 벡터(slot 3) 쪽이었고, 그건 scrub으로 이미 고쳐져서 로그에 안 남는 게 맞다.

**해결.** 기대값을 1로 고쳤다. 커널 결과를 의심하기 전에 테스트의 전제부터 다시 보는 게 빠르다는 걸 또 배웠다.

## 아직 모르는 것

- **오버헤드.** 인터프리터는 느려서 속도를 재 봐야 의미가 없다. GPU에서 decode와 extend를 따로 재야 한다. 목표는 TPOT 증가 3% 미만이다.
- **attention 커널 안으로 옮기기.** decode 커널은 K 타일이 `[BLOCK_DMODEL, BLOCK_N]` 모양이라 열 하나가 벡터 하나다. 리덕션 방향이 독립 커널과 달라서 옮길 때 다시 맞춰야 한다.
- 3%를 크게 넘으면 코드워드 나누기, 검사 위치 옮기기(타일 읽은 직후 대신 softmax 전), 워프 셔플 리덕션을 차례로 시도해 볼 생각이다.

---

코드: `radixmend/kernels/secded_triton.py`, `tests/test_triton_kernels.py`
