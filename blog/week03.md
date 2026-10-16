# [RadixMend 3주차·선행] GPU 없이 Triton 커널 검증하기: encode, 읽기 시점 교정, scrub

> 3주차 계획에는 "신드롬 계산만 넣은 커널 프로토타입으로 decode·extend 경로 오버헤드 선측정"이 있다. 오버헤드 측정은 GPU가 있어야 하지만, 커널이 맞게 계산하는지는 CPU에서도 확인할 수 있다. 그래서 커널 정확성 검증을 먼저 해 두었다. 이 커널들은 6~7주차 모듈 A 구현의 뼈대가 된다.

## 이번 주 목표 (WBS)

- K·V 텐서 비트 위치별 영향 측정 → GPU 필요, 미착수
- 커널 검사 프로토타입: 신드롬 계산만 넣고 decode·extend 경로별 오버헤드 선측정 → **커널 작성과 정확성 검증 완료, 오버헤드 측정은 GPU 필요**

## 1. 왜 독립 커널부터 만들었나

최종 목표는 SGLang Triton 백엔드의 decode/extend attention 커널 안, K/V 타일을 load한 직후에 검사를 넣는 것이다. 하지만 바로 attention 커널을 고치면 attention 수학과 SECDED 수학의 버그가 섞인다. 그래서 같은 연산을 하는 독립 커널 세 개를 먼저 만들고, 2주차에 검증한 numpy 참조 구현과 비트 단위로 맞췄다.

| 커널 | 하는 일 | 나중에 들어갈 곳 |
|---|---|---|
| `encode_kernel` | 새로 쓴 K/V 행의 체크워드 계산 | KV store 커널 (encode-at-store) |
| `check_load_kernel` | slot 번호로 벡터를 gather하면서 신드롬 계산, 1비트는 레지스터에서 교정, 오류는 로그에 atomic 기록 | decode/extend attention의 타일 load 직후 |
| `scrub_kernel` | 로그의 벡터를 재검증한 뒤 메모리에 교정값 쓰기 | 별도 커널 (스텝 사이 또는 유휴 시간) |

## 2. 커널에서의 계산

원소마다 하는 일은 popcount와 XOR뿐이다. Triton 인터프리터에서도 돌아가도록 popcount 대신 XOR 접기로 패리티를 구했다.

```python
@triton.jit
def _par16(v):
    v = v ^ (v >> 8)
    v = v ^ (v >> 4)
    v = v ^ (v >> 2)
    v = v ^ (v >> 1)
    return v & 1

@triton.jit
def _data_terms(x, e_lab, bmask_ptr, B: tl.constexpr):
    par = _par16(x)                                     # 원소별 패리티
    p = tl.sum(par, axis=0) & 1                         # P
    se = tl.reduce(tl.where(par == 1, e_lab, 0), 0, _xor)   # S_e
    sb = p * 0
    for k in tl.static_range(B):                        # S_b, 비트마다 마스크 하나
        m = tl.load(bmask_ptr + k)
        sb = sb | ((tl.sum(_par16(x & m), axis=0) & 1) << k)
    return p, se, sb
```

교정은 분기 없이 한다. 판정이 "데이터 오류"이고 원소 번호가 맞는 lane만 해당 비트를 XOR한다.

```python
fix = tl.where((status == 1) & (d == elem), 1 << bit, 0)
x = x ^ fix          # attention 수학은 고친 값을 쓰고, 메모리는 그대로 둔다
```

메모리를 커널 안에서 바로 고치지 않는 이유는 기술서에 적은 대로다. 같은 벡터를 여러 CTA가 동시에 읽을 수 있고(prefix 공유, split-KV, GQA), 3비트 이상의 오교정 값이 메모리에 영구히 남으면 안 된다. 1주차 코드 분석에서 extend 커널이 같은 prefix 벡터를 여러 번 읽는다는 것을 확인했기 때문에 이 분리가 더 중요해졌다.

## 3. CPU에서 검증: `TRITON_INTERPRET=1`

Triton은 `TRITON_INTERPRET=1`이면 커널을 numpy로 해석해서 실행한다. GPU 없이 결과를 확인할 수 있다.

```bash
TRITON_INTERPRET=1 python -m pytest tests/test_triton_kernels.py
```

| 테스트 | 결과 |
|---|---|
| encode 결과 vs numpy 참조 구현 | 비트 단위 일치 |
| 한 벡터의 단일 오류 2,063개 (데이터 2048 + 체크비트 15) | 모두 교정 |
| 데이터 오류, 체크워드 오류, 이중 오류를 섞은 배치 | 각각 교정 / 체크워드만 교정 / 탐지로 판정, 메모리는 그대로 |
| check_load 후 scrub | 메모리 값과 체크워드 복원. 같은 벡터가 로그에 두 번 있어도 결과 동일. 이중 오류는 쓰지 않음 |
| scrub 후 다시 check_load | 이중 오류 벡터만 다시 로그에 남음 |

## 4. 아직 모르는 것

- **오버헤드.** 인터프리터는 느려서 속도를 재는 의미가 없다. GPU에서 decode/extend를 따로 재야 한다. 목표는 TPOT 증가 3% 미만이다.
- **attention 커널 안에서의 배치.** decode 커널은 K 타일이 `[BLOCK_DMODEL, BLOCK_N]`이라 열 하나가 벡터 하나다. 리덕션 방향이 독립 커널과 다르므로 옮길 때 다시 맞춰야 한다.
- 3%를 크게 넘으면 시도할 것: 코드워드 분할, 검사 위치 변경(타일 load 직후 대 softmax 전), 워프 셔플 리덕션.

---

코드: `radixmend/kernels/secded_triton.py`, `tests/test_triton_kernels.py`
