# RadixMend

SGLang radix tree KV 캐시의 비트플립을 attention 커널이 KV를 읽는 순간 교정하는 연구 코드다. 이 저장소에는 지금 GPU 없이 만들고 검증할 수 있는 부분만 들어 있다.

## 구성

| 경로 | 내용 |
|---|---|
| `radixmend/secded_vec.py` | 모듈 A: 헤드 벡터 SECDED 참조 구현 (비밀 라벨, 판정 규칙, 코드워드 분할) |
| `radixmend/labels.py` | 키 기반 비밀 라벨 순열 |
| `radixmend/secded_index.py` | 모듈 B: slot 인덱스 상위 비트 SECDED (int32/int64) |
| `radixmend/kernels/secded_triton.py` | encode / check-on-load / scrub Triton 커널 |
| `radixmend/fault_inject.py` | KV, 인덱스 결함 주입기와 radix tree 대상 선택 |
| `scripts/` | 전수 검증, 오교정률, 코드워드 길이 실험 |
| `results/` | 위 스크립트의 결과 (JSON) |
| `docs/secded_verification.md` | 검증 결과 요약과 설계 제안 |
| `docs/sglang_code_analysis.md` | SGLang v0.5.22 코드 분석과 패치 지점 |
| `docs/experiment_spec.md` | 실험 환경, 워크로드, 비교 절차 명세 |
| `docs/server_runbook.md` | GPU 서버 대여 첫날 설치·확인부터 반납까지의 절차, 알려진 문제, 기준값 |
| `scripts/server/` | 서버 설치(`setup.sh`), 첫날 확인(`day1.sh`), TPOT 측정(`bench_tpot.sh`), 작업 잇기(`after.sh`), 결과 회수(`pull_results.sh`) |

## 실행

```bash
pip install numpy torch triton pytest
python -m pytest tests                          # GPU가 없으면 커널 테스트는 건너뜀
TRITON_INTERPRET=1 python -m pytest tests       # 커널까지 CPU에서 검증
python scripts/verify_secded_exhaustive.py      # 단일/이중/삼중 오류 전수 검사
python scripts/verify_index_code.py
python scripts/triple_miscorrection_sweep.py
python scripts/codeword_length_mc.py
```
