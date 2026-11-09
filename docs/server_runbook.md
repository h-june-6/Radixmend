# GPU 서버 운영 절차 (대여 첫날부터 반납까지)

2026-11-06~09 첫 대여에서 겪은 문제와 실측 시간을 바탕으로 정리했다. 서버는 `dongguk-gpu`
(RTX A6000 1장, 드라이버 580.178.04 / CUDA 13.0, ECC Disabled, 개인 쿼터 50GB)이다.
`/tmp`는 대여마다 초기화된다고 보고, 홈에는 코드와 결과만 둔다.

## 0. 대여 전 (PC)

```bash
ssh dongguk-gpu 'nvidia-smi --query-gpu=name,memory.used --format=csv'   # 접속 확인 (비밀번호 없이 키로 접속)
cd ~/radixmend_code && git pull --ff-only origin main
```

## 1. 첫날: 코드 복사와 설치 (약 15~40분)

```bash
cd ~/radixmend_code && git archive --format=tar HEAD | ssh dongguk-gpu 'mkdir -p ~/radixmend && tar xf - -C ~/radixmend'
ssh dongguk-gpu "tmux new -d -s setup 'bash ~/radixmend/scripts/server/setup.sh 2>&1 | tee ~/radixmend/setup.log'"
ssh dongguk-gpu 'tail -n 5 ~/radixmend/setup.log'      # 'setup done'이 나올 때까지
```

- `.gitattributes`(`eol=lf`) 덕분에 Windows에서 그냥 `git archive` 해도 LF로 간다. 예전에는 CRLF 때문에 `set: pipefail: invalid option`으로 죽었다.
- `results/`는 git에 없으므로 전체 archive를 다시 복사해도 서버 결과를 덮어쓰지 않는다.
- setup.sh가 처리하는 것: Python 3.11 venv(시스템 3.10에는 sglang 0.5.22 wheel이 없음), sglang 0.5.22, nvcc/crt/nvvm 13.0.88 고정과 `lib64`/`libcudart.so` 링크, `~/.radixmend_env`(CUDA_HOME을 venv의 cu13으로), uv 캐시는 `/tmp`, Qwen3-8B(16GB).

## 2. 첫날: 확인 (약 6분)

```bash
ssh dongguk-gpu "tmux new -d -s day1 'bash ~/radixmend/scripts/server/day1.sh'"
ssh dongguk-gpu 'tail -n 8 $(ls -td ~/radixmend/results/day1/*/ | head -1)day1.log'
```

마지막 줄이 `DAY1 PASS`여야 한다. 확인하는 것:

1. GPU, ECC, nvcc 경로 (venv의 `nvidia/cu13/bin/nvcc`여야 함. 시스템 12.2를 쓰면 flashinfer는 `--compress-mode` 오류, triton KV JIT 커널은 ICE로 실패)
2. 전체 pytest (GPU, `TRITON_INTERPRET` 없이)
3. 훅 서버의 golden ID가 `94c11b0adbc5` (프롬프트나 모델이 바뀌면 `EXPECT_GOLDEN=<새 값>`으로 돌린다)
4. 모듈 A 서버가 결함 없을 때 출력이 비트 단위로 같은지, 카운터가 0인지

## 3. 실험 실행

모든 긴 작업은 **서버 tmux 안에서** 돈다. PC가 잠들어도 계속된다(첫 대여에서 PC가 여러 번 오프라인이 됐다).
작업을 이어 붙일 때는 `scripts/server/after.sh`를 쓴다.

```bash
# 앞 세션(drv)이 끝나면 다음 명령 실행
tmux new -d -s q2 "bash ~/radixmend/scripts/server/after.sh drv -- bash scripts/server/bench_tpot.sh results/rq3/t1 nosecded check"
```

| 작업 | 명령 (서버 `~/radixmend`, `source ~/.radixmend_env` 후) | 시간 |
|---|---|---|
| 훅 서버 | `PYTHONPATH=$HOME/radixmend:$HOME/radixmend/radixmend/sglang_site RADIXMEND_CMD_DIR=/tmp/$USER/rm_cmd python -m sglang.launch_server --model-path Qwen/Qwen3-8B --attention-backend triton --page-size 1 --port 30000 --mem-fraction-static 0.85 --kv-canary none` (+ 모듈 A는 `RADIXMEND_SECDED=1`) | 기동 약 60초 |
| KV 비트 sweep | `python scripts/rq1_kv_sweep.py --out results/rq1/<이름> --kinds v --bits 0-15 --reps 50` (대조: `--kinds none --bits 0`) | 시행당 약 5.9초 |
| 인덱스 결함 | `bash scripts/server/run_supervised.sh results/rq1/index_logs -- python scripts/rq1_index_sweep.py --out results/rq1/index --reps 30` (서버를 스스로 띄움) | 크래시 1회당 약 80초 손실 |
| 파생 오염 | `python scripts/rq1_derived.py --out results/rq1/derived --kinds v --bits 14,7 --reps 30` | 시행당 약 3초 |
| 커널 단독 비용 | `PYTHONPATH=$HOME/radixmend python scripts/bench_secded_kernels.py --out results/rq3/kernel_bench.json` | 수 분 |
| TPOT/TTFT | `bash scripts/server/bench_tpot.sh results/rq3/<이름> nosecded encode check` (서버를 스스로 띄움, 아래) | 구성당 약 55분 |

### TPOT 측정 (`bench_tpot.sh`)

- 구성은 이름 또는 `이름:VAR=값,...` 이다. 구성마다 `RADIXMEND_*`를 모두 지우고 다시 정하므로 서로 섞이지 않는다.
  - `nosecded`, `encode`(`RADIXMEND_SECDED=1 RADIXMEND_SECDED_CHECK=0`), `check`(`RADIXMEND_SECDED=1`)
  - 예: `gate:RADIXMEND_SECDED=1,RADIXMEND_SECDED_MODE=gate`, `s4:RADIXMEND_SECDED=1,RADIXMEND_SECDED_MODE=gate,RADIXMEND_SECDED_SAMPLE=4`
  - 구성별로 실제 쓴 변수가 `<결과>/<구성>/config.env`에 남는다.
- `PROFILE=reduced`(기본, 구성당 약 55분) / `full`(약 2.2시간) / `smoke`(약 3분, 스크립트 확인용)
- `KERNEL_BENCH=1`이면 서버 측정 전에 `scripts/bench_fused_decode.py --out results/rq3/fused_kernel.json`을 먼저 돈다.
- 끝나면 `table.txt`에 첫 구성(보통 `nosecded`) 대비 TPOT/TTFT 변화가 정리된다. 다시 만들려면 `python scripts/server/tpot_table.py <결과 폴더>`.
- TPOT는 훅 없이(`RADIXMEND_CMD_DIR` 없음) 잰다. 비교는 같은 세션에서 다시 잰 `nosecded`와 한다.

## 4. 반납 전 (PC)

```bash
bash ~/radixmend_code/scripts/server/pull_results.sh        # ~/radixmend_results/<시각>/results
```

받은 폴더에서 시행 수(`wc -l */trials.jsonl`)를 확인한다. venv와 모델은 `/tmp`라 반납하면 사라진다.

## 알아 둘 것 (첫 대여에서 실제로 겪은 문제)

| 증상 | 원인과 대처 |
|---|---|
| 로그 없이 setup이 끝남 | `set -e`와 `[ -f x ] && source x`. 고쳐 둠 |
| `huggingface-cli` 동작 안 함 | `hf download`로 바꿈. `--exclude`는 패턴마다 따로 |
| rq1 스크립트가 HTTP 400, 공유 프리픽스 2토큰 | transformers의 `apply_chat_template`이 dict를 돌려줌. `return_dict=False`로 고쳐 둠 |
| `No module named 'radixmend'` | `python scripts/x.py`는 `scripts/`만 경로에 넣는다. `PYTHONPATH=$HOME/radixmend`를 붙이거나 스크립트에서 루트를 추가 |
| 같은 greedy 요청인데 출력이 다름 | prefix 캐시 적중 여부에 따라 수치가 달라진다. 기준선과 결함 실행은 같은 캐시 상태(같은 flow)로 돌린다. `--enable-deterministic-inference`는 일치시키지만 처리량이 약 2.5배 느리다 |
| 실행 중인 스크립트를 고쳤더니 이상 동작 | bash는 스크립트를 읽어 가며 실행한다. 돌고 있는 파일은 고치지 말고, 다음에 돌 파일만 고친다 |
| 실행 중 새 코드를 복사하면 섞임 | 서버 프로세스는 기동할 때 radixmend 모듈을 읽는다. 구성 비교 도중에는 코드를 바꾸지 말고, 새 코드는 따로 두었다가 `after.sh`로 끝난 뒤 복사한다 |
| PC가 꺼져 진행이 멈춘 것처럼 보임 | 서버 tmux는 계속 돈다. PC에서 기다리는 작업만 끊긴다. 돌아와서 서버 로그로 확인 |
| 홈 쿼터가 차 감 | uv 캐시(약 11GB)가 홈에 쌓였던 문제. 지금은 `/tmp`로 감 |

## 기준값 (2026-11-08~09 실측)

| 항목 | 값 |
|---|---|
| KV 풀 | 180,040토큰 (K, V 각 12.36GB), 로드 후 GPU 42,637 MiB, 모듈 A는 +200 MiB |
| TPOT (shared-prefix c1 / c8 / c32) | 24.1 / 27.6 / 34.6 ms (256요청 기준, c32는 1024요청이면 40.9) |
| TPOT (random c8 / c32) | 28.9 / 49.4 ms |
| 모듈 A TPOT 증가 | encode만 +0.3~0.7%, encode+검사 +7~81% (`docs/rq3_fused_results.md`) |
| golden ID (W1, 공유 프리픽스 864토큰) | `94c11b0adbc5` |
