#!/usr/bin/env bash
# RQ3 TPOT/TTFT 측정: 서버 구성마다 서버를 새로 띄우고 bench_serving 행렬을 돈다. tmux 안에서 실행한다.
#   bash scripts/server/bench_tpot.sh <결과 폴더> <구성>...
# 구성은 "이름" 또는 "이름:VAR=값,VAR=값" 이다. 구성마다 RADIXMEND_* 변수를 모두 지운 뒤 다시 정한다.
#   미리 정한 이름: nosecded (모듈 A 없음) | encode (RADIXMEND_SECDED=1 RADIXMEND_SECDED_CHECK=0)
#                   check (RADIXMEND_SECDED=1, 모드 기본값 full)
#   그 밖의 이름은 뒤의 변수만 쓴다. 값에는 RADIXMEND_* 변수만 받는다.
#   예: tmux new -d -s tpot "bash ~/radixmend/scripts/server/bench_tpot.sh results/rq3/tpot_$(date +%m%d) \
#         nosecded encode check \
#         gate:RADIXMEND_SECDED=1,RADIXMEND_SECDED_MODE=gate \
#         s4:RADIXMEND_SECDED=1,RADIXMEND_SECDED_MODE=gate,RADIXMEND_SECDED_SAMPLE=4"
# KERNEL_BENCH=1 이면 서버 측정 전에 커널 마이크로벤치를 먼저 돈다
#   (python scripts/bench_fused_decode.py --out ${KERNEL_BENCH_OUT:-results/rq3/fused_kernel.json})
# 프로파일 (PROFILE 환경 변수):
#   reduced (기본, 구성당 약 55분): 반복 2, shared-prefix c1은 8그룹(128요청), c8/c32는 16그룹(256요청), random 500요청
#   full    (구성당 약 2.2시간):    반복 3, shared-prefix c1 8그룹, c8/c32 64그룹(1024요청), random 500요청
#   smoke   (구성당 약 3분):        반복 1, shared-prefix c8 2그룹, random c8 32요청 (스크립트 확인용)
# shared-prefix 나머지 값은 SGLang 기본값: 시스템 프롬프트 2048, 질문 128, 출력 256, range_ratio 1.0
# random: 입력 1024, 출력 256, range_ratio 0.0 (실제 평균 입력 약 513, 출력 약 132). ShareGPT를 HF에서 받는다.
# 결과: <결과 폴더>/<구성>/<ds>_c<동시성>.jsonl (반복마다 한 줄), summarize: scripts/server/tpot_table.py
set -u
OUT_ROOT="$1"; shift
source "$HOME/radixmend/scripts/server/lib.sh"
cd "$RM"; mkdir -p "$OUT_ROOT"
export RM_LOG="$OUT_ROOT/bench.log"
case "${PROFILE:-reduced}" in
  reduced) REPS=2; MATRIX="gsp:1:8 gsp:8:16 gsp:32:16 random:8:500 random:32:500" ;;
  full)    REPS=3; MATRIX="gsp:1:8 gsp:8:64 gsp:32:64 random:8:500 random:32:500" ;;
  smoke)   REPS=1; MATRIX="gsp:8:2 random:8:32" ;;
  *) echo "unknown PROFILE"; exit 2 ;;
esac
rm_log "profile ${PROFILE:-reduced}: reps $REPS, matrix $MATRIX, configs $*"
rm_gpu_free || exit 1

bench() {  # <dir> <ds> <conc> <groups or prompts>
  local d=$1 ds=$2 c=$3 n=$4 extra r
  if [ "$ds" = gsp ]; then extra="--dataset-name generated-shared-prefix --gsp-num-groups $n"
  else extra="--dataset-name random --random-input-len 1024 --random-output-len 256 --num-prompts $n"; fi
  for r in $(seq 1 "$REPS"); do
    python -m sglang.bench_serving --backend sglang --port 30000 $extra --max-concurrency "$c" --flush-cache \
      --disable-tqdm --output-file "$d/${ds}_c${c}.jsonl" > "$d/out_${ds}_c${c}_r$r.txt" 2>&1
    rm_log "  $ds c$c rep$r exit $?"
  done
}

if [ "${KERNEL_BENCH:-0}" = 1 ]; then
  kout="${KERNEL_BENCH_OUT:-results/rq3/fused_kernel.json}"
  if [ -f scripts/bench_fused_decode.py ]; then
    PYTHONPATH="$RM" python scripts/bench_fused_decode.py --out "$kout" > "$OUT_ROOT/kernel_bench.txt" 2>&1
    rm_log "kernel bench exit $? -> $kout (log $OUT_ROOT/kernel_bench.txt)"
  else
    rm_log "KERNEL_BENCH=1 but scripts/bench_fused_decode.py is missing, skip"
  fi
fi

# set_config <spec>: RADIXMEND_* 를 지우고 구성대로 다시 export 한다. 실패하면 1
set_config() {
  local spec=$1 name vars kv
  name=${spec%%:*}; vars=""; [[ "$spec" == *:* ]] && vars=${spec#*:}
  for v in $(compgen -e | grep '^RADIXMEND_'); do unset "$v"; done
  case $name in
    nosecded) ;;
    encode) export RADIXMEND_SECDED=1 RADIXMEND_SECDED_CHECK=0 ;;
    check)  export RADIXMEND_SECDED=1 ;;
    *) [ -n "$vars" ] || { rm_log "config $name has no preset and no variables"; return 1; } ;;
  esac
  IFS=, read -ra kvs <<< "$vars"
  for kv in "${kvs[@]}"; do
    [[ "$kv" =~ ^RADIXMEND_[A-Z_]+=[A-Za-z0-9._-]*$ ]] || { rm_log "bad variable '$kv' in $spec"; return 1; }
    export "$kv"
  done
  CFG_NAME=$name
}

trap rm_stop EXIT
for spec in "$@"; do
  set_config "$spec" || { rm_log "SKIP $spec"; continue; }
  D="$OUT_ROOT/$CFG_NAME"; mkdir -p "$D"
  env | grep '^RADIXMEND_' | sort > "$D/config.env"   # TPOT는 훅 없이 잰다 (RADIXMEND_CMD_DIR 없음)
  rm_log "config $CFG_NAME: $(tr '\n' ' ' < "$D/config.env")"
  rm_start "$D/server.log" || { rm_log "SKIP $CFG_NAME"; continue; }
  for item in $MATRIX; do IFS=: read -r ds c n <<< "$item"; bench "$D" "$ds" "$c" "$n"; done
  rm_stop
done
trap - EXIT
python scripts/server/tpot_table.py "$OUT_ROOT" | tee "$OUT_ROOT/table.txt"
rm_log "BENCH DONE"
