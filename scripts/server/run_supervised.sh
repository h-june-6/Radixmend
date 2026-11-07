#!/usr/bin/env bash
# 결함 주입으로 서버가 죽을 수 있는 시행(RQ1-5 인덱스 등)을 돌린다.
# 서버를 띄우고, 드라이버가 종료코드 3(서버 다운)으로 끝나면 서버를 다시 띄워 이어서 실행한다.
# 사용: bash scripts/server/run_supervised.sh <로그 폴더> -- python scripts/rq1_index_sweep.py --out results/rq1/index
# 이미 30000 포트에 서버가 떠 있으면 먼저 내려야 한다.
set -uo pipefail
LOG_DIR="$1"; shift; [ "$1" = "--" ] && shift
mkdir -p "$LOG_DIR"
source "$HOME/.radixmend_env"
cd "$HOME/radixmend"
export PYTHONPATH="$HOME/radixmend:$HOME/radixmend/radixmend/sglang_site"
export RADIXMEND_CMD_DIR="/tmp/$USER/rm_cmd"
mkdir -p "$RADIXMEND_CMD_DIR"
SERVER_ARGS=${SERVER_ARGS:-"--model-path Qwen/Qwen3-8B --attention-backend triton --page-size 1 --port 30000 --mem-fraction-static 0.85 --kv-canary none"}
MAX_RESTARTS=${MAX_RESTARTS:-200}
RUN_ID="$(date +%Y%m%d-%H%M%S)"  # 실행마다 로그 이름이 겹치지 않게

start_server() {
  local n="$1"
  python -m sglang.launch_server $SERVER_ARGS > "$LOG_DIR/server_${RUN_ID}_$n.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 300); do
    curl -sf http://127.0.0.1:30000/health >/dev/null && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    sleep 2
  done
  return 1
}

stop_server() {
  pkill -P "$SERVER_PID" 2>/dev/null; kill "$SERVER_PID" 2>/dev/null
  sleep 5; pkill -9 -f "sglang.launch_server" 2>/dev/null
  # GPU 메모리가 풀릴 때까지 기다린다
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 1000 ] && break; sleep 2
  done
}

for n in $(seq 0 "$MAX_RESTARTS"); do
  echo "[$(date -Is)] start server #$n" | tee -a "$LOG_DIR/supervisor.log"
  if ! start_server "$n"; then
    echo "[$(date -Is)] server failed to start, see server_${RUN_ID}_$n.log" | tee -a "$LOG_DIR/supervisor.log"
    stop_server; continue
  fi
  "$@" 2>&1 | tee -a "$LOG_DIR/driver.log"
  rc=${PIPESTATUS[0]}
  echo "[$(date -Is)] driver exit $rc" | tee -a "$LOG_DIR/supervisor.log"
  stop_server
  [ "$rc" -eq 3 ] || exit "$rc"
done
