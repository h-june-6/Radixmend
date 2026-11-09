# 서버 실행 공통 함수. 다른 스크립트에서 source 해서 쓴다.
#   source ~/radixmend/scripts/server/lib.sh
# 서버 기능은 호출하는 쪽에서 환경 변수로 고른다.
#   RADIXMEND_CMD_DIR=...  결함 주입 훅 (rq1_* 스크립트가 쓴다)
#   RADIXMEND_SECDED=1     모듈 A (RADIXMEND_SECDED_CHECK=0 이면 encode만)
# 둘 다 없으면 radixmend 코드가 전혀 끼지 않은 기본 SGLang 서버다.

source "$HOME/.radixmend_env"
RM="$HOME/radixmend"
RM_CMD_DIR="/tmp/$USER/rm_cmd"
RM_SITE="$RM:$RM/radixmend/sglang_site"
RM_SERVER_ARGS="${RM_SERVER_ARGS:---model-path Qwen/Qwen3-8B --attention-backend triton --page-size 1 --port 30000 --mem-fraction-static 0.85 --kv-canary none}"
RM_URL="http://127.0.0.1:30000"
mkdir -p "$RM_CMD_DIR"

rm_log() { echo "[$(date -Is)] $*" | tee -a "${RM_LOG:-/dev/null}"; }

rm_gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

# GPU가 비어 있고 이 계정의 sglang 서버가 없으면 0
rm_gpu_free() {
  if pgrep -u "$USER" -f "sglang.launch_server" >/dev/null; then rm_log "sglang server already running"; return 1; fi
  local u; u=$(rm_gpu_used)
  if [ "${u:-99999}" -ge 1000 ]; then rm_log "GPU busy: ${u} MiB"; return 1; fi
  return 0
}

# rm_start <로그 파일>: 서버를 띄우고 /health 가 될 때까지 기다린다 (최대 15분). RM_SRV 에 pid.
rm_start() {
  local log="$1"
  setsid env PYTHONPATH="$RM_SITE" python -m sglang.launch_server $RM_SERVER_ARGS > "$log" 2>&1 &
  RM_SRV=$!
  for _ in $(seq 1 300); do
    if curl -sf "$RM_URL/health" >/dev/null; then
      local bad; bad=$(grep -c "\[radixmend\] .* not installed" "$log")
      rm_log "server ready: $log, gpu $(rm_gpu_used) MiB, secded=${RADIXMEND_SECDED:-0}, hook=$([ -n "${RADIXMEND_CMD_DIR:-}" ] && echo on || echo off), not-installed lines=$bad"
      [ "$bad" -eq 0 ] || { rm_log "radixmend install failed, see $log"; return 1; }
      return 0
    fi
    kill -0 "$RM_SRV" 2>/dev/null || break
    sleep 3
  done
  rm_log "server failed to start: $log"
  grep -iE "out of memory|not enough memory|Traceback" "$log" | head -3 | tee -a "${RM_LOG:-/dev/null}"
  return 1
}

# rm_stop: 서버 프로세스 그룹을 내리고 GPU 메모리가 풀릴 때까지 기다린다
rm_stop() {
  [ -n "${RM_SRV:-}" ] || return 0
  kill -TERM -- "-$RM_SRV" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$RM_SRV" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$RM_SRV" 2>/dev/null
  for _ in $(seq 1 30); do [ "$(rm_gpu_used)" -lt 1000 ] && break; sleep 2; done
  rm_log "server stopped, gpu $(rm_gpu_used) MiB"
  RM_SRV=
}

rm_flush() { curl -s -X POST "$RM_URL/flush_cache" >/dev/null; }

# rm_stats [reset]: 모듈 A 교정 카운터 (훅이 켜진 서버에서만)
rm_stats() {
  local reset=False; [ "${1:-}" = reset ] && reset=True
  python -c "import sys;sys.path.insert(0,'$RM/scripts');import rq1_kv_sweep as k;from pathlib import Path;r=k.hook(Path('$RM_CMD_DIR'),{'op':'secded_stats','reset':$reset});r.pop('id',None);r.pop('ok',None);print(r)" 2>&1 | tail -1
}
