#!/usr/bin/env bash
# 앞 작업(tmux 세션)이 끝나면 다음 명령을 실행한다. PC가 꺼져 있어도 서버에서 이어서 돈다.
#   tmux new -d -s next "bash ~/radixmend/scripts/server/after.sh <기다릴 세션>... -- <명령>..."
#   예: tmux new -d -s q2 "bash ~/radixmend/scripts/server/after.sh drv -- bash scripts/server/bench_tpot.sh results/rq3/t nosecded check"
# 주의: 실행 중인 bash 스크립트 파일은 고치지 않는다 (bash가 파일을 읽어 가며 실행한다).
#       다음에 돌 스크립트는 고쳐도 되고, 새 코드가 필요하면 앞 작업이 끝난 뒤 복사되게 이 스크립트로 묶는다.
set -u
waits=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do waits+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: after.sh <session>... -- <command>..."; exit 2; }
for s in "${waits[@]}"; do
  while tmux has-session -t "$s" 2>/dev/null; do sleep 20; done
done
echo "[$(date -Is)] after.sh: ${waits[*]} finished, running: $*"
cd "$HOME/radixmend" && exec "$@"
