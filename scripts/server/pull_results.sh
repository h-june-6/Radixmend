#!/usr/bin/env bash
# PC(Git Bash)에서 실행: 서버 ~/radixmend/results 를 PC로 받아 온다. rsync가 없어 tar+ssh를 쓴다.
# 사용: bash pull_results.sh [대상폴더]   (기본 ~/radixmend_results)
# 덮어쓰지 않고 받은 시각별 폴더에 쌓는다.
set -euo pipefail
HOST="${HOST:-dongguk-gpu}"
DEST="${1:-$HOME/radixmend_results}/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"
ssh "$HOST" 'cd ~/radixmend && tar czf - results' | tar xzf - -C "$DEST"
echo "받음: $DEST ($(du -sh "$DEST" | cut -f1))"
