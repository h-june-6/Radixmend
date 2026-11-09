#!/usr/bin/env bash
# 대여 첫날 setup.sh 다음에 돌리는 확인. tmux 안에서 실행한다 (약 6분).
#   tmux new -d -s day1 "bash ~/radixmend/scripts/server/day1.sh"
# 확인 항목:
#   1. GPU, 드라이버, ECC, nvcc 경로
#   2. 전체 pytest (GPU에서, TRITON_INTERPRET 없이)
#   3. 훅만 켠 서버: 고정 프롬프트 출력 저장 + golden ID (지난 대여 기준 94c11b0adbc5)
#   4. 훅 + 모듈 A 서버: 같은 출력인지(결함이 없으면 비트 단위로 같아야 함), golden ID, 카운터 0
# 결과: results/day1/<시각>/, 마지막 줄에 DAY1 PASS 또는 DAY1 FAIL
set -u
source "$HOME/radixmend/scripts/server/lib.sh"
cd "$RM"
OUT="$RM/results/day1/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
export RM_LOG="$OUT/day1.log"
EXPECT_GOLDEN="${EXPECT_GOLDEN:-94c11b0adbc5}"
fail=0

rm_log "1) gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total,ecc.mode.current --format=csv,noheader)"
rm_log "   nvcc: $(command -v nvcc) $(nvcc --version | tail -1)"
rm_gpu_free || { rm_log "DAY1 FAIL (GPU not free)"; exit 1; }

python -m pytest -q tests > "$OUT/pytest.txt" 2>&1 || fail=1
rm_log "2) pytest: $(tail -n 1 "$OUT/pytest.txt")"

golden() {  # <out dir>: one control trial; prints the golden id
  python scripts/rq1_kv_sweep.py --out "$1" --kinds none --bits 0 --reps 1 > "$1.log" 2>&1
  grep -o "golden ok ([0-9a-f]*)" "$1.log" | grep -o "[0-9a-f]\{12\}"
}

trap rm_stop EXIT
export RADIXMEND_CMD_DIR="$RM_CMD_DIR"
unset RADIXMEND_SECDED RADIXMEND_SECDED_CHECK
if rm_start "$OUT/server_plain.log"; then
  rm_flush; python scripts/smoke_client.py "$OUT/plain.json" >/dev/null
  g=$(golden "$OUT/golden_plain"); rm_log "3) hook server: golden ${g:-none} (expect $EXPECT_GOLDEN)"
  [ "$g" = "$EXPECT_GOLDEN" ] || fail=1
  rm_stop
else fail=1; fi

export RADIXMEND_SECDED=1
if rm_start "$OUT/server_secded.log"; then
  rm_stats reset >/dev/null
  rm_flush; python scripts/smoke_client.py "$OUT/secded.json" >/dev/null
  same=$(python scripts/smoke_client.py --compare "$OUT/plain.json" "$OUT/secded.json" | tail -1)
  g=$(golden "$OUT/golden_secded")
  st=$(rm_stats)
  rm_log "4) secded server: outputs $same, golden ${g:-none}, counters $st"
  [ "$same" = "5/5 identical" ] && [ "$g" = "$EXPECT_GOLDEN" ] || fail=1
  echo "$st" | grep -qE "corrected': [1-9]|uncorrectable': [1-9]" && { rm_log "   nonzero counters without faults"; fail=1; }
  rm_stop
else fail=1; fi
trap - EXIT

[ $fail -eq 0 ] && rm_log "DAY1 PASS" || rm_log "DAY1 FAIL (see $OUT)"
exit $fail
