#!/usr/bin/env bash
# 빈 계정(3일 대여, 디스크 초기화 후)에서 실험 환경을 한 번에 복구한다.
# 사용: bash setup.sh            # 기본: SGLang 0.5.22, Qwen3-8B
#       SGLANG_VERSION=0.5.19 bash setup.sh
#       MODELS="Qwen/Qwen3-8B meta-llama/Meta-Llama-3-8B-Instruct" HF_TOKEN=... bash setup.sh
#
# 배치 원칙 (개인 쿼터 50GB):
#   ~/radixmend            코드와 결과 (작음, 남겨야 하는 것)
#   /tmp/$USER/radixmend   venv, 모델, 캐시 (다시 받으면 되는 것)
set -euo pipefail

SGLANG_VERSION="${SGLANG_VERSION:-0.5.22}"
MODELS="${MODELS:-Qwen/Qwen3-8B}"
CODE_DIR="${CODE_DIR:-$HOME/radixmend}"
SCRATCH="${SCRATCH:-/tmp/$USER/radixmend}"
VENV="$SCRATCH/venv"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$CODE_DIR/results/env/$STAMP"

mkdir -p "/tmp/$USER" && chmod 700 "/tmp/$USER"
mkdir -p "$SCRATCH"/{hf,triton,tmp} "$LOG_DIR"

# 1) 셸 환경: 다음 로그인부터 자동 적용
cat > "$HOME/.radixmend_env" <<ENV
export PATH=/usr/local/cuda-12.2/bin:\$PATH
export CUDA_HOME=/usr/local/cuda-12.2
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton
export TMPDIR=$SCRATCH/tmp
export PIP_NO_CACHE_DIR=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
[ -f $VENV/bin/activate ] && source $VENV/bin/activate
ENV
grep -q radixmend_env "$HOME/.bashrc" 2>/dev/null || echo '[ -f ~/.radixmend_env ] && source ~/.radixmend_env' >> "$HOME/.bashrc"
# shellcheck disable=SC1090
source "$HOME/.radixmend_env"

# 2) venv (python3-venv가 없으면 virtualenv로 대체)
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" 2>/dev/null || { python3 -m pip install --user -q virtualenv && python3 -m virtualenv "$VENV"; }
fi
source "$VENV/bin/activate"
python -m pip install -q -U pip uv

# 3) SGLang (torch, triton 포함) + 연구 코드 의존성
uv pip install "sglang[all]==$SGLANG_VERSION" pytest numpy huggingface_hub
uv pip freeze > "$LOG_DIR/pip_freeze.txt"

# 4) GPU 확인
python - <<'PY' | tee "$LOG_DIR/torch_check.txt"
import torch, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
PY

# 5) 모델 내려받기 (Llama는 HF_TOKEN 필요)
for m in $MODELS; do
  huggingface-cli download "$m" --exclude "*.pth" "original/*" >/dev/null
  echo "downloaded $m"
done

# 6) 환경 기록 (대여마다 남김)
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "sglang: $SGLANG_VERSION"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,ecc.mode.current --format=csv
  nvcc --version | tail -1; uname -r; python --version
  quota -s 2>/dev/null || true; df -h "$HOME" "$SCRATCH"
} > "$LOG_DIR/system.txt" 2>&1
cat "$LOG_DIR/system.txt"
echo "setup done. 결과 기록: $LOG_DIR"
