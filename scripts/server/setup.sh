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
mkdir -p "$SCRATCH"/{hf,triton,tmp,uv-cache} "$LOG_DIR"

# 1) 셸 환경: 다음 로그인부터 자동 적용
cat > "$HOME/.radixmend_env" <<ENV
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton
export TMPDIR=$SCRATCH/tmp
export PIP_NO_CACHE_DIR=1
export UV_CACHE_DIR=$SCRATCH/uv-cache
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
if [ -f $VENV/bin/activate ]; then source $VENV/bin/activate; fi
# JIT 커널(flashinfer, sglang KV 커널)은 venv의 CUDA 13 nvcc로 빌드한다.
# 시스템 nvcc 12.2를 쓰면 flashinfer는 '--compress-mode' 오류로 죽고 triton KV 커널 빌드도 실패한다.
CU13=$VENV/lib/python3.11/site-packages/nvidia/cu13
if [ -x \$CU13/bin/nvcc ]; then export CUDA_HOME=\$CU13; export PATH=\$CU13/bin:\$PATH; fi
ENV
grep -q radixmend_env "$HOME/.bashrc" 2>/dev/null || echo '[ -f ~/.radixmend_env ] && source ~/.radixmend_env' >> "$HOME/.bashrc"
# shellcheck disable=SC1090
source "$HOME/.radixmend_env"

# 2) venv: sglang 0.5.22 wheel은 Python 3.11~3.13만 있다 (시스템 python3는 3.10)
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.13 /opt/anaconda3/bin/python3.11 /opt/anaconda3/bin/python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(not (3,11) <= sys.version_info[:2] <= (3,13))'; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "Python 3.11~3.13을 찾지 못함. PYTHON=/경로/python3.11 로 지정" >&2; exit 1; }
echo "python: $PY ($("$PY" --version))"
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV" || { "$PY" -m pip install --user -q virtualenv && "$PY" -m virtualenv "$VENV"; }
fi
source "$VENV/bin/activate"
python -m pip install -q -U pip uv

# 3) SGLang (torch, triton 포함) + 연구 코드 의존성
uv pip install "sglang[all]==$SGLANG_VERSION" pytest numpy huggingface_hub
# nvcc/crt/nvvm은 런타임 헤더(13.0)와 맞춰 고정하고, JIT가 찾는 경로 링크를 만든다
uv pip install "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88"
CU13="$(python -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')"
[ -e "$CU13/lib64" ] || ln -s lib "$CU13/lib64"
if [ ! -e "$CU13/lib/libcudart.so" ] && [ -e "$CU13/lib/libcudart.so.13" ]; then ln -s libcudart.so.13 "$CU13/lib/libcudart.so"; fi
source "$HOME/.radixmend_env"
nvcc --version | tail -1
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
  hf download "$m" --exclude "*.pth" --exclude "original/*" >/dev/null
  echo "downloaded $m"
done

# 6) 환경 기록 (대여마다 남김)
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "sglang: $SGLANG_VERSION"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,ecc.mode.current --format=csv
  echo "nvcc: $(command -v nvcc)"; nvcc --version | tail -1; uname -r; python --version
  quota -s 2>/dev/null || true; df -h "$HOME" "$SCRATCH"
} > "$LOG_DIR/system.txt" 2>&1
cat "$LOG_DIR/system.txt"
echo "setup done. 결과 기록: $LOG_DIR"
