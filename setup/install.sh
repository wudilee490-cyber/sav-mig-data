#!/usr/bin/env bash
# =====================================================================
# sav_mig_data 环境安装 (轻量,不依赖 VACE 仓库)
# =====================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

ENV_NAME="sav_mig_data"
PY_VERSION="3.10"
CUDA_VERSION="12.4"
SKIP_MODELS=0
SKIP_FLASH_ATTN=0
SKIP_VLM=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --env_name) ENV_NAME="$2"; shift 2 ;;
        --py_version) PY_VERSION="$2"; shift 2 ;;
        --cuda) CUDA_VERSION="$2"; shift 2 ;;
        --skip_models) SKIP_MODELS=1; shift ;;
        --skip_flash_attn) SKIP_FLASH_ATTN=1; shift ;;
        --skip_vlm) SKIP_VLM=1; shift ;;
        -h|--help) echo "Usage: bash setup/install.sh [--cuda 12.4] [--skip_models] [--skip_flash_attn] [--skip_vlm]"; exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

case $CUDA_VERSION in
    12.4|12.1|11.8|12.6) ;;
    *) echo "✗ Unsupported CUDA: $CUDA_VERSION"; exit 1 ;;
esac
CU_TAG="cu${CUDA_VERSION//./}"

GREEN="\033[0;32m"; YELLOW="\033[0;33m"; RED="\033[0;31m"; NC="\033[0m"
say()  { echo -e "${GREEN}[install]${NC} $*"; }
warn() { echo -e "${YELLOW}[install]${NC} $*"; }
err()  { echo -e "${RED}[install]${NC} $*"; }

say "============================================================"
say " sav_mig_data env setup"
say "   env: $ENV_NAME, python $PY_VERSION, CUDA $CUDA_VERSION"
say "============================================================"

# Step 0: 系统检查
if ! command -v conda &> /dev/null; then
    err "conda not found"; exit 1
fi
say "  ✓ conda: $(conda --version)"
if ! command -v nvidia-smi &> /dev/null; then
    err "nvidia-smi not found"; exit 1
fi
say "  ✓ driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# Step 1: conda env
if conda env list | grep -q "^${ENV_NAME} "; then
    say "  env '$ENV_NAME' exists"
else
    say "  creating env '$ENV_NAME' python=$PY_VERSION..."
    conda create -n $ENV_NAME python=$PY_VERSION -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
pip install --upgrade pip setuptools wheel ninja packaging -q

# Step 2: PyTorch
say "Step 2: PyTorch (target: torch + $CU_TAG)"
TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ -z "$TORCH_INSTALLED" ]] || [[ "$TORCH_INSTALLED" != *"$CU_TAG"* ]]; then
    [[ -n "$TORCH_INSTALLED" ]] && pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CU_TAG"
else
    say "  ✓ torch $TORCH_INSTALLED"
fi
python -c "import torch; assert torch.cuda.is_available(); print(f'  ✓ torch {torch.__version__}, GPUs: {torch.cuda.device_count()}')"

# Step 3: 安装本项目
say "Step 3: install sav_mig_data package + deps"
pip install -e .

# Step 4: Flash Attention (可选)
if [[ $SKIP_FLASH_ATTN -eq 1 ]]; then
    warn "Step 4: flash-attn (SKIPPED)"
else
    if python -c "import flash_attn" 2>/dev/null; then
        say "  ✓ flash-attn already installed"
    else
        say "  installing flash-attn..."
        pip install flash-attn==2.7.4.post1 --no-build-isolation 2>/dev/null || \
            warn "  flash-attn install failed; VLM will run without flash attention (slower but works)"
    fi
fi

# Step 5: 模型权重
if [[ $SKIP_MODELS -eq 1 ]]; then
    warn "Step 5: model weights (SKIPPED)"
else
    say "Step 5: model weights"
    mkdir -p models

    # Wan VAE + T5 (diffusers 格式)
    if [[ -d "models/Wan2.1-VACE-1.3B-diffusers/vae" ]]; then
        say "  ✓ Wan2.1-VACE-1.3B-diffusers already downloaded"
    else
        say "  downloading Wan2.1-VACE-1.3B-diffusers (~6 GB) for VAE+T5..."
        huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B-diffusers \
            --local-dir models/Wan2.1-VACE-1.3B-diffusers
    fi

    # Qwen3-VL-2B-Instruct
    if [[ $SKIP_VLM -eq 1 ]]; then
        warn "  Qwen3-VL-2B-Instruct (SKIPPED)"
    else
        if [[ -d "models/Qwen3-VL-2B-Instruct" ]] && \
           [[ -n "$(ls -A models/Qwen3-VL-2B-Instruct 2>/dev/null)" ]]; then
            say "  ✓ Qwen3-VL-2B-Instruct already downloaded"
        else
            say "  downloading Qwen3-VL-2B-Instruct (~5 GB)..."
            huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
                --local-dir models/Qwen3-VL-2B-Instruct
        fi
    fi
fi

# Step 6: 验证
say "Step 6: verification"
python setup/verify_env.py
EXIT=$?
if [ $EXIT -eq 0 ]; then
    say "============================================================"
    say " ✓ Installation complete"
    say "============================================================"
    say ""
    say "Next steps:"
    say "  conda activate $ENV_NAME"
    say "  bash scripts/run_all.sh \\"
    say "      --sav_root /data/SA-V/sav_train \\"
    say "      --out_root /data/sav_mig_cache \\"
    say "      --wan_diffusers_id models/Wan2.1-VACE-1.3B-diffusers \\"
    say "      --vlm_model_path models/Qwen3-VL-2B-Instruct \\"
    say "      --max_videos 10"
else
    err "verification failed"
    exit 1
fi
