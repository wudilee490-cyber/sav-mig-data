#!/usr/bin/env bash
# =====================================================================
# Run full SA-V → MIG cache pipeline
# =====================================================================
# 这是一个独立项目,不依赖 VACE 仓库,只依赖:
#   - diffusers (AutoencoderKLWan)
#   - transformers (UMT5EncoderModel, Qwen3VL)
#   - 标准 ML 数据处理包
#
# 用法:
#   bash scripts/run_all.sh \
#     --sav_root /data/SA-V/sav_train \
#     --out_root /data/sav_mig_cache \
#     --wan_diffusers_id Wan-AI/Wan2.1-VACE-1.3B-diffusers \
#     --vlm_backend qwen3_vl \
#     --max_videos 100             # 先用 100 个视频跑通,再扩到全量
#
# 各阶段独立 resume:任一阶段中断重跑只会处理未完成的 clip
# =====================================================================

set -e

# 默认值
NUM_FRAMES=81
FRAME_STRIDE=2
TARGET_H=480
TARGET_W=832
CLIPS_PER_VIDEO=2
MAX_OBJECTS=5
S1_WORKERS=8
VLM_BACKEND="qwen3_vl"
VLM_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
VLM_API_ENDPOINT=""
VLM_QPS=0
S2_WORKERS=1
WAN_DIFFUSERS_ID="Wan-AI/Wan2.1-VACE-1.3B-diffusers"
DEVICE="cuda:0"
DTYPE="bfloat16"
VAE_DTYPE="float32"
MAX_VIDEOS=-1
USE_AUTO_MASKLETS=""

# 参数解析
while [[ $# -gt 0 ]]; do
  case $1 in
    --sav_root) SAV_ROOT="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --wan_diffusers_id) WAN_DIFFUSERS_ID="$2"; shift 2 ;;
    --num_frames) NUM_FRAMES="$2"; shift 2 ;;
    --frame_stride) FRAME_STRIDE="$2"; shift 2 ;;
    --target_size) TARGET_H="$2"; TARGET_W="$3"; shift 3 ;;
    --clips_per_video) CLIPS_PER_VIDEO="$2"; shift 2 ;;
    --max_objects) MAX_OBJECTS="$2"; shift 2 ;;
    --s1_workers) S1_WORKERS="$2"; shift 2 ;;
    --vlm_backend) VLM_BACKEND="$2"; shift 2 ;;
    --vlm_model_path) VLM_MODEL_PATH="$2"; shift 2 ;;
    --vlm_endpoint) VLM_API_ENDPOINT="$2"; shift 2 ;;
    --vlm_qps) VLM_QPS="$2"; shift 2 ;;
    --s2_workers) S2_WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --vae_dtype) VAE_DTYPE="$2"; shift 2 ;;
    --max_videos) MAX_VIDEOS="$2"; shift 2 ;;
    --use_auto_masklets) USE_AUTO_MASKLETS="--use_auto_masklets"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$SAV_ROOT" ] || [ -z "$OUT_ROOT" ]; then
  echo "Usage: bash run_all.sh --sav_root ... --out_root ... [--wan_diffusers_id Wan-AI/Wan2.1-VACE-1.3B-diffusers]"
  exit 1
fi

mkdir -p "$OUT_ROOT"

# 切到项目根目录 (scripts/run_all.sh 的上一级)
cd "$(dirname "$0")/.."

echo "================================================================"
echo " STAGE 1: extract clips"
echo "================================================================"
python scripts/01_extract_clips.py \
  --sav_root "$SAV_ROOT" \
  --out_root "$OUT_ROOT" \
  --num_frames $NUM_FRAMES \
  --frame_stride $FRAME_STRIDE \
  --target_size $TARGET_H $TARGET_W \
  --clips_per_video $CLIPS_PER_VIDEO \
  --max_objects $MAX_OBJECTS \
  --num_workers $S1_WORKERS \
  --max_videos $MAX_VIDEOS \
  $USE_AUTO_MASKLETS

echo ""
echo "================================================================"
echo " STAGE 2: caption objects (VLM)"
echo "================================================================"
S2_ARGS="--clips_root $OUT_ROOT --backend $VLM_BACKEND --num_workers $S2_WORKERS"
if [ "$VLM_BACKEND" = "qwen3_vl" ] || [ "$VLM_BACKEND" = "qwen2_vl" ]; then
  S2_ARGS="$S2_ARGS --model_path $VLM_MODEL_PATH"
elif [ "$VLM_BACKEND" = "openai_api" ]; then
  S2_ARGS="$S2_ARGS --api_endpoint $VLM_API_ENDPOINT --api_qps $VLM_QPS"
fi
python scripts/02_caption_objects.py $S2_ARGS

echo ""
echo "================================================================"
echo " STAGE 3: VAE encode"
echo "================================================================"
python scripts/03_encode_latents.py \
  --clips_root "$OUT_ROOT" \
  --vae_model_id "$WAN_DIFFUSERS_ID" \
  --device "$DEVICE" --dtype "$VAE_DTYPE" --output_dtype "$DTYPE"

echo ""
echo "================================================================"
echo " STAGE 4: T5 encode"
echo "================================================================"
python scripts/04_encode_text.py \
  --clips_root "$OUT_ROOT" \
  --text_model_id "$WAN_DIFFUSERS_ID" \
  --device "$DEVICE" --dtype "$DTYPE"

echo ""
echo "================================================================"
echo " STAGE 5: build index"
echo "================================================================"
python scripts/05_build_index.py \
  --clips_root "$OUT_ROOT" \
  --val_ratio 0.05

echo ""
echo "================================================================"
echo " ✓ ALL DONE: $OUT_ROOT"
echo "================================================================"
ls -la "$OUT_ROOT"/index.json "$OUT_ROOT"/train.json "$OUT_ROOT"/val.json "$OUT_ROOT"/_stats.json
