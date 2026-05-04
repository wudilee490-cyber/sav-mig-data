# Pipeline 阶段详解

每个阶段独立可跑,失败可重试。下面给出**生产环境推荐用法**。

## Stage 1: extract_clips (CPU 多进程)

```bash
python scripts/01_extract_clips.py \
    --sav_root /data/SA-V/sav_train \
    --out_root /data/sav_mig_cache \
    --num_frames 81 --frame_stride 2 \
    --target_size 480 832 \
    --clips_per_video 2 \
    --max_objects 5 \
    --num_workers 16
```

**关键参数**:
- `--num_frames 81`:与 Wan2.1 训练默认一致
- `--frame_stride 2`:24fps × stride 2 = 12fps 训练帧率
- `--clips_per_video 2`:每段长视频抽 2 个不重叠 clip,数据增强
- `--use_auto_masklets`:加上后也用 `_auto.json` (量大但质量参差)

**速度**: 16 核机器 ~5K clips/小时。

## Stage 2: caption_objects (VLM)

### 选项 A: 本地 GPU 跑 Qwen3-VL-2B-Instruct (默认,简单)

```bash
python scripts/02_caption_objects.py \
    --clips_root /data/sav_mig_cache \
    --backend qwen3_vl \
    --model_path models/Qwen3-VL-2B-Instruct \
    --device cuda:0
```

单卡 A100 上约 0.5-1 clip/s。

### 选项 B: vLLM 远程服务 (高吞吐,推荐生产)

```bash
# 终端1: 起 vLLM 服务 (单独 conda env 避免依赖冲突)
conda create -n vllm_serving python=3.10 -y && conda activate vllm_serving
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct --port 8000 \
    --max-model-len 8192

# 终端2: caption 用 16 并发 API 调用
conda activate sav_mig_data
python scripts/02_caption_objects.py \
    --clips_root /data/sav_mig_cache \
    --backend openai_api \
    --api_endpoint http://localhost:8000/v1/chat/completions \
    --api_model Qwen/Qwen3-VL-2B-Instruct \
    --num_workers 16
```

5-8 倍提速,100K clips 约 6-10 小时。

### 选项 C: dummy backend (流水线冒烟测试)

```bash
python scripts/02_caption_objects.py \
    --clips_root /data/sav_mig_cache_test \
    --backend dummy
```

不调真实 VLM,所有 caption 是固定字符串 `"moving across the scene"`。
验证流水线代码是否完整后再跑真实 VLM。

### 切换到 Thinking 版本(质量优先)

```bash
python scripts/02_caption_objects.py \
    --clips_root /data/sav_mig_cache \
    --backend qwen3_vl \
    --model_path Qwen/Qwen3-VL-2B-Thinking
```

慢 3-5 倍但 caption 质量更准。脚本会自动切到 thinking-friendly prompt。

## Stage 3: VAE encode

```bash
python scripts/03_encode_latents.py \
    --clips_root /data/sav_mig_cache \
    --vae_model_id models/Wan2.1-VACE-1.3B-diffusers \
    --device cuda:0 --dtype float32 --output_dtype bfloat16
```

**关键**:
- `--dtype float32`:VAE 计算用 fp32 数值稳定
- `--output_dtype bfloat16`:保存时压成 bf16 省盘

约 0.8 clip/s on A100。

完成后**强烈建议删原视频帧**:
```bash
find /data/sav_mig_cache -name "frames.npy" -delete
find /data/sav_mig_cache -name "masks_orig.npy" -delete
```
能省 90%+ 磁盘。

## Stage 4: T5 encode

```bash
python scripts/04_encode_text.py \
    --clips_root /data/sav_mig_cache \
    --text_model_id models/Wan2.1-VACE-1.3B-diffusers \
    --device cuda:0 --dtype bfloat16
```

UMT5-XXL 显存约 11 GB (bf16)。如果显存不够可以 `--dtype float16`,但精度有损。

## Stage 5: build_index

```bash
python scripts/05_build_index.py \
    --clips_root /data/sav_mig_cache \
    --val_ratio 0.05 \
    --exclude_static
```

`--exclude_static` 排除所有物体都是 "standing still" 的 clip。

完成后看统计:
```bash
cat /data/sav_mig_cache/_stats.json
```

期望:
```json
{
  "total_clips": 87392,
  "train_clips": 83022,
  "val_clips": 4370,
  "unique_videos": 50000,
  "n_objects_dist": {"1": 12000, "2": 35000, "3": 28000, "4": 9000, "5": 3000},
  "phrase_word_count": {"min": 2, "max": 8, "mean": 4.3}
}
```

## 验收 (caption 质量目检)

```bash
# 抽 20 个看
python -c "
import json, glob, random
clips = glob.glob('/data/sav_mig_cache/*/captions.json')
for p in random.sample(clips, 20):
    with open(p) as f: c = json.load(f)
    print(c['phrases'])
"
```

**好的 caption 长这样**:
```
['running across grass', 'standing still', 'walking toward camera']
```

**坏的 caption** (需要换更大的 VLM 或调 prompt):
```
['the object is moving', 'this is a dog', 'objects in the scene']
```
