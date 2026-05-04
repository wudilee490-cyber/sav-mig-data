# sav_mig_data

把 SA-V (或类似格式) 的视频实例分割数据集,转成 MIG adapter 训练所需的离线缓存。

## 这是什么

一个**完全独立**的数据生成流水线项目。它:
- 不依赖 VACE 主仓
- 只依赖 `diffusers + transformers + 数据包`
- 输出固定格式的 .pt 缓存,供下游训练项目消费

下游训练项目: [`vace_mig_model`](../vace_mig_model/) — 该项目读这里产出的 `clips_root/` 直接训练。

## 输出格式 (与下游项目的契约)

```
clips_root/
├── index.json         全部样本元数据
├── train.json         训练集 (~95%)
├── val.json           验证集 (~5%)
├── _stats.json        数据集统计
├── _broken.json       不合格样本及原因
└── {clip_id}/
    ├── frames.npy             [F, H_t, W_t, 3] uint8     (Stage 1, 可在 Stage 3 后删除)
    ├── masks_orig.npy         [K, F, H_o, W_o] uint8     (Stage 1, 可在 Stage 3 后删除)
    ├── captions.json          {phrases, global_prompt, ...} (Stage 2)
    ├── video_latent.pt        [C_lat, F_lat, H_lat, W_lat]  bf16 (Stage 3)
    ├── obj_image_masks.pt     [K, H_lat, W_lat]              fp32 (Stage 3, 首帧)
    ├── obj_volume_masks.pt    [K, F_p, H_p, W_p]             fp32 (Stage 3, patch化)
    ├── verb_embeddings.pt     [K, N_v, text_dim]             bf16 (Stage 4)
    ├── verb_masks.pt          [K, N_v]                       bool (Stage 4)
    ├── global_context.pt      [L_g, text_dim]                bf16 (Stage 4)
    └── meta.json              (累积所有 stage 元信息)
```

## 安装

```bash
git clone <this repo>
cd sav_mig_data
bash setup/install.sh                                # CUDA 12.4 默认
# bash setup/install.sh --cuda 12.1 --skip_models    # 自定义
```

环境名: `sav_mig_data` (conda)。

## 跑数据

```bash
conda activate sav_mig_data

# 推荐: 先用 10 个视频走通流程
bash scripts/run_all.sh \
    --sav_root /data/SA-V/sav_train \
    --out_root /data/sav_mig_cache_test \
    --wan_diffusers_id models/Wan2.1-VACE-1.3B-diffusers \
    --vlm_model_path models/Qwen3-VL-2B-Instruct \
    --max_videos 10
```

详细的分阶段运行说明见 [`scripts/README.md`](scripts/README.md)。

## 流水线概览

```
Stage 1: extract_clips    SA-V mp4+json  →  frames.npy + masks_orig.npy        (CPU多进程)
Stage 2: caption_objects  frames+masks   →  captions.json                       (GPU, VLM)
Stage 3: encode_latents   frames         →  video_latent.pt + masks 各种产物    (GPU, VAE)
Stage 4: encode_text      captions       →  verb_embeddings.pt + global_context.pt (GPU, T5)
Stage 5: build_index      全部产物       →  index/train/val.json                (CPU)
```

每个阶段独立可重跑、可断点续跑、可单独调度到不同机器。

## 模型文件

需要的模型权重 (install.sh 默认下载):

| 模型 | 用途 | 大小 |
|---|---|---|
| `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | VAE (Stage 3) + T5 (Stage 4) | ~6 GB |
| `Qwen/Qwen3-VL-2B-Instruct` | VLM caption (Stage 2) | ~5 GB |

## 时间估计 (单卡 A100)

| 阶段 | 速度 | SA-V 全量预估 |
|---|---|---|
| Stage 1 | ~5K clips/h | 20 h |
| Stage 2 (本地) | 0.5-1 clip/s | 30-60 h |
| Stage 2 (vLLM API ×16) | 4-8 clips/s | 6-10 h |
| Stage 3 | 0.8 clip/s | 30 h |
| Stage 4 | 3 clips/s | 10 h |
| Stage 5 | <1 min | <1 min |

## 磁盘空间

| 阶段后 | 单 clip | SA-V 全量 (~100K clips) |
|---|---|---|
| Stage 1+2 后 | ~70 MB | ~7 TB |
| Stage 3+4 后 (含 frames) | ~75 MB | ~7.5 TB |
| 删 frames+masks 后 | ~5 MB | **~500 GB** ← 推荐保留状态 |

Stage 3 完成后可以删 `frames.npy` 和 `masks_orig.npy` 大幅省盘:
```bash
find /data/sav_mig_cache -name "frames.npy" -delete
find /data/sav_mig_cache -name "masks_orig.npy" -delete
```

## 不依赖 VACE 仓库

本项目刻意做到只依赖 `diffusers + transformers`:
- VAE 用 `diffusers.AutoencoderKLWan` 加载 (与 VACE 训练用同一权重)
- T5 用 `transformers.UMT5EncoderModel` 加载
- 所以可以在没装 VACE/Wan 仓库的机器上运行

下游训练项目 (`vace_mig_model`) 读 .pt 文件,也不需要知道这个数据项目的存在。
