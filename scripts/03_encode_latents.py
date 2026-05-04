"""
Stage 3: encode video frames → VAE latent  (diffusers-based)
=============================================================

输入:
    Stage 1 产出: {clip_id}/frames.npy  [F, H_t, W_t, 3] uint8

输出:
    {clip_id}/video_latent.pt        [C_lat, F_lat, H_lat, W_lat]  bf16
    {clip_id}/obj_image_masks.pt     [K, H_lat, W_lat]             float32 (首帧)
    {clip_id}/obj_volume_masks.pt    [K, F_p, H_p, W_p]            float32 (patch化)

依赖: diffusers + transformers, **不依赖 VACE 仓库**。
    AutoencoderKLWan 是 diffusers 官方提供的 Wan VAE 实现,
    与 VACE 主仓使用的 VAE 是同一个权重 (Wan-AI/Wan2.1-VACE-1.3B-diffusers)。

Wan VAE 时间下采样:
    输入 81 帧 → latent ~21 帧 (factor 4)
    具体倍率由 VAE 自己决定,F_lat = video_latent.shape[-3]
    
    patch_size 在 diffusers 的 AutoencoderKLWan.config 不直接给,
    我们在 Wan2.1 论文/代码里查到:patch_size = (1, 2, 2)
    H_p = H_lat // 2, W_p = W_lat // 2, F_p = F_lat (因为 p_t=1)

mask 对齐:
    masks_orig 来自 SA-V 标注 (24fps stride 后约 12fps)
    需要先时间下采样到 F_lat 帧 (用 nearest 因为是二值)
    再空间 resize 到 latent 分辨率
    再 patch 化得到 [F_p, H_p, W_p]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


# Wan VAE 的 patch_size (来自 Wan2.1 论文,与 diffusers 实现一致)
WAN_PATCH_SIZE = (1, 2, 2)


# =========================================================================
# Mask 工具
# =========================================================================
def temporal_align_masks(masks: np.ndarray, F_target: int) -> np.ndarray:
    """[K, F_in, H, W] → [K, F_target, H, W],用 nearest 采样保持二值"""
    K, F_in, H, W = masks.shape
    if F_in == F_target:
        return masks
    indices = np.linspace(0, F_in - 1, F_target).round().astype(int)
    return masks[:, indices]


def patchify_volume(masks: np.ndarray, p_t: int, p_h: int, p_w: int) -> np.ndarray:
    """[K, F_lat, H_lat, W_lat] → [K, F_p, H_p, W_p],max-pool"""
    K, F_lat, H_lat, W_lat = masks.shape
    F_p = F_lat // p_t
    H_p = H_lat // p_h
    W_p = W_lat // p_w
    m = masks[:, :F_p*p_t, :H_p*p_h, :W_p*p_w]
    m = m.reshape(K, F_p, p_t, H_p, p_h, W_p, p_w)
    return m.max(axis=(2, 4, 6)).astype(np.uint8)


def resize_masks_to_latent(masks: np.ndarray, H_lat: int, W_lat: int) -> np.ndarray:
    """[K, F, H_t, W_t] → [K, F, H_lat, W_lat],PIL nearest"""
    K, F = masks.shape[:2]
    out = np.zeros((K, F, H_lat, W_lat), dtype=np.uint8)
    for k in range(K):
        for f in range(F):
            img = Image.fromarray(masks[k, f] * 255).resize(
                (W_lat, H_lat), Image.NEAREST)
            out[k, f] = (np.array(img) > 127).astype(np.uint8)
    return out


# =========================================================================
# VAE 加载 + encode (diffusers 路径)
# =========================================================================
def load_wan_vae(model_id: str, device: torch.device, dtype: torch.dtype):
    """
    用 diffusers 加载 Wan VAE。
    
    model_id 接受:
        - HF id:    "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        - 本地路径: "/path/to/Wan2.1-VACE-1.3B-diffusers"
    
    diffusers 仓库里 vae 在 subfolder="vae" 下。
    """
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as e:
        raise ImportError(
            "diffusers 未安装或版本过低。\n"
            "请确保安装 diffusers >= 0.31.0:\n"
            "  pip install 'diffusers>=0.31.0'"
        ) from e

    print(f"[VAE] loading from {model_id} (subfolder='vae')...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=dtype,
    )
    vae = vae.to(device).eval()
    return vae


@torch.no_grad()
def encode_video(frames: np.ndarray, vae, device, dtype) -> torch.Tensor:
    """
    frames [F, H, W, 3] uint8 → latent [C_lat, F_lat, H_lat, W_lat]
    
    AutoencoderKLWan 接口:
        输入: [B, 3, F, H, W],范围 [-1, 1]
        输出 vae.encode(x): {latent_dist: DiagonalGaussianDistribution}
              .latent_dist.sample() → [B, C_lat, F_lat, H_lat, W_lat]
    """
    x = torch.from_numpy(frames).to(device).float() / 127.5 - 1.0     # [F, H, W, 3]
    x = x.permute(3, 0, 1, 2).unsqueeze(0).to(dtype)                   # [1, 3, F, H, W]

    out = vae.encode(x)
    if hasattr(out, "latent_dist"):
        z = out.latent_dist.sample()
    elif isinstance(out, (list, tuple)):
        z = out[0]
    else:
        z = out

    # diffusers VAE 通常会做 scaling,但 AutoencoderKLWan 默认 scaling_factor=1.0
    # 如果 config 给了非 1 的 scaling_factor,这里需要乘上
    sf = getattr(vae.config, "scaling_factor", 1.0)
    if sf != 1.0:
        z = z * sf

    if z.dim() == 5:
        z = z.squeeze(0)                                                # [C_lat, F_lat, H_lat, W_lat]
    return z.cpu()


# =========================================================================
# 主循环
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clips_root", required=True)
    p.add_argument("--vae_model_id",
                   default="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                   help="HF id 或本地路径,内部应有 subfolder='vae'")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"],
                   help="VAE 通常用 float32 (encode 数值稳定),"
                        "bfloat16 也可以但需要测试")
    p.add_argument("--output_dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"],
                   help="保存 video_latent.pt 的精度")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_clips", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    out_dtype = {"float32": torch.float32, "float16": torch.float16,
                  "bfloat16": torch.bfloat16}[args.output_dtype]

    vae = load_wan_vae(args.vae_model_id, device, dtype)
    p_t, p_h, p_w = WAN_PATCH_SIZE
    print(f"[Init] VAE loaded, patch_size = ({p_t}, {p_h}, {p_w})")

    clips_root = Path(args.clips_root)
    clips = sorted([d for d in clips_root.iterdir() if d.is_dir()])
    if args.max_clips > 0:
        clips = clips[:args.max_clips]

    n_ok = 0; n_err = 0; n_skip = 0
    t0 = time.time()
    for i, cd in enumerate(clips):
        if not (cd / "DONE").exists() or (cd / "SKIPPED_NO_OBJ").exists():
            n_skip += 1; continue
        if (cd / "video_latent.pt").exists() and not args.overwrite:
            n_skip += 1; continue

        try:
            frames = np.load(cd / "frames.npy")                       # [F_in, H_t, W_t, 3]
            masks_orig = np.load(cd / "masks_orig.npy")               # [K, F_in, H_o, W_o]
            with open(cd / "meta.json") as f:
                meta = json.load(f)

            # 1) VAE encode
            video_latent = encode_video(frames, vae, device, dtype)
            video_latent = video_latent.to(out_dtype)
            C_lat, F_lat, H_lat, W_lat = video_latent.shape

            # 2) mask 对齐
            K, F_in = masks_orig.shape[:2]
            masks_t = temporal_align_masks(masks_orig, F_lat)
            masks_lat = resize_masks_to_latent(masks_t, H_lat, W_lat)

            # 3) patchify volume
            volume_mask = patchify_volume(masks_lat, p_t, p_h, p_w)

            # 4) 首帧 mask
            first_frame_mask = masks_lat[:, 0]

            # 保存
            torch.save(video_latent.contiguous(), cd / "video_latent.pt")
            torch.save(torch.from_numpy(first_frame_mask).float(),
                       cd / "obj_image_masks.pt")
            torch.save(torch.from_numpy(volume_mask).float(),
                       cd / "obj_volume_masks.pt")
            # 额外: 保存 latent 分辨率下的 dense mask 时序 [K, F_lat, H_lat, W_lat]
            # Motion Mask Predictor 训练用的 ground-truth.
            # 不 patch 化是因为 mask predictor 输出要保留像素分辨率.
            # 用 uint8 节省存储 (比 patchified float32 大 ~4x, 但仍可接受).
            torch.save(torch.from_numpy(masks_lat).to(torch.uint8),
                       cd / "obj_dense_masks_lat.pt")

            # 更新 meta
            meta.update({
                "F_lat": int(F_lat), "H_lat": int(H_lat), "W_lat": int(W_lat),
                "F_p": int(volume_mask.shape[1]),
                "H_p": int(volume_mask.shape[2]),
                "W_p": int(volume_mask.shape[3]),
                "C_lat": int(C_lat),
                "vae_model_id": args.vae_model_id,
                "has_dense_masks": True,
                "stage": 3,
            })
            with open(cd / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err < 20:
                print(f"  ✗ {cd.name}: {type(e).__name__}: {e}")

        if (i+1) % 20 == 0:
            dt = time.time() - t0
            print(f"[{i+1}/{len(clips)}] ok={n_ok} skip={n_skip} err={n_err} "
                  f"{(i+1)/max(dt,1):.2f} clips/s")

    print(f"\n[Done] ok={n_ok} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()
