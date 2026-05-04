"""
Stage 1: extract clips from SA-V
=================================
把 SA-V 原始 mp4 + masklet json 切成训练用的 clip,每个 clip 输出:
    {clip_id}/
    ├── frames.npy          [F, H, W, 3] uint8 (用于VLM和之后的VAE)
    ├── masks_orig.npy      [K, F, H, W] uint8 (原分辨率,给VLM裁图用)
    ├── meta.json           {video_id, F, H, W, fps, stride, indices, obj_ids}

物体筛选策略:
    按"出现帧数 × 平均面积"打分,取 top-K (K = max_objects, 默认5)
    过滤掉昙花一现的小物体,保留贯穿整段clip的主要物体

为什么要这一步:
    - SA-V 视频长度从几秒到几十秒不等,需要截取固定长度 clip
    - SA-V 标注是 6fps,而我们要 12fps (帧 stride 2),需要对齐
    - 同一段视频可以截出多个 clip (从不同起始帧),数据增强
    - 物体筛选要基于完整 clip 范围,不能逐帧做

可断点续跑:每个 clip 写完才追加到 done.txt,中断后跳过已完成的。
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# =========================================================================
# Video reading (decord 优先, cv2 fallback)
# =========================================================================
def open_video(path: str):
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path)
        return ("decord", vr, len(vr), float(vr.get_avg_fps()))
    except Exception:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"cannot open {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        return ("cv2", cap, n, fps)


def read_frames(handle_tuple, indices: List[int]) -> np.ndarray:
    backend, h, _, _ = handle_tuple
    if backend == "decord":
        return h.get_batch(indices).asnumpy()                      # [F, H, W, 3] RGB
    else:
        import cv2
        out = []
        for i in indices:
            h.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = h.read()
            if not ok:
                raise IOError(f"failed reading frame {i}")
            out.append(fr[:, :, ::-1])                              # BGR → RGB
        return np.stack(out)


def close_video(handle_tuple):
    backend, h, _, _ = handle_tuple
    if backend == "cv2":
        h.release()


# =========================================================================
# RLE decoding
# =========================================================================
def decode_rle_safe(rle, fallback_shape: Tuple[int, int]) -> np.ndarray:
    if rle is None:
        return np.zeros(fallback_shape, dtype=np.uint8)
    try:
        from pycocotools import mask as mu
        m = mu.decode(rle)
        if m.ndim == 3:    # multi-instance,这里只取第一个
            m = m[..., 0]
        return m.astype(np.uint8)
    except Exception:
        return np.zeros(fallback_shape, dtype=np.uint8)


def area_rle(rle) -> int:
    if rle is None: return 0
    try:
        from pycocotools import mask as mu
        return int(mu.area(rle))
    except Exception:
        return 0


# =========================================================================
# 物体筛选: 在选定的 clip 帧范围内打分排序
# =========================================================================
def select_top_k_objects(
    masklet: List[List],
    sampled_indices: List[int],
    k: int,
    min_visibility: float = 0.3,
) -> List[int]:
    """
    返回得分最高的 k 个物体 id (按 score 降序)。
    得分 = sum(area) over sampled_indices,只保留可见率 ≥ min_visibility 的物体。
    可见率 = (该物体在多少 sampled_indices 中 area > 0) / len(sampled_indices)
    """
    if len(masklet) == 0: return []
    n_obj = max(len(m) for m in masklet) if masklet else 0
    if n_obj == 0: return []

    scores = np.zeros(n_obj, dtype=np.float64)
    visible = np.zeros(n_obj, dtype=np.int32)
    F = len(sampled_indices)
    for f in sampled_indices:
        if f >= len(masklet): continue
        per = masklet[f]
        for oi in range(min(n_obj, len(per))):
            a = area_rle(per[oi])
            if a > 0:
                scores[oi] += a
                visible[oi] += 1

    visibility = visible / max(F, 1)
    valid = visibility >= min_visibility
    valid_ids = np.where(valid)[0]
    if len(valid_ids) == 0: return []
    sorted_ids = valid_ids[np.argsort(-scores[valid_ids])]
    return [int(i) for i in sorted_ids[:k]]


# =========================================================================
# 单 clip 处理
# =========================================================================
def process_one_clip(
    video_path: str,
    json_path: str,
    out_dir: Path,
    clip_id: str,
    num_frames: int,
    frame_stride: int,
    start_frame: int,
    max_objects: int,
    min_visibility: float,
    target_size: Tuple[int, int],
) -> Optional[Dict]:
    """
    返回该 clip 的 meta dict,失败返回 None。
    """
    h_t, w_t = target_size
    if (out_dir / "DONE").exists():
        # 已处理过,直接返回 meta
        with open(out_dir / "meta.json") as f:
            return json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 读 mask json
    with open(json_path) as f:
        mdata = json.load(f)
    masklet = mdata["masklet"]
    H_orig = mdata["video_height"]
    W_orig = mdata["video_width"]

    # 视频帧抽样
    vh = open_video(video_path)
    backend, _, total_frames, fps = vh

    span = num_frames * frame_stride
    if start_frame + span > total_frames:
        # clip 越界,截到能取多少算多少 (保证至少有 num_frames * stride 帧)
        if total_frames < span:
            start_frame = 0
            indices = list(range(total_frames))
            # 不够就重复最后一帧
            while len(indices) < num_frames:
                indices.append(indices[-1])
            indices = indices[::frame_stride] if len(indices) >= num_frames * frame_stride \
                      else indices[:num_frames]
        else:
            start_frame = total_frames - span
            indices = list(range(start_frame, start_frame + span, frame_stride))
    else:
        indices = list(range(start_frame, start_frame + span, frame_stride))
    indices = indices[:num_frames]

    try:
        frames = read_frames(vh, indices)                          # [F, H, W, 3]
    finally:
        close_video(vh)

    # resize 到 target_size (BICUBIC)
    if frames.shape[1] != h_t or frames.shape[2] != w_t:
        frames_resized = np.stack([
            np.array(Image.fromarray(f).resize((w_t, h_t), Image.BICUBIC))
            for f in frames
        ])
    else:
        frames_resized = frames

    # 物体筛选 (在原分辨率 mask 空间打分)
    obj_ids = select_top_k_objects(masklet, indices, max_objects, min_visibility)
    if len(obj_ids) == 0:
        # 没有合格物体,标记跳过
        (out_dir / "SKIPPED_NO_OBJ").touch()
        return None

    # 解码所选物体的所有帧 mask (原分辨率,因为 VLM 要在这上面裁图)
    K = len(obj_ids)
    F = len(indices)
    masks_orig = np.zeros((K, F, H_orig, W_orig), dtype=np.uint8)
    for j, oi in enumerate(obj_ids):
        for f_idx, f_orig in enumerate(indices):
            if f_orig >= len(masklet): continue
            if oi >= len(masklet[f_orig]): continue
            masks_orig[j, f_idx] = decode_rle_safe(masklet[f_orig][oi], (H_orig, W_orig))

    # 保存
    np.save(out_dir / "frames.npy", frames_resized)
    np.save(out_dir / "masks_orig.npy", masks_orig)

    meta = {
        "clip_id": clip_id,
        "video_id": Path(video_path).stem,
        "video_path": str(video_path),
        "json_path": str(json_path),
        "F": F, "H_target": h_t, "W_target": w_t,
        "H_orig": H_orig, "W_orig": W_orig,
        "fps": fps, "stride": frame_stride, "start_frame": start_frame,
        "indices": indices, "obj_ids": obj_ids,
        "stage": 1,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    (out_dir / "DONE").touch()
    return meta


# =========================================================================
# 多进程 worker (一个进程处理一个 clip)
# =========================================================================
def _worker(args_tuple):
    try:
        return process_one_clip(*args_tuple), None
    except Exception as e:
        return None, f"{args_tuple[3]}: {type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"


# =========================================================================
# 主流程
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sav_root", required=True,
                   help="SA-V 训练目录,内含 sav_xxxxx.mp4 和 sav_xxxxx_manual/_auto.json")
    p.add_argument("--out_root", required=True, help="Stage 1 输出目录")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--frame_stride", type=int, default=2)
    p.add_argument("--target_size", type=int, nargs=2, default=[480, 832],
                   help="[H, W],与 Wan2.1 默认 480p 一致")
    p.add_argument("--clips_per_video", type=int, default=2,
                   help="每段长视频抽几个 clip (起始帧均匀间隔)")
    p.add_argument("--max_objects", type=int, default=5)
    p.add_argument("--min_visibility", type=float, default=0.3,
                   help="物体在 clip 内至少多少比例帧出现才算有效")
    p.add_argument("--use_auto_masklets", action="store_true",
                   help="也使用 _auto.json (量大但质量参差)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_videos", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    sav_root = Path(args.sav_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 收集所有视频 + json
    videos = sorted(sav_root.glob("*.mp4"))
    if args.max_videos > 0: videos = videos[:args.max_videos]

    tasks = []
    for vp in videos:
        vid = vp.stem
        manual = vp.with_name(f"{vid}_manual.json")
        auto = vp.with_name(f"{vid}_auto.json")
        json_path = manual if manual.exists() else (auto if args.use_auto_masklets and auto.exists() else None)
        if json_path is None: continue

        # 读视频长度,决定能抽多少 clip
        try:
            vh = open_video(str(vp))
            close_video(vh)
            total_frames = vh[2]
        except Exception:
            continue
        span = args.num_frames * args.frame_stride
        if total_frames < args.num_frames:    # 太短直接跳过
            continue

        # 起始帧均匀分布
        n_clips = max(1, min(args.clips_per_video,
                             max(1, total_frames // span)))
        if n_clips == 1:
            starts = [0]
        else:
            stride = max(0, (total_frames - span)) // max(1, n_clips - 1)
            starts = [i * stride for i in range(n_clips)]

        for ci, s in enumerate(starts):
            clip_id = f"{vid}_c{ci:02d}"
            out_dir = out_root / clip_id
            tasks.append((str(vp), str(json_path), out_dir, clip_id,
                          args.num_frames, args.frame_stride, s,
                          args.max_objects, args.min_visibility,
                          tuple(args.target_size)))

    print(f"[Init] {len(videos)} videos → {len(tasks)} clip tasks")

    # 已完成清单
    done_log = out_root / "done.txt"
    done_set = set()
    if done_log.exists():
        with open(done_log) as f:
            done_set = set(line.strip() for line in f if line.strip())
    tasks = [t for t in tasks if t[3] not in done_set]
    print(f"[Resume] {len(done_set)} clips already done, {len(tasks)} remaining")

    # 并行执行
    n_ok = 0; n_skip = 0; n_err = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex, \
         open(done_log, "a") as df:
        futures = {ex.submit(_worker, t): t[3] for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            cid = futures[fut]
            res, err = fut.result()
            if err is not None:
                n_err += 1
                if n_err < 20: print(f"  ✗ {err}")
            elif res is None:
                n_skip += 1
            else:
                n_ok += 1
                df.write(cid + "\n"); df.flush()
            if (i+1) % 100 == 0:
                dt = time.time() - t0
                rate = (i+1) / max(dt, 1)
                print(f"[{i+1}/{len(tasks)}] ok={n_ok} skip={n_skip} err={n_err} "
                      f"{rate:.1f} clips/s")

    print(f"\n[Done] ok={n_ok} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()
