"""
Stage 5: build training index + train/val split + sanity check
================================================================

输入:
    Stage 1-4 产出的所有 clip 目录

输出:
    {clips_root}/index.json     全部可用样本元信息列表
    {clips_root}/train.json     训练集 split (95%)
    {clips_root}/val.json       验证集 split (5%)
    {clips_root}/_stats.json    数据集统计

每个 entry 字段:
    {
      "clip_id": "sav_000001_c00",
      "video_id": "sav_000001",
      "n_objects": 3,
      "F_p": 21, "H_p": 60, "W_p": 104,
      "C_lat": 16,
      "text_dim": 4096,
      "N_v_max": 8,
      "L_g": 24,
      "phrases": ["jumping", "running", "standing still"],
      "global_prompt": "A video showing ...",
      "has_phrases": true,
      "has_text_emb": true,
      "has_video_latent": true,
    }

完整性检查:
    跳过缺任何关键文件的 clip,记录到 _broken.json 便于排查
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch


REQUIRED_FILES = [
    "video_latent.pt",
    "obj_image_masks.pt",
    "obj_volume_masks.pt",
    "verb_embeddings.pt",
    "verb_masks.pt",
    "global_context.pt",
    "captions.json",
    "meta.json",
]


def check_clip(clip_dir: Path) -> tuple:
    """返回 (ok, entry_or_reason)"""
    if (clip_dir / "SKIPPED_NO_OBJ").exists():
        return False, "skipped_no_obj"
    missing = [f for f in REQUIRED_FILES if not (clip_dir / f).exists()]
    if missing:
        return False, f"missing:{','.join(missing)}"

    # 读 meta + captions 组装 entry
    try:
        with open(clip_dir / "meta.json") as f: meta = json.load(f)
        with open(clip_dir / "captions.json") as f: caps = json.load(f)
        # 抽样验证 tensor shape
        vol = torch.load(clip_dir / "obj_volume_masks.pt", map_location="cpu",
                         weights_only=True)
        ve = torch.load(clip_dir / "verb_embeddings.pt", map_location="cpu",
                        weights_only=True)
        vl = torch.load(clip_dir / "video_latent.pt", map_location="cpu",
                        weights_only=True)
    except Exception as e:
        return False, f"load_err:{type(e).__name__}"

    if vol.shape[0] != ve.shape[0]:
        return False, f"obj_count_mismatch:{vol.shape[0]}!={ve.shape[0]}"
    if vol.shape[0] == 0:
        return False, "zero_objects"

    entry = {
        "clip_id": clip_dir.name,
        "video_id": meta.get("video_id"),
        "n_objects": int(vol.shape[0]),
        "F_p": int(vol.shape[1]),
        "H_p": int(vol.shape[2]),
        "W_p": int(vol.shape[3]),
        "C_lat": int(vl.shape[0]),
        "text_dim": int(ve.shape[-1]),
        "N_v_max": int(ve.shape[1]),
        "L_g": int(meta.get("L_g", 0)),
        "phrases": caps["phrases"],
        "global_prompt": caps["global_prompt"],
        "has_phrases": True,
        "has_text_emb": True,
        "has_video_latent": True,
    }
    return True, entry


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clips_root", required=True)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_objects", type=int, default=1)
    p.add_argument("--exclude_static", action="store_true",
                   help="排除所有物体都是 'standing still' 的 clip")
    return p.parse_args()


def main():
    args = parse_args()
    clips_root = Path(args.clips_root)
    clips = sorted([d for d in clips_root.iterdir() if d.is_dir()])

    entries = []
    broken = []
    for cd in clips:
        ok, res = check_clip(cd)
        if ok:
            if res["n_objects"] < args.min_objects:
                broken.append({"clip_id": cd.name, "reason": "too_few_objects"})
                continue
            if args.exclude_static:
                if all(p.startswith("standing still") or p == "moving"
                       for p in res["phrases"]):
                    broken.append({"clip_id": cd.name, "reason": "all_static"})
                    continue
            entries.append(res)
        else:
            broken.append({"clip_id": cd.name, "reason": res})

    print(f"[Scan] {len(entries)} usable, {len(broken)} broken")

    # 按 video_id 划分 train/val (避免同一段视频既在 train 又在 val)
    video_ids = sorted(set(e["video_id"] for e in entries))
    rng = random.Random(args.seed)
    rng.shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * args.val_ratio))
    val_vids = set(video_ids[:n_val])
    train = [e for e in entries if e["video_id"] not in val_vids]
    val   = [e for e in entries if e["video_id"] in val_vids]

    # 写出
    with open(clips_root / "index.json", "w") as f:
        json.dump(entries, f, indent=1)
    with open(clips_root / "train.json", "w") as f:
        json.dump(train, f, indent=1)
    with open(clips_root / "val.json", "w") as f:
        json.dump(val, f, indent=1)
    with open(clips_root / "_broken.json", "w") as f:
        json.dump(broken, f, indent=1)

    # 统计
    n_obj_dist = Counter(e["n_objects"] for e in entries)
    F_p_dist = Counter(e["F_p"] for e in entries)
    grid_dist = Counter((e["H_p"], e["W_p"]) for e in entries)
    phrase_lens = [len(p.split()) for e in entries for p in e["phrases"]]

    stats = {
        "total_clips": len(entries),
        "broken_clips": len(broken),
        "train_clips": len(train),
        "val_clips": len(val),
        "unique_videos": len(video_ids),
        "n_objects_dist": dict(n_obj_dist),
        "F_p_dist": dict(F_p_dist),
        "grid_dist": {f"{k[0]}x{k[1]}": v for k, v in grid_dist.items()},
        "phrase_word_count": {
            "min": min(phrase_lens) if phrase_lens else 0,
            "max": max(phrase_lens) if phrase_lens else 0,
            "mean": sum(phrase_lens) / max(len(phrase_lens), 1),
        },
    }
    with open(clips_root / "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[Stats]")
    print(f"  total: {stats['total_clips']} (train: {stats['train_clips']}, "
          f"val: {stats['val_clips']})")
    print(f"  unique videos: {stats['unique_videos']}")
    print(f"  n_objects distribution: {dict(n_obj_dist)}")
    print(f"  phrase length: min={stats['phrase_word_count']['min']}, "
          f"max={stats['phrase_word_count']['max']}, "
          f"mean={stats['phrase_word_count']['mean']:.1f}")
    print(f"\n  index.json / train.json / val.json / _stats.json / _broken.json saved.")


if __name__ == "__main__":
    main()
