"""
Stage 2: caption objects (per-object motion phrases + global prompt)
=====================================================================

输入:
    Stage 1 产出的 clip 目录,内含 frames.npy + masks_orig.npy + meta.json

输出:
    每个 clip 的 captions.json:
    {
        "phrases": ["jumping over a fence", "running fast", ...],   # 长度 = K
        "global_prompt": "A person jumps over a fence as a dog runs alongside.",
        "raw_phrases": [...],          # VLM 原始输出 (用于调试)
        "raw_global": "...",
        "vlm_backend": "qwen3_vl",
        "vlm_model": "Qwen/Qwen3-VL-2B-Instruct",
        "is_thinking": false,
        "n_keyframes_per_obj": [4, 4, ...],
    }

后端选项 (--backend):
    qwen3_vl     本地 transformers 加载 Qwen3-VL (默认, Instruct 推荐)
                 模型路径决定是 Instruct 还是 Thinking 版本:
                   --model_path Qwen/Qwen3-VL-2B-Instruct  (默认, 快, ~5GB)
                   --model_path Qwen/Qwen3-VL-2B-Thinking  (慢但更准, ~5GB)
    openai_api   远程 OpenAI 兼容 API (vLLM/SGLang 部署的多模态模型)
    qwen2_vl     旧版 Qwen2-VL,向后兼容
    dummy        固定字符串,用于流水线冒烟测试

并发策略:
    本地 GPU 模型: 单进程顺序跑 (GPU 抢占问题)
    远程 API: 多进程并发 (--num_workers 4-16)

为什么把 caption 和 frame extract 分开:
    - VLM 推理是流水线最慢的一步 (~3-5s/clip with Qwen3-VL-2B)
    - 失败率较高 (网络抖动 / OOM / 模型偶发输出垃圾)
    - 需要单独重试和质量过滤,不能拖累其他阶段

Qwen3-VL 注意事项:
    - 需要 transformers >= 4.57.0 (会与 VACE 官方上限冲突,需要手动 upgrade)
    - 推荐配合 flash-attn 加速 (5-10x 提速)
    - bf16 加载 ~5GB 显存 (2B 模型),A100/V100/RTX 3090 都跑得动
"""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sav_mig_data.data.vlm_caller import (
    build_caller, clean_verb_phrase, clean_global_caption,
)
from sav_mig_data.data.caption_prompts import (
    get_object_phrase_prompt, get_global_prompt,
    select_keyframes_by_motion, select_global_keyframes,
    crop_object_keyframes,
)


# =========================================================================
# 单 clip 的 caption
# =========================================================================
def caption_one_clip(
    clip_dir: Path,
    vlm_call,
    n_keyframes: int = 4,
    overwrite: bool = False,
    static_motion_threshold: float = 5.0,
) -> Optional[dict]:
    """
    返回 captions dict;若 clip 已经处理过且不覆盖,返回 None。
    
    根据 vlm_call._is_thinking 自动选用对应的 prompt 模板。
    """
    out_path = clip_dir / "captions.json"
    if out_path.exists() and not overwrite:
        return None

    if not (clip_dir / "DONE").exists():
        return None  # Stage 1 没完成

    if (clip_dir / "SKIPPED_NO_OBJ").exists():
        return None

    # 选 prompt (根据 caller 元信息自动适配 Instruct/Thinking)
    is_thinking = getattr(vlm_call, "_is_thinking", False)
    obj_prompt = get_object_phrase_prompt(is_thinking=is_thinking)
    global_prompt_template = get_global_prompt(is_thinking=is_thinking)

    # 读取 Stage 1 产物
    with open(clip_dir / "meta.json") as f:
        meta = json.load(f)
    frames = np.load(clip_dir / "frames.npy")               # [F, H_t, W_t, 3]
    masks_orig = np.load(clip_dir / "masks_orig.npy")       # [K, F, H_o, W_o]

    K = masks_orig.shape[0]
    F = frames.shape[0]

    # frames 是 target_size 分辨率;masks_orig 是 original 分辨率
    # 我们需要把 mask resize 到 frames 分辨率,这样裁框是一致的
    H_t, W_t = frames.shape[1:3]
    H_o, W_o = masks_orig.shape[2:]
    if (H_t, W_t) != (H_o, W_o):
        masks_at_t = np.zeros((K, F, H_t, W_t), dtype=np.uint8)
        for k in range(K):
            for f in range(F):
                m = Image.fromarray(masks_orig[k, f] * 255).resize(
                    (W_t, H_t), Image.NEAREST)
                masks_at_t[k, f] = (np.array(m) > 127).astype(np.uint8)
    else:
        masks_at_t = masks_orig

    # ---- per-object caption ----
    raw_phrases = []
    phrases = []
    n_kf_list = []
    for k in range(K):
        m_k = masks_at_t[k]                                   # [F, H_t, W_t]

        # 静止物体直接给"standing still",不调 VLM 省钱
        ys, xs = np.where(m_k.sum(axis=0) > 0)
        if len(ys) == 0:
            phrases.append("standing still")
            raw_phrases.append("[static-empty]")
            n_kf_list.append(0)
            continue

        # 计算总位移
        centers = []
        for f in range(F):
            mm = m_k[f]
            if mm.sum() < 4: continue
            ys2, xs2 = np.where(mm > 0)
            centers.append([ys2.mean(), xs2.mean()])
        if len(centers) >= 2:
            centers = np.array(centers)
            total_disp = np.linalg.norm(centers[-1] - centers[0]) + \
                         np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)) * 0.1
        else:
            total_disp = 0.0

        if total_disp < static_motion_threshold:
            phrases.append("standing still")
            raw_phrases.append(f"[static-disp={total_disp:.1f}]")
            n_kf_list.append(0)
            continue

        # 选关键帧 + 裁图
        kf_idx = select_keyframes_by_motion(m_k, n_keyframes=n_keyframes)
        n_kf_list.append(len(kf_idx))
        crops = crop_object_keyframes(frames, m_k, kf_idx)

        # 调 VLM
        try:
            raw = vlm_call(crops, obj_prompt)
        except Exception as e:
            raw = f"[error:{e}]"

        raw_phrases.append(raw)
        cleaned = clean_verb_phrase(raw)
        phrases.append(cleaned if cleaned else "moving")

    # ---- global caption ----
    g_idx = select_global_keyframes(F, n_keyframes=n_keyframes)
    g_imgs = [Image.fromarray(frames[i]) for i in g_idx]
    try:
        raw_global = vlm_call(g_imgs, global_prompt_template)
    except Exception as e:
        raw_global = f"[error:{e}]"
    global_prompt_text = clean_global_caption(raw_global)

    captions = {
        "phrases": phrases,
        "global_prompt": global_prompt_text,
        "raw_phrases": raw_phrases,
        "raw_global": raw_global,
        "vlm_backend": getattr(vlm_call, "_backend_name", "unknown"),
        "vlm_model": getattr(vlm_call, "_model_path", "unknown"),
        "is_thinking": is_thinking,
        "n_keyframes_per_obj": n_kf_list,
    }
    with open(out_path, "w") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    return captions


# =========================================================================
# 主流程
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2: caption objects with VLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clips_root", required=True, help="Stage 1 输出目录")
    p.add_argument("--backend", default="qwen3_vl",
                   choices=["qwen3_vl", "qwen2_vl", "openai_api", "dummy"],
                   help="VLM 后端")

    # 本地模型参数
    p.add_argument("--model_path", default="Qwen/Qwen3-VL-2B-Instruct",
                   help="本地后端的 HF id 或路径. "
                        "Qwen/Qwen3-VL-2B-Instruct (默认, 快) | "
                        "Qwen/Qwen3-VL-2B-Thinking (慢但更准) | "
                        "Qwen/Qwen3-VL-4B-Instruct (更大更准)")
    p.add_argument("--device", default="cuda",
                   help="local 后端 device (cuda / cuda:0 / auto / cpu)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--no_flash_attn", action="store_true",
                   help="不用 flash_attention_2 (兼容性差但通用)")

    # 远程 API 参数
    p.add_argument("--api_endpoint", default=None,
                   help="openai_api 后端端点, e.g. http://localhost:8000/v1/chat/completions")
    p.add_argument("--api_model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_qps", type=float, default=0.0,
                   help="远程 API 速率限制 (qps),0=无限制")

    # 通用
    p.add_argument("--n_keyframes", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=1,
                   help="远程 API 后端可调高 (8-16),本地 GPU 后端保持 1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_clips", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()

    # 构造 caller
    if args.backend == "qwen3_vl":
        caller = build_caller(
            "qwen3_vl",
            model_path=args.model_path,
            device=args.device,
            dtype_str=args.dtype,
            use_flash_attn=not args.no_flash_attn,
        )
    elif args.backend == "qwen2_vl":
        caller = build_caller(
            "qwen2_vl",
            model_path=args.model_path,
            device=args.device,
            dtype_str=args.dtype,
        )
    elif args.backend == "openai_api":
        if not args.api_endpoint:
            raise ValueError("--api_endpoint 必须提供")
        caller = build_caller(
            "openai_api",
            endpoint=args.api_endpoint,
            model=args.api_model,
            api_key=args.api_key,
            rate_limit_qps=args.api_qps,
        )
    else:
        caller = build_caller("dummy")

    # 注入后端名给 captions.json 记录
    caller._backend_name = args.backend

    print(f"[Init] backend={args.backend}, model={getattr(caller, '_model_path', 'unknown')}, "
          f"thinking={getattr(caller, '_is_thinking', False)}")

    clips_root = Path(args.clips_root)
    clips = sorted([d for d in clips_root.iterdir() if d.is_dir()])
    if args.max_clips > 0: clips = clips[:args.max_clips]

    # 过滤已完成 (除非 overwrite)
    todo = []
    n_done = 0
    n_skip = 0
    for cd in clips:
        if not (cd / "DONE").exists():
            n_skip += 1; continue
        if (cd / "SKIPPED_NO_OBJ").exists():
            n_skip += 1; continue
        if (cd / "captions.json").exists() and not args.overwrite:
            n_done += 1; continue
        todo.append(cd)
    print(f"[Init] {len(clips)} total, {n_done} captioned, {n_skip} skipped, "
          f"{len(todo)} to process")

    # 执行
    t0 = time.time()
    n_ok = 0; n_err = 0

    if args.backend in ("qwen3_vl", "qwen2_vl") or args.num_workers <= 1:
        # 单进程顺序 (本地 GPU 模型必须这样)
        for i, cd in enumerate(todo):
            try:
                res = caption_one_clip(cd, caller, args.n_keyframes,
                                        overwrite=args.overwrite)
                if res is not None:
                    n_ok += 1
                    if n_ok <= 3:    # 头 3 个打印一下看看 caption 质量
                        print(f"  [sample] {cd.name}: phrases={res['phrases']}")
                        print(f"           global={res['global_prompt'][:80]}")
            except Exception as e:
                n_err += 1
                if n_err < 20:
                    print(f"  ✗ {cd.name}: {type(e).__name__}: {e}")
                    if n_err < 3:
                        traceback.print_exc()
            if (i+1) % 20 == 0:
                dt = time.time() - t0
                rate = (i+1) / max(dt, 1)
                print(f"[{i+1}/{len(todo)}] ok={n_ok} err={n_err} "
                      f"{rate:.2f} clips/s "
                      f"(ETA: {(len(todo)-i-1)/max(rate,0.01)/60:.1f} min)")
    else:
        # 远程 API 用线程池
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(caption_one_clip, cd, caller, args.n_keyframes,
                                  args.overwrite): cd
                       for cd in todo}
            for i, fut in enumerate(as_completed(futures)):
                cd = futures[fut]
                try:
                    res = fut.result()
                    if res is not None:
                        n_ok += 1
                        if n_ok <= 3:
                            print(f"  [sample] {cd.name}: phrases={res['phrases']}")
                except Exception as e:
                    n_err += 1
                    if n_err < 20:
                        print(f"  ✗ {cd.name}: {type(e).__name__}: {e}")
                if (i+1) % 20 == 0:
                    dt = time.time() - t0
                    print(f"[{i+1}/{len(todo)}] ok={n_ok} err={n_err} "
                          f"{(i+1)/max(dt,1):.2f} clips/s")

    print(f"\n[Done] ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
