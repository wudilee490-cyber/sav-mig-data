"""
Caption quality audit — 30 秒判断整个数据集质量
================================================
用法:
    python scripts/06_audit_captions.py --clips_root /data/sav_mig_cache

输出:
    - 总览: 多少 clip, 总 phrase 数
    - 词汇多样性: unique verb 数, top-20 频率 (用来发现 "walking/moving" 占比过高)
    - 静态比例: standing still 占多少
    - 长度分布: phrase 词数分布
    - 全局 vs phrase 一致性: 检查 global_prompt 里是否包含 phrase 里的 verb
    - 物体间区分度: 同一 clip 内 phrase 多样性
    - 抽样: 随机 20 个 clip 完整打印,人工目检

判断标准 (脚本会给绿色/黄色/红色信号):
    🟢 健康: 静态 < 20%, top1 verb < 15%, 物体间区分度高
    🟡 可用: 静态 20-40%, top1 verb 15-25%, 物体间区分度中等
    🔴 风险: 静态 > 40%, top1 verb > 25%, 物体间高度相似
"""

import argparse
import json
import re
import random
from collections import Counter
from pathlib import Path
from typing import List


# 颜色
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"; NC = "\033[0m"


def first_verb(phrase: str) -> str:
    """提取首词 (即 verb in -ing form)"""
    if not phrase: return ""
    return phrase.strip().split()[0].lower() if phrase.split() else ""


def is_static(phrase: str) -> bool:
    p = phrase.strip().lower()
    return p in ("standing still", "staying in place", "remaining still", "moving")


def jaccard_similarity(s1: str, s2: str) -> float:
    """词级 Jaccard,用于物体间相似度"""
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 or not w2: return 0.0
    return len(w1 & w2) / len(w1 | w2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clips_root", required=True)
    p.add_argument("--max_clips", type=int, default=-1, help="只看前 N 个 clip 加速")
    p.add_argument("--show_samples", type=int, default=20)
    p.add_argument("--top_verbs", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    clips_root = Path(args.clips_root)
    cap_files = sorted(clips_root.glob("*/captions.json"))
    if args.max_clips > 0:
        cap_files = cap_files[:args.max_clips]

    print(f"{CYAN}{'=' * 70}")
    print(f" Caption Quality Audit — {len(cap_files)} clips")
    print(f"{'=' * 70}{NC}")

    if not cap_files:
        print(f"{RED}No captions.json found. Run Stage 2 first.{NC}")
        return

    # ---- 收集统计 ----
    all_phrases: List[str] = []
    all_globals: List[str] = []
    static_count = 0
    obj_count_per_clip = []
    intra_clip_similarities = []      # 同 clip 内 phrase 两两相似度
    global_phrase_consistency = []    # global 提到了 phrase 里的 verb 吗
    per_clip_data = []

    for fp in cap_files:
        with open(fp) as f:
            c = json.load(f)
        phrases = c.get("phrases", [])
        global_p = c.get("global_prompt", "")

        all_phrases.extend(phrases)
        all_globals.append(global_p)
        obj_count_per_clip.append(len(phrases))
        for p in phrases:
            if is_static(p): static_count += 1

        # 同 clip 内两两相似度 (只测移动物体)
        moving = [p for p in phrases if not is_static(p)]
        if len(moving) >= 2:
            sims = []
            for i in range(len(moving)):
                for j in range(i+1, len(moving)):
                    sims.append(jaccard_similarity(moving[i], moving[j]))
            intra_clip_similarities.append(sum(sims)/len(sims))

        # global 一致性: 提取 phrase 里的 verb,看 global 提了没
        gl = global_p.lower()
        verbs_in_phrases = {first_verb(p) for p in moving if first_verb(p)}
        if verbs_in_phrases:
            # 同源词检查 (走 walking → walks/walked)
            verb_roots = {v.rstrip('ing') for v in verbs_in_phrases}
            consistent = any(v in gl or vr in gl
                            for v, vr in zip(verbs_in_phrases, verb_roots))
            global_phrase_consistency.append(consistent)

        per_clip_data.append({
            "clip_id": fp.parent.name,
            "phrases": phrases,
            "global": global_p,
        })

    total = len(all_phrases)

    # ---- 1. 总览 ----
    print(f"\n{YELLOW}── 1. 总览 ──{NC}")
    print(f"  Clips: {len(cap_files)}")
    print(f"  Total phrases: {total}")
    print(f"  Phrases per clip: avg={sum(obj_count_per_clip)/len(obj_count_per_clip):.1f}")

    # ---- 2. 静态比例 ----
    print(f"\n{YELLOW}── 2. 静态物体比例 ──{NC}")
    pct_static = static_count / total * 100 if total else 0
    color = GREEN if pct_static < 20 else (YELLOW if pct_static < 40 else RED)
    icon = "🟢" if pct_static < 20 else ("🟡" if pct_static < 40 else "🔴")
    print(f"  {color}{icon} static: {static_count}/{total} ({pct_static:.1f}%){NC}")
    if pct_static > 40:
        print(f"  {RED}  → 太多静态,模型学不到动作.建议:"
              f"提高 02_caption.py 的 static_motion_threshold (默认 5.0,改成 2-3 让更多物体被 VLM 描述)\n"
              f"     或在 Stage 5 加 --exclude_static{NC}")

    # ---- 3. 词汇多样性 ----
    print(f"\n{YELLOW}── 3. 词汇多样性 (verb 频率) ──{NC}")
    verb_counter = Counter()
    for p in all_phrases:
        if is_static(p): continue
        v = first_verb(p)
        if v: verb_counter[v] += 1
    moving_total = sum(verb_counter.values())
    unique_verbs = len(verb_counter)
    print(f"  unique verbs: {unique_verbs}")

    print(f"  Top {args.top_verbs} verbs:")
    for v, n in verb_counter.most_common(args.top_verbs):
        pct = n / moving_total * 100
        print(f"    {v:20s}: {n:5d} ({pct:5.1f}%)")
    if verb_counter:
        top1_pct = verb_counter.most_common(1)[0][1] / moving_total * 100
        color = GREEN if top1_pct < 15 else (YELLOW if top1_pct < 25 else RED)
        icon = "🟢" if top1_pct < 15 else ("🟡" if top1_pct < 25 else "🔴")
        print(f"  {color}{icon} top-1 verb 占比: {top1_pct:.1f}%{NC}")
        if top1_pct > 25:
            print(f"  {RED}  → top-1 verb 主导,数据集多样性低. 建议:\n"
                  f"     - 换更大的 VLM (Qwen3-VL-4B/8B)\n"
                  f"     - 在 caption_prompts.py 加更多 few-shot 例子覆盖不同动作类型{NC}")

    # 完整短语多样性 (不止 verb)
    full_phrases_counter = Counter(p for p in all_phrases if not is_static(p))
    unique_full = len(full_phrases_counter)
    print(f"\n  unique full phrases: {unique_full} / {moving_total} = "
          f"{unique_full/max(moving_total,1)*100:.1f}%")

    # ---- 4. 物体间区分度 (同 clip 内) ----
    print(f"\n{YELLOW}── 4. 物体间区分度 (同 clip 内) ──{NC}")
    if intra_clip_similarities:
        avg_sim = sum(intra_clip_similarities) / len(intra_clip_similarities)
        # 区分度 = 1 - similarity
        diversity = 1 - avg_sim
        color = GREEN if diversity > 0.6 else (YELLOW if diversity > 0.4 else RED)
        icon = "🟢" if diversity > 0.6 else ("🟡" if diversity > 0.4 else "🔴")
        print(f"  平均同 clip phrase 相似度 (Jaccard): {avg_sim:.2f}")
        print(f"  {color}{icon} 物体间区分度: {diversity:.2f}{NC}")
        high_sim_count = sum(1 for s in intra_clip_similarities if s > 0.6)
        print(f"  ({high_sim_count}/{len(intra_clip_similarities)} clip 同质化严重 sim>0.6)")
        if diversity < 0.4:
            print(f"  {RED}  → VLM 看不出不同物体的差别,可能是 mask 裁剪太大或物体本身相似.\n"
                  f"     检查 caption_prompts.py 的 crop_object_keyframes pad_ratio (默认 0.15,降到 0.05){NC}")
    else:
        print(f"  (无可比较对象)")

    # ---- 5. global vs phrase 一致性 ----
    print(f"\n{YELLOW}── 5. global_prompt vs phrases 一致性 ──{NC}")
    if global_phrase_consistency:
        consistent_pct = sum(global_phrase_consistency) / len(global_phrase_consistency) * 100
        color = GREEN if consistent_pct > 60 else (YELLOW if consistent_pct > 40 else RED)
        icon = "🟢" if consistent_pct > 60 else ("🟡" if consistent_pct > 40 else "🔴")
        print(f"  {color}{icon} 一致率: {consistent_pct:.1f}%{NC}")
        print(f"  ({sum(global_phrase_consistency)}/{len(global_phrase_consistency)} clip 的 global 提到了 phrase 里的动作)")
        if consistent_pct < 40:
            print(f"  {RED}  → global 关键帧和 per-object 关键帧抽到不同时间段.\n"
                  f"     这不是 bug,但训练时 global_context 给的语义和 phrase 不匹配,\n"
                  f"     模型可能产生轻微歧义{NC}")

    # ---- 6. 长度分布 ----
    print(f"\n{YELLOW}── 6. Phrase 长度分布 ──{NC}")
    lens = [len(p.split()) for p in all_phrases if not is_static(p)]
    if lens:
        from statistics import mean, median
        print(f"  min={min(lens)}, max={max(lens)}, mean={mean(lens):.1f}, median={median(lens):.0f}")
        too_short = sum(1 for l in lens if l < 2)
        too_long = sum(1 for l in lens if l > 8)
        if too_short:
            print(f"  {YELLOW}  {too_short} phrase 太短 (< 2 词),可能 VLM 输出问题{NC}")
        if too_long:
            print(f"  {YELLOW}  {too_long} phrase 太长 (> 8 词),可能没遵守格式{NC}")

    # ---- 7. 抽样目检 ----
    print(f"\n{YELLOW}── 7. 随机抽样 (人工目检 {args.show_samples} 个) ──{NC}")
    samples = random.sample(per_clip_data, min(args.show_samples, len(per_clip_data)))
    for s in samples:
        print(f"\n  {CYAN}{s['clip_id']}:{NC}")
        for i, p in enumerate(s['phrases']):
            print(f"    obj{i}: {p}")
        print(f"    global: {s['global']}")

    # ---- 总结 ----
    print(f"\n{CYAN}{'=' * 70}")
    print(f" 总结")
    print(f"{'=' * 70}{NC}")

    flags = []
    if pct_static > 40: flags.append("过多静态")
    if verb_counter and verb_counter.most_common(1)[0][1] / moving_total > 0.25:
        flags.append("verb 多样性低")
    if intra_clip_similarities and (1 - sum(intra_clip_similarities)/len(intra_clip_similarities)) < 0.4:
        flags.append("物体间区分度低")

    if not flags:
        print(f"{GREEN}🟢 数据集质量良好,可以放心训练{NC}")
    elif len(flags) == 1:
        print(f"{YELLOW}🟡 单一问题: {flags[0]} - 可以训练,但建议优化后再正式跑全量{NC}")
    else:
        print(f"{RED}🔴 多重问题: {', '.join(flags)} - 强烈建议优化 caption 后再训练{NC}")
        print(f"{RED}   优先级: 换更大 VLM > 调 prompt > 调静态阈值{NC}")


if __name__ == "__main__":
    main()
