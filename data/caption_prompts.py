"""
Caption prompt templates & keyframe selection
==============================================

我们要让 VLM 生成两类文本:

1) per-object 动作短语 (motion phrase)
   → 用于训练 PhaseAwareMotionEncoder 的 verb_emb
   → 必须是动名词短语 ("running fast", "jumping over a fence")
   → 不能是名词 ("a dog") 或形容词 ("blue and small")

2) 全局视频 caption (global prompt)  
   → 用于 WanModel 的 context (主干文本条件)
   → 完整句子,涵盖所有物体和场景

关于 Instruct vs Thinking prompts:

Instruct 模型直接执行指令,prompt 越短越好,关键约束清晰即可。
Thinking 模型会先 reason 再回答,prompt 需要:
    - 给一个明确的"思考方向" (否则会跑偏)
    - 强调"在思考完成后,只输出最终答案"
    - Few-shot examples 必不可少 (Thinking 模型在没有例子时容易生成
      解释性文本而非简洁短语)

Thinking 模型还有个特殊技巧: 在 prompt 末尾用 "/no_think" 可以关掉
thinking 行为,让它退化成 Instruct 模式。但这违背了用 Thinking 的初衷,
所以本模块只在确实需要更慢更准时才上 thinking。

关键帧选择 (keyframe selection):
    简单等距抽帧 (首中末) 在动作快速变化的视频上会丢失关键姿态。
    我们用"基于物体重心位移"的策略,选择物体运动差异最大的若干帧。
"""

from typing import List, Tuple
import numpy as np


# =========================================================================
# Prompt templates — Instruct 模式 (Qwen3-VL-2B-Instruct, Qwen2-VL 等)
# =========================================================================
OBJECT_PHRASE_PROMPT_INSTRUCT = (
    "These are keyframes of a single tracked object in a short video, "
    "shown in chronological order. Describe the action of this object "
    "using a concise English phrase in -ing form.\n"
    "\n"
    "Strict rules:\n"
    "- Output ONE phrase, 2 to 8 words.\n"
    "- Must start with a verb in -ing form (e.g., 'running', 'jumping').\n"
    "- Describe the ACTION/MOTION, not appearance, color, or identity.\n"
    "- If the object is mostly static, say 'standing still' or 'staying in place'.\n"
    "- Do NOT include words like 'the object', 'a person', 'this'.\n"
    "- Do NOT add any explanation, just the phrase.\n"
    "\n"
    "Examples of good outputs:\n"
    "  jumping over a fence\n"
    "  running across the field\n"
    "  rotating slowly to the left\n"
    "  walking toward the camera\n"
    "\n"
    "Output the phrase only:"
)

GLOBAL_PROMPT_INSTRUCT = (
    "These are keyframes from a video, shown in chronological order. "
    "Write ONE descriptive English sentence summarizing the whole video. "
    "Include the main subjects and their actions. Keep it under 30 words. "
    "Do not start with 'The video shows' or similar phrases.\n"
    "\n"
    "Output the sentence only:"
)


# =========================================================================
# Prompt templates — Thinking 模式 (Qwen3-VL-2B-Thinking 等)
#
# Thinking 模型必须告诉它:
#   1. 思考过程是什么 (避免在 think 里走神)
#   2. 最终输出格式严格 (避免最终答案变成 "Looking at the keyframes...")
# =========================================================================
OBJECT_PHRASE_PROMPT_THINKING = (
    "Task: Describe the action of a tracked object across video keyframes.\n"
    "\n"
    "How to think:\n"
    "  1. Look at the first and last keyframe to identify the dominant motion.\n"
    "  2. Check intermediate frames to confirm the action type.\n"
    "  3. Pick the most specific verb that captures the motion.\n"
    "  4. Form a 2-8 word phrase starting with that verb in -ing form.\n"
    "\n"
    "Final output rules (STRICT):\n"
    "  - After thinking, output ONLY the phrase on a single line.\n"
    "  - Phrase must start with an -ing verb (running, jumping, etc.).\n"
    "  - 2 to 8 words total.\n"
    "  - No quotes, no explanation, no period at end.\n"
    "  - If the object is mostly static, output: standing still\n"
    "\n"
    "Good final outputs:\n"
    "  jumping over a fence\n"
    "  running across the field\n"
    "  rotating slowly to the left\n"
    "  walking toward the camera\n"
    "\n"
    "Bad final outputs (do not produce these):\n"
    "  'The object is jumping over a fence.'  ← has explanation/quotes\n"
    "  'a dog'                                ← not a verb phrase\n"
    "  'jumps'                                ← not -ing form\n"
    "\n"
    "Now describe the object in the keyframes."
)

GLOBAL_PROMPT_THINKING = (
    "Task: Summarize the video shown in keyframes.\n"
    "\n"
    "How to think:\n"
    "  1. Identify the main subjects.\n"
    "  2. Identify what each subject is doing.\n"
    "  3. Note the setting/scene if relevant.\n"
    "  4. Compose ONE sentence under 30 words.\n"
    "\n"
    "Final output rules (STRICT):\n"
    "  - Exactly ONE sentence, ending with a period.\n"
    "  - Under 30 words.\n"
    "  - Do not start with 'The video shows' or 'In this video'.\n"
    "  - No quotes, no markdown, no list format.\n"
    "\n"
    "Now summarize the video."
)


# =========================================================================
# Prompt selection helper
# =========================================================================
def get_object_phrase_prompt(is_thinking: bool = False) -> str:
    return OBJECT_PHRASE_PROMPT_THINKING if is_thinking else OBJECT_PHRASE_PROMPT_INSTRUCT


def get_global_prompt(is_thinking: bool = False) -> str:
    return GLOBAL_PROMPT_THINKING if is_thinking else GLOBAL_PROMPT_INSTRUCT


# 向后兼容: 旧脚本里的常量名
OBJECT_PHRASE_PROMPT = OBJECT_PHRASE_PROMPT_INSTRUCT
GLOBAL_PROMPT_PROMPT = GLOBAL_PROMPT_INSTRUCT


# =========================================================================
# Keyframe selection (与之前完全一致)
# =========================================================================
def select_keyframes_by_motion(
    object_masks: np.ndarray,                # [F, H, W] uint8 单个物体的mask时序
    n_keyframes: int = 4,
) -> List[int]:
    """
    根据物体重心位移变化选关键帧:
    总是包含首末帧,中间帧选位移变化率最大的几个 (动作转折点)。

    思路:
        1. 计算每帧物体重心 (cx, cy)
        2. 计算相邻帧重心位移 d_t = ||p_t - p_{t-1}||
        3. 在 d_t 中找局部峰值 (动作转折点),优先保留这些
        4. 不够则用等距填充

    Returns:
        排好序的关键帧 index 列表,长度 ≤ n_keyframes
    """
    F = object_masks.shape[0]
    if F <= n_keyframes:
        return list(range(F))

    # 计算重心
    centers = np.zeros((F, 2), dtype=np.float32)
    valid = np.zeros(F, dtype=bool)
    for f in range(F):
        m = object_masks[f]
        if m.sum() < 4: continue
        ys, xs = np.where(m > 0)
        centers[f] = [ys.mean(), xs.mean()]
        valid[f] = True

    if not valid.any():
        return list(np.linspace(0, F - 1, n_keyframes, dtype=int))

    valid_idx = np.where(valid)[0]
    for i in range(F):
        if not valid[i]:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
            centers[i] = centers[nearest]

    # 位移
    disp = np.zeros(F, dtype=np.float32)
    for f in range(1, F):
        disp[f] = np.linalg.norm(centers[f] - centers[f - 1])

    # 锚点: 首+末
    chosen = {0, F - 1}
    n_more = n_keyframes - len(chosen)
    if n_more > 0:
        candidates = [i for i in range(1, F - 1)]
        sorted_cands = sorted(candidates, key=lambda i: -disp[i])
        for c in sorted_cands:
            if all(abs(c - x) >= 2 for x in chosen):
                chosen.add(c)
                if len(chosen) >= n_keyframes: break
    return sorted(chosen)


def crop_object_keyframes(
    frames: np.ndarray,                       # [F, H, W, 3] uint8
    object_masks: np.ndarray,                 # [F, H, W] uint8
    keyframe_indices: List[int],
    pad_ratio: float = 0.15,
    min_size: int = 64,
):
    """
    在每个关键帧上,把物体 bbox 裁出来,padding 周围 pad_ratio 比例。
    返回 List[PIL.Image] (无效帧用全帧代替)。
    """
    from PIL import Image
    H, W = frames.shape[1], frames.shape[2]
    crops = []
    for fi in keyframe_indices:
        m = object_masks[fi]
        if m.sum() < min_size:
            crops.append(Image.fromarray(frames[fi]))
            continue
        ys, xs = np.where(m > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        # padding
        h = y1 - y0; w = x1 - x0
        py = max(int(h * pad_ratio), 12)
        px = max(int(w * pad_ratio), 12)
        y0 = max(0, y0 - py); y1 = min(H, y1 + py)
        x0 = max(0, x0 - px); x1 = min(W, x1 + px)
        # 太小的 bbox 扩到 min_size
        if (y1 - y0) < min_size:
            cy = (y0 + y1) // 2
            y0 = max(0, cy - min_size // 2)
            y1 = min(H, y0 + min_size)
        if (x1 - x0) < min_size:
            cx = (x0 + x1) // 2
            x0 = max(0, cx - min_size // 2)
            x1 = min(W, x0 + min_size)
        crops.append(Image.fromarray(frames[fi, y0:y1, x0:x1]))
    return crops


def select_global_keyframes(F: int, n_keyframes: int = 4) -> List[int]:
    """全局 caption 用等距抽帧即可。"""
    if F <= n_keyframes: return list(range(F))
    return list(np.linspace(0, F - 1, n_keyframes, dtype=int))
