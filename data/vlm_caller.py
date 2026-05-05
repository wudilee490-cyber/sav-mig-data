"""
VLM caller abstraction
=======================
统一不同后端的接口为:
    caller([List[PIL.Image]], prompt: str) -> str

支持的后端 (按推荐度排序):
    1. qwen3_vl      — 本地 transformers 加载 Qwen3-VL (默认推荐, 2B/4B/8B/32B/30B-A3B)
                       支持 Instruct (快) 和 Thinking (慢但推理更细)
    2. openai_api    — OpenAI 兼容 API (vLLM / LMDeploy / SGLang 部署的多模态模型)
    3. qwen2_vl      — 旧版 Qwen2-VL,保留向后兼容
    4. dummy         — 流水线测试用

关于 Qwen3-VL Thinking 模式:
    Thinking 版本输出格式: <think>\\n推理过程\\n</think>\\n\\n实际答案
    本模块的 _strip_thinking() 自动剥离 thinking tag,只返回最终答案。
    
    什么时候用 Thinking:
        - Caption 质量出现明显错误 (识别错物体, 描述错动作) 时切换
        - 复杂多物体场景 (Thinking 帮助消歧)
    什么时候用 Instruct:
        - 默认情况 (caption 任务相对简单, Instruct 已足够)
        - 大规模处理 (Instruct 快 3-5x)

设计要点:
    - 失败重试 (网络错误 / OOM)
    - 输出后处理 (剥离 thinking tag、引号、剪超长、规整格式)
    - 速率限制 (避免打爆远程 API)
"""

import base64
import io
import re
import time
from typing import Callable, List, Optional, Tuple
from PIL import Image


# =========================================================================
# 通用工具: 剥离 Qwen3-Thinking 输出里的 <think>...</think>
# =========================================================================
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", flags=re.DOTALL)


def _strip_thinking(raw: str) -> Tuple[str, Optional[str]]:
    """
    返回 (final_answer, reasoning_or_None)
    
    Qwen3 thinking 输出格式约定:
        <think>\nreasoning content\n</think>\n\nfinal answer
    
    我们只关心 final answer,但保留 reasoning 给调试用。
    如果模型没用 thinking 标签,直接返回原文。
    """
    m = _THINK_RE.search(raw)
    if m:
        reasoning = m.group(1).strip()
        # 取 </think> 后面的内容
        final = raw[m.end():].strip()
        return final, reasoning
    return raw.strip(), None


# =========================================================================
# 后端 1: 本地 Qwen3-VL via transformers (推荐)
# =========================================================================
def make_qwen3_vl_caller(
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str = "cuda",
    dtype_str: str = "bfloat16",
    use_flash_attn: bool = True,
    max_new_tokens: int = 256,           # Thinking 模式需要更多 (默认 thinking 占 200-500)
    enable_thinking_strip: bool = True,  # 自动剥离 thinking tag
):
    """
    本地加载 Qwen3-VL,与 Qwen2-VL API 不同 (Qwen3VLForConditionalGeneration)。

    Thinking vs Instruct:
        模型路径 "Qwen/Qwen3-VL-2B-Thinking" → thinking 模式 (默认 enabled)
        模型路径 "Qwen/Qwen3-VL-2B-Instruct" → 普通模式
        
        Thinking 模型会自动在输出里加 <think>...</think>,本 caller 自动剥离。
    
    显存 (bf16):
        2B  ~5 GB
        4B  ~9 GB
        8B  ~17 GB
        32B ~65 GB (A100-80G 单卡可)
    """
    import torch
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "Qwen3-VL needs transformers >= 4.57.0. "
            "Run: pip install -U 'transformers>=4.57.0'"
        ) from e

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[dtype_str]

    kwargs = dict(
        torch_dtype=dtype,             # transformers 4.57+ 也支持 dtype= 别名
        device_map=device if device == "auto" else None,
    )
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    print(f"[VLM] loading {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    # 检测是否是 Thinking 版
    is_thinking = "thinking" in model_path.lower()
    if is_thinking:
        print(f"[VLM] Thinking model detected — will strip <think> tags from outputs")
        # Thinking 模型给更高 max_new_tokens
        max_new_tokens = max(max_new_tokens, 512)

    @torch.inference_mode()
    def call(imgs: List[Image.Image], prompt: str) -> str:
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # Qwen3-VL 推荐的标准用法 (transformers >= 4.57)
        # apply_chat_template 内部会处理图像 token
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # 显式 to(device) 因为 device_map=None 时 inputs 不会自动到 GPU
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(gen, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

        if enable_thinking_strip:
            final, _ = _strip_thinking(raw)
            return final
        return raw

    # 暴露元信息给上层用 (e.g. 让 02_caption 选择关闭 thinking_strip 来调试)
    call._is_thinking = is_thinking
    call._model_path = model_path
    return call


# =========================================================================
# 后端 2: 远程 OpenAI 兼容 API (vLLM/SGLang 部署)
# =========================================================================
def make_openai_api_caller(
    endpoint: str,
    model: str = "Qwen/Qwen3-VL-2B-Instruct",
    api_key: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 60,
    rate_limit_qps: float = 0.0,
    max_tokens: int = 256,
    enable_thinking_strip: bool = True,
) -> Callable:
    """
    端点格式:
        vLLM:    http://localhost:8000/v1/chat/completions
        SGLang:  http://localhost:30000/v1/chat/completions
        OpenAI:  https://api.openai.com/v1/chat/completions
    
    注意: 远程 API 也可能返回带 <think> 的输出 (取决于服务端是否启用 thinking),
    本 caller 始终尝试剥离。
    """
    import requests
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_call = [0.0]
    min_interval = (1.0 / rate_limit_qps) if rate_limit_qps > 0 else 0.0

    is_thinking = "thinking" in model.lower()
    if is_thinking:
        max_tokens = max(max_tokens, 512)

    def _b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def call(imgs: List[Image.Image], prompt: str) -> str:
        if min_interval > 0:
            wait = min_interval - (time.time() - last_call[0])
            if wait > 0:
                time.sleep(wait)
            last_call[0] = time.time()

        content = [{"type": "text", "text": prompt}]
        for im in imgs:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_b64(im)}"},
            })
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        last_err = None
        for retry in range(max_retries):
            try:
                r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                # vLLM/SGLang 在启用 thinking 时通常会把推理放到 reasoning_content 字段
                # 优先用 content,如果 content 为空再看 reasoning_content
                raw = msg.get("content", "") or ""
                if not raw and "reasoning_content" in msg:
                    raw = msg["reasoning_content"]
                if enable_thinking_strip:
                    final, _ = _strip_thinking(raw)
                    return final
                return raw
            except Exception as e:
                last_err = e
                time.sleep(1.5 ** retry)
        raise RuntimeError(f"VLM API call failed after {max_retries} retries: {last_err}")

    call._is_thinking = is_thinking
    call._model_path = model
    return call


# =========================================================================
# 后端 3: 本地 Qwen2-VL (向后兼容,保留)
# =========================================================================
def make_qwen2_vl_caller(
    model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
    device: str = "cuda",
    dtype_str: str = "bfloat16",
):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[dtype_str]
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)

    @torch.inference_mode()
    def call(imgs: List[Image.Image], prompt: str) -> str:
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=imgs, padding=True,
                            return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(gen, skip_special_tokens=True)[0]

    call._is_thinking = False
    call._model_path = model_path
    return call


# =========================================================================
# 后端 4: dummy (流水线冒烟测试)
# =========================================================================
def make_dummy_caller():
    def call(imgs: List[Image.Image], prompt: str) -> str:
        if "phrase" in prompt.lower() or "-ing" in prompt.lower():
            return "moving across the scene"
        return "A scene shows objects moving in a video."
    call._is_thinking = False
    call._model_path = "dummy"
    return call


# =========================================================================
# 工厂方法
# =========================================================================
def build_caller(backend: str, **kwargs) -> Callable:
    """
    backend ∈ {"qwen3_vl", "qwen2_vl", "openai_api", "dummy"}
    
    Examples:
        # 默认: Qwen3-VL-2B Instruct (快)
        build_caller("qwen3_vl", model_path="Qwen/Qwen3-VL-2B-Instruct")
        
        # Thinking 版本 (慢但更准)
        build_caller("qwen3_vl", model_path="Qwen/Qwen3-VL-2B-Thinking")
        
        # 远程 vLLM (高吞吐)
        build_caller("openai_api",
                     endpoint="http://localhost:8000/v1/chat/completions",
                     model="Qwen/Qwen3-VL-2B-Instruct")
    """
    if backend == "qwen3_vl":
        return make_qwen3_vl_caller(**kwargs)
    elif backend == "qwen2_vl":
        return make_qwen2_vl_caller(**kwargs)
    elif backend == "openai_api":
        return make_openai_api_caller(**kwargs)
    elif backend == "dummy":
        return make_dummy_caller()
    else:
        raise ValueError(f"unknown backend: {backend!r}. "
                         f"Supported: qwen3_vl, qwen2_vl, openai_api, dummy")


# =========================================================================
# 输出后处理 (与之前版本相同,只是放在这里集中)
# =========================================================================
_VERB_PHRASE_PATTERN = re.compile(r"^([a-zA-Z][a-zA-Z\s\-,'/]*)$")


def clean_verb_phrase(raw: str, max_words: int = 10) -> str:
    """
    清洗 VLM 输出的动作短语:
    - 剥离 thinking tag (双保险, caller 已经剥过一次了)
    - 剥离引号、句号、换行
    - 截断到 max_words 个词
    - 移除常见无效前缀 ("the object is" / "this shows" 等)
    
    Thinking 模型偶尔会在 final answer 里继续解释 "Looking at this image, the action is...",
    我们做最大努力提取关键短语。
    """
    s, _ = _strip_thinking(raw)        # 双保险

    s = s.strip().strip("'\"`*").strip()
    s = s.split("\n")[0].strip()        # 只取第一行,避免 thinking 模型的多行解释
    s = re.sub(r"\.+$", "", s).strip()

    # 剥离常见无效前缀
    prefixes = [
        # 通用
        "the object is ", "the object ", "this object is ", "this is ",
        "the action is ", "the action ", "i see ", "the video shows ",
        "this shows ", "the phrase is ", "answer: ", "answer:",
        "phrase: ", "output: ",
        # Thinking 模型常见的 "Looking at this..." 前缀
        "looking at this image ", "looking at the image ",
        "looking at this ", "based on the image ", "based on the keyframes ",
        "in the keyframes ", "in this video ", "the subject is ",
    ]
    for _ in range(3):                  # 多重前缀, 重剥几次
        sl = s.lower()
        changed = False
        for p in prefixes:
            if sl.startswith(p):
                s = s[len(p):].strip()
                changed = True
                break
        if not changed:
            break

    # 截断到 max_words
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])

    # 取出最长合法前缀
    if not _VERB_PHRASE_PATTERN.match(s):
        s = re.sub(r"[^a-zA-Z\s\-,'/]", " ", s).strip()
        s = " ".join(s.split())

    return s.lower() if s else "moving"


def clean_global_caption(raw: str, max_words: int = 35) -> str:
    s, _ = _strip_thinking(raw)
    s = s.strip().strip("'\"`*").strip()
    s = s.split("\n")[0].strip()
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    if not s.endswith("."):
        s = s + "."
    return s if s else "A video showing moving objects."
