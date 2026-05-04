"""
Stage 4: encode captions → UMT5 embeddings  (diffusers/transformers-based)
==========================================================================

输入:
    Stage 2 产出: {clip_id}/captions.json {phrases: [...], global_prompt: "..."}

输出:
    {clip_id}/verb_embeddings.pt    [K, N_v_max, text_dim] bf16
    {clip_id}/verb_masks.pt         [K, N_v_max] bool
    {clip_id}/global_context.pt     [L_g, text_dim] bf16

依赖: transformers (UMT5EncoderModel) + tokenizers,**不依赖 VACE 仓库**。
    text encoder 与 VACE 训练时用的是同一个 (google/umt5-xxl,Wan-AI 仓库自带)。

模型加载:
    Wan-AI/Wan2.1-VACE-1.3B-diffusers/text_encoder/  (umt5-xxl 的权重)
    Wan-AI/Wan2.1-VACE-1.3B-diffusers/tokenizer/

显存:
    UMT5-XXL bf16 ≈ 11 GB,A100/H100 单卡可,V100/RTX3090 可能要 fp16 (但精度有损)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch


def load_umt5_encoder(model_id: str, device: torch.device, dtype: torch.dtype):
    """
    用 transformers + diffusers 加载 Wan 用的 UMT5-XXL。
    
    diffusers 仓库布局: text_encoder/ 子目录里有完整 UMT5-XXL 权重。
    """
    try:
        from transformers import UMT5EncoderModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "需要 transformers. UMT5EncoderModel 在 transformers >= 4.40 可用。"
        ) from e

    # 优先尝试从 Wan diffusers 仓库的 text_encoder 子目录加载
    print(f"[T5] loading from {model_id} (subfolder='text_encoder')...")
    try:
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder",
            torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, subfolder="tokenizer",
        )
    except Exception as e:
        print(f"[T5] failed loading from {model_id}: {e}")
        print("[T5] falling back to google/umt5-xxl directly...")
        text_encoder = UMT5EncoderModel.from_pretrained(
            "google/umt5-xxl", torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    text_encoder = text_encoder.to(device).eval()
    return text_encoder, tokenizer


@torch.no_grad()
def encode_text_batch(
    texts: List[str],
    text_encoder,
    tokenizer,
    device,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """
    输入 texts,输出 List[Tensor [L_b, D]]。
    
    采用 Wan 训练时的格式:
        - tokenize 成 fixed length (text_len=512 是 Wan 默认)
        - text_encoder.last_hidden_state 作为 embedding
        - 返回 per-sample 截到实际长度的 tensor (没有 padding)
    """
    if len(texts) == 0:
        return []

    # 一次性 tokenize
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    # 编码
    out = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    hidden = out.last_hidden_state                                  # [B, L, D]

    # 按真实长度截断,返回 per-sample tensor
    result = []
    for b in range(hidden.shape[0]):
        true_len = int(attn_mask[b].sum().item())
        result.append(hidden[b, :true_len].contiguous().cpu())
    return result


def pad_to_batch(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """List[Tensor [L_b, D]] → ([K, L_max, D], [K, L_max] bool)"""
    K = len(tensors)
    L_max = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[-1]
    dtype = tensors[0].dtype
    out = torch.zeros(K, L_max, D, dtype=dtype)
    mask = torch.zeros(K, L_max, dtype=torch.bool)
    for k, t in enumerate(tensors):
        out[k, :t.shape[0]] = t
        mask[k, :t.shape[0]] = True
    return out, mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clips_root", required=True)
    p.add_argument("--text_model_id",
                   default="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                   help="HF id 或本地路径,内部应有 subfolder='text_encoder' 和 'tokenizer'")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_length", type=int, default=512,
                   help="tokenize 截断长度,Wan 默认 512")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_clips", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    text_encoder, tokenizer = load_umt5_encoder(args.text_model_id, device, dtype)
    print(f"[Init] text encoder loaded, dtype={dtype}")

    clips_root = Path(args.clips_root)
    clips = sorted([d for d in clips_root.iterdir() if d.is_dir()])
    if args.max_clips > 0:
        clips = clips[:args.max_clips]

    todo = []
    for cd in clips:
        if not (cd / "captions.json").exists(): continue
        if (cd / "verb_embeddings.pt").exists() and not args.overwrite: continue
        todo.append(cd)
    print(f"[Init] {len(todo)} clips to encode")

    n_ok = 0; n_err = 0
    t0 = time.time()
    for i, cd in enumerate(todo):
        try:
            with open(cd / "captions.json") as f:
                caps = json.load(f)
            phrases = caps["phrases"]
            global_prompt = caps["global_prompt"]
            K = len(phrases)
            if K == 0: continue

            # 一次性编码所有短语 + global,省启动开销
            all_texts = phrases + [global_prompt]
            embs = encode_text_batch(
                all_texts, text_encoder, tokenizer, device, args.max_length,
            )

            verb_embs = embs[:K]
            global_emb = embs[K]                                    # [L_g, D]

            verb_pad, verb_mask = pad_to_batch(verb_embs)
            torch.save(verb_pad.contiguous(), cd / "verb_embeddings.pt")
            torch.save(verb_mask.contiguous(), cd / "verb_masks.pt")
            torch.save(global_emb.contiguous(), cd / "global_context.pt")

            with open(cd / "meta.json") as f: meta = json.load(f)
            meta.update({
                "text_dim": int(verb_pad.shape[-1]),
                "N_v_max": int(verb_pad.shape[1]),
                "L_g": int(global_emb.shape[0]),
                "text_model_id": args.text_model_id,
                "stage": 4,
            })
            with open(cd / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err < 20:
                print(f"  ✗ {cd.name}: {type(e).__name__}: {e}")

        if (i+1) % 50 == 0:
            dt = time.time() - t0
            print(f"[{i+1}/{len(todo)}] ok={n_ok} err={n_err} "
                  f"{(i+1)/max(dt,1):.2f} clips/s")

    print(f"\n[Done] ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
