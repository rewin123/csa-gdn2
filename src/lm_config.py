"""~124M-class DeepSeek-V4 (CSA) models at depth, for the real-text comparison.

Three models, param-matched, dense (MoE off), GPT-2 vocab, tied embeddings (so the
budget matches GPT-2-124M):
  DeepSeek : all-CSA stack            (baseline; = toy "A" scaled up)
  C        : [csa, gdn] interleaved   (GDN-2 main-path layers; = toy "C" scaled up)
  PCat     : all-CSA + shared GDN-2 gist (concat coupling; = toy "PCat" scaled up)

  from lm_config import build_lm, tune_ffn_for_target
  ffn, n = tune_ffn_for_target("DeepSeek", target=124_000_000)   # find FFN width
  m = build_lm("PCat", ffn=ffn, device="cuda")
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csa_candidates as cc          # noqa: E402
from csa_model import make_csa_config, count_params  # noqa: E402

GPT2_VOCAB = 50257
GIST = {"DeepSeek": None, "C": None, "PCat": "pcat"}


def deep_layers(candidate, depth):
    if candidate in ("DeepSeek", "PCat"):
        return ["csa"] * depth
    if candidate == "C":                                  # csa/gdn interleave, csa first
        return (["csa", "gdn"] * ((depth + 1) // 2))[:depth]
    raise ValueError(candidate)


def build_lm(candidate, *, depth=12, hidden=768, heads=12, head_dim=64,
             q_lora=384, o_lora=384, index_heads=8, index_head_dim=64,
             ctx=2048, vocab=GPT2_VOCAB, ffn=2048, m=4, sw=128, topk=64,
             gist_dim=128, gist_heads=4, gist_head_dim=64, seed=0, device="cpu"):
    layers = deep_layers(candidate, depth)
    key = f"{candidate}@lm{depth}"
    cc.CANDIDATES[key] = {"layers": layers, "gist": GIST[candidate]}
    cfg = make_csa_config(
        vocab, hidden_size=hidden, num_hidden_layers=depth,
        num_attention_heads=heads, head_dim=head_dim, q_lora_rank=q_lora, o_lora_rank=o_lora,
        index_n_heads=index_heads, index_head_dim=index_head_dim, index_topk=topk,
        compress_rate_csa=m, sliding_window=sw, moe_intermediate_size=ffn,
        max_position_embeddings=ctx, layer_types=[cc._TYPE[k] for k in layers])
    cfg.tie_word_embeddings = True                        # match GPT-2-124M budget
    torch.manual_seed(seed)
    model = cc.CandidateModel(key, cfg, gist_dim=gist_dim, gist_heads=gist_heads,
                              gist_head_dim=gist_head_dim).to(device)
    model.lm.tie_weights()
    return model


def tune_ffn_for_target(candidate, target=124_000_000, *, lo=256, hi=8192, **kw):
    """Binary-search the dense FFN width so count_params ~= target (for DeepSeek; the
    others are then built at the same ffn and reported alongside)."""
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        m = build_lm(candidate, ffn=mid, device="cpu", **kw)
        n = count_params(m)
        del m
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best
