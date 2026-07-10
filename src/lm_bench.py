"""Throughput/memory benchmark for the 124M LM trainer — sweep batch/ckpt/compile to
pick the fastest config that fits 24 GB before a long run.

  python src/lm_bench.py --batch 6 --ga 1 --ckpt 1 --compile 0
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_config import build_lm                       # noqa: E402
from data_fineweb import ShardLoader                 # noqa: E402
from muon import build_optimizer                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="DeepSeek")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--ga", type=int, default=1)
    ap.add_argument("--ffn", type=int, default=2293)   # ~124M for DeepSeek
    ap.add_argument("--ckpt", type=int, default=1)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()
    dev = "cuda"
    m = build_lm(a.cand, ffn=a.ffn, ctx=2048, device=dev)
    m.set_ste(False)
    if a.ckpt:
        m.lm.gradient_checkpointing_enable()
    if a.compile:
        m.lm = torch.compile(m.lm)
    opt = build_optimizer(m, opt="muon", lr=2e-3, muon_lr=0.02, gist_lr_mult=0.1)
    tr = ShardLoader("data/fineweb", "train", 2048, a.batch, dev)
    m.train()

    def step():
        opt.zero_grad(set_to_none=True)
        for _ in range(a.ga):
            x, y = tr.batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(x)
                loss = F.cross_entropy(lo.reshape(-1, lo.size(-1)).float(), y.reshape(-1))
            (loss / a.ga).backward()
        opt.step()

    try:
        for _ in range(5):                              # warmup (incl. compile/triton)
            step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        for _ in range(a.steps):
            step()
        torch.cuda.synchronize(); dt = time.time() - t
        tok = a.batch * a.ga * 2048 * a.steps
        print("RESULT batch=%d ga=%d ckpt=%d compile=%d -> %.0f tok/s | peak %.1f GB | %.3f it/s"
              % (a.batch, a.ga, a.ckpt, a.compile, tok / dt, torch.cuda.max_memory_allocated() / 1e9,
                 a.steps / dt), flush=True)
    except torch.OutOfMemoryError:
        print("RESULT batch=%d ga=%d ckpt=%d compile=%d -> OOM" % (a.batch, a.ga, a.ckpt, a.compile), flush=True)


if __name__ == "__main__":
    main()
