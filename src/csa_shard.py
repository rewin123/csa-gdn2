"""Single (candidate, seed) runner for the §4 v2 regime-S experiment, V4 3-stage
training recipe. Designed to be one process per job so it shards trivially across a
vast.ai / runpod GPU pool (and also runs locally on the box).

V4 dense->indexer-warmup->sparse, 3 stages (default 20k / 10k / 20k = 50k steps):
  1. DENSE   : index_topk non-binding (attend ALL compressed blocks), all params train.
  2. INDEXER : index_topk binding (sparse); FREEZE everything except the Lightning
               Indexer params (`.indexer.` in name) -> warm up the selector alone.
  3. SPARSE  : index_topk binding; all params train -> full sparse fine-tune.
Per-stage cosine LR (each stage warms up + decays). Eval always at the sparse target.

Writes one result JSON: {cand, seed, recall, recall_ablate?, curve, stages, cell}.

  python src/csa_shard.py --cand PCat --seed 0 \
      --dense_steps 20000 --indexer_steps 10000 --sparse_steps 20000 \
      --N 48 --topk 16 --seq 256 --out results_rank7/shard_PCat_0.json
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csa_candidates import build_candidate, tune_moe_for_target  # noqa: E402
from csa_model import count_params  # noqa: E402
from csa_tasks_v2 import Vocab2, make_v2, loss_acc_v2, nb_for, seq_for  # noqa: E402
from muon import build_optimizer  # noqa: E402

EVAL_SEED = 555_001


def cosine_mult(step, total, warmup=50):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


def set_index_topk(model, val):
    for m in model.modules():
        if hasattr(m, "index_topk"):
            m.index_topk = int(val)


def set_trainable(model, indexer_only):
    for n, p in model.named_parameters():
        p.requires_grad = (".indexer." in n) if indexer_only else True


@torch.no_grad()
def evaluate(model, *, vocab, cfg, eval_topk=None, ablate=False, batch=64, nbatch=4):
    was = model.training
    model.eval()
    if eval_topk is not None:                           # else: eval at the CURRENT stage's topk
        set_index_topk(model, eval_topk)               # (dense during warmup, sparse during sparse)
    prev = getattr(model, "_ablate_gist", False)
    if hasattr(model, "_ablate_gist"):
        model._ablate_gist = ablate
    recs = []
    for j in range(nbatch):
        ids, q, t, _ = make_v2(batch, vocab=vocab, n_pairs=cfg["N"], K=cfg["K"], Q=cfg["Q"],
                               nb=cfg["nb"], sliding_window=cfg["sw"], contention=cfg["c"],
                               split="eval", seed=EVAL_SEED + j, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)
            _, r = loss_acc_v2(logits, q, t, vocab, cfg["Q"])
        recs.append(r)
    if hasattr(model, "_ablate_gist"):
        model._ablate_gist = prev
    if was:
        model.train()
    return sum(recs) / len(recs)


def run(args):
    assert torch.cuda.is_available()
    dev = "cuda"
    vocab = Vocab2(n_markers=args.K, n_values=512, n_filler=64)
    nb = nb_for(args.seq)
    dense_topk = nb                                     # non-binding => dense
    sparse_topk = args.topk
    pos_weight = (vocab.V - args.Q) / args.Q
    cell_model = {"m": args.m, "sw": args.sw, "topk": sparse_topk, "N": nb}
    cfg = {"N": args.N, "K": args.K, "Q": args.Q, "nb": nb, "sw": args.sw, "c": args.contention}

    inter, npar = tune_moe_for_target(args.cand, vocab.size, cell_model, 1_000_000)
    torch.manual_seed(args.seed)
    model = build_candidate(args.cand, vocab.size, cell_model, moe_intermediate_size=inter,
                            seed=args.seed, device=dev)
    model.set_ste(True)          # straight-through top-k: LM loss trains the Lightning Indexer
    opt = build_optimizer(model, opt="muon", lr=args.lr, muon_lr=args.muon_lr, gist_lr_mult=0.1)
    base = [g["lr"] for g in opt.param_groups]
    base_seed = (args.seed + 1) * 1_000_003
    print(f"[shard] {args.cand} seed{args.seed} params={count_params(model):,} "
          f"stages dense/idx/sparse={args.dense_steps}/{args.indexer_steps}/{args.sparse_steps} "
          f"cell(N={args.N} topk={sparse_topk} seq={seq_for(nb)})", flush=True)

    stages = [("dense", args.dense_steps, dense_topk, False),
              ("indexer", args.indexer_steps, sparse_topk, True),
              ("sparse", args.sparse_steps, sparse_topk, False)]
    curve = []
    gstep = 0
    rec = 0.0
    model.train()
    for sname, nsteps, topk, idx_only in stages:
        if nsteps <= 0:
            continue
        set_index_topk(model, topk)
        set_trainable(model, indexer_only=idx_only)
        for ls in range(nsteps):
            fr = cosine_mult(ls, nsteps)
            for i, g in enumerate(opt.param_groups):
                g["lr"] = base[i] * fr
            ids, q, t, _ = make_v2(args.batch, vocab=vocab, n_pairs=args.N, K=args.K, Q=args.Q,
                                   nb=nb, sliding_window=args.sw, contention=args.contention,
                                   split="train", seed=base_seed + gstep, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)
                loss, _ = loss_acc_v2(logits, q, t, vocab, args.Q, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            gstep += 1
            if gstep % args.eval_every == 0 or (sname == "sparse" and ls == nsteps - 1):
                rec = evaluate(model, vocab=vocab, cfg=cfg)   # eval at the current stage's mode
                curve.append([gstep, sname, round(loss.item(), 4), round(rec, 4)])
                print(f"[shard] {args.cand} s{args.seed} {sname} gstep{gstep} "
                      f"loss{loss.item():.4f} rec{rec:.3f}", flush=True)

    out = {"cand": args.cand, "seed": args.seed, "recall": rec,
           "stages": {"dense": args.dense_steps, "indexer": args.indexer_steps, "sparse": args.sparse_steps},
           "cell": {"N": args.N, "topk": sparse_topk, "seq": seq_for(nb), "K": args.K, "Q": args.Q,
                    "sw": args.sw, "m": args.m, "contention": args.contention},
           "params": count_params(model), "curve": curve}
    if getattr(model, "use_gist", False):
        out["recall_ablate"] = evaluate(model, vocab=vocab, cfg=cfg, eval_topk=sparse_topk, ablate=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print(f"[shard] DONE {args.cand} seed{args.seed} -> recall {rec:.3f}"
          + (f" ablate {out['recall_ablate']:.3f}" if "recall_ablate" in out else "")
          + f"  wrote {args.out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dense_steps", type=int, default=20000)
    ap.add_argument("--indexer_steps", type=int, default=10000)
    ap.add_argument("--sparse_steps", type=int, default=20000)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Q", type=int, default=8)
    ap.add_argument("--sw", type=int, default=8)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--contention", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
