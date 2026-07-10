"""Gate #36: validate the indexer KL recipe at 1M scale BEFORE any 100M spend.

Trains A (=DeepSeek, all-CSA) and C ([csa,gdn]) on the synthetic MV-NIAH task (csa_tasks_v2),
each in two modes:
  frozen : legacy dense->sparse schedule, indexer NEVER trained (first-section baseline)
  kl     : DeepSeek-V3.2/V4 recipe -- indexer trained by KL-to-main-attention (detached),
           dense warm-up then sparse top-k (see deepseek-indexer-kl-recipe)
and reports graded recall. The recipe WORKS if `kl` lifts recall above `frozen`
(first section: CSA frozen ~0.28, STE-floored A ~0.07). Also characterizes DeepSeek vs C.

  python src/csa_kl_validate.py --cands A C --steps 6000 --warmup_steps 1500 --seeds 1
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csa_candidates import build_candidate, tune_moe_for_target  # noqa: E402
from csa_model import count_params                               # noqa: E402
from csa_tasks_v2 import Vocab2, nb_for                          # noqa: E402
from csa_night_v2 import train, RESDIR                           # noqa: E402
from runlog import RunLogger                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", default=["A", "C"])
    ap.add_argument("--modes", nargs="+", default=["frozen", "kl"])
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Q", type=int, default=8)
    ap.add_argument("--sw", type=int, default=8)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--N", type=int, default=48)             # task pairs (fixed; partial zone)
    ap.add_argument("--topk", type=int, default=8)           # sparse selection target
    ap.add_argument("--contention", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--warmup_steps", type=int, default=1500)   # KL dense warm-up steps
    ap.add_argument("--kl_weight", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="kl_validate")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = "cuda"
    os.makedirs(RESDIR, exist_ok=True)
    vocab = Vocab2(n_markers=args.K, n_values=512, n_filler=64)
    nb = nb_for(args.seq)
    pos_weight = (vocab.V - args.Q) / args.Q
    cell = {"m": args.m, "sw": args.sw, "topk": args.topk, "N": nb}
    cfg = {"N": args.N, "K": args.K, "Q": args.Q, "nb": nb, "sw": args.sw,
           "c": args.contention, "topk": args.topk}
    lg = RunLogger(RESDIR, run_name=args.tag)
    lg.reset()
    print(f"[klval] seq={args.seq} nb={nb} cell={cell} cfg N={args.N} topk={args.topk} "
          f"steps={args.steps} warmup={args.warmup_steps} cands={args.cands} modes={args.modes}", flush=True)

    moe = {}
    for name in args.cands:
        inter, n = tune_moe_for_target(name, vocab.size, cell, 1_000_000)
        moe[name] = inter
        print(f"[tune] {name:5s} moe={inter} params={n:,}", flush=True)

    results = {}
    for name in args.cands:
        for mode in args.modes:
            kl = (mode == "kl")
            recs = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                model = build_candidate(name, vocab.size, cell,
                                        moe_intermediate_size=moe[name], seed=seed, device=dev)
                label = f"{name}-{mode}:{seed}"
                print(f"[run] {label} kl={kl} params={count_params(model):,}", flush=True)
                rec = train(model, label, vocab=vocab, cfg=cfg, steps=args.steps, seed=seed,
                            pos_weight=pos_weight, lg=lg, phase="kl_validate",
                            warmup_steps=args.warmup_steps, kl=kl, kl_weight=args.kl_weight)
                recs.append(round(rec, 4))
                del model
                torch.cuda.empty_cache()
            results[f"{name}-{mode}"] = {"recalls": recs, "mean": sum(recs) / len(recs)}
            print(f"[result] {name}-{mode} recall mean={results[f'{name}-{mode}']['mean']:.3f} {recs}", flush=True)
            _dump(args, cell, cfg, results)

    print("\n=== KL VALIDATION SUMMARY ===", flush=True)
    for k, v in results.items():
        print(f"  {k:12s} recall {v['mean']:.3f}", flush=True)
    for name in args.cands:
        fr = results.get(f"{name}-frozen", {}).get("mean")
        klr = results.get(f"{name}-kl", {}).get("mean")
        if fr is not None and klr is not None:
            verdict = "WORKS" if klr > fr + 0.03 else "no lift"
            print(f"  {name}: KL {klr:.3f} vs frozen {fr:.3f}  delta {klr - fr:+.3f}  -> {verdict}", flush=True)
    print(f"wrote {os.path.join(RESDIR, args.tag + '.json')}", flush=True)


def _dump(args, cell, cfg, results):
    out = {"args": vars(args), "cell": cell, "cfg": cfg, "results": results}
    with open(os.path.join(RESDIR, args.tag + ".json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
