"""Autonomous overnight orchestrator for the §4 v2 experiment (Muon).

Regime-parameterized (P or S):
  Stage 1  wall sweep: train baseline A (CSA x2) over the regime's difficulty axis
           (regime P -> sweep N at fixed non-binding topk; regime S -> sweep topk
           at fixed N) -> converged recall per sweep value.
  Stage 2  auto-pick the sweet value: recall in the partial zone (0.4,0.9) closest
           to the middle (0.65); else the value closest to 0.65 (always proceeds).
  Stage 3  rank-7 at the sweet cell, SEED-MAJOR, full candidate set, paired per
           seed, grok@0.8 + gist ablation. Seeds 0..N; whatever finishes is written.

Resumable: state in results_rank7/<tag>_state.json (atomic write); a restart / the
self-restart wrapper skips finished work. Streams to the live dashboard.

  # regime P (pooling-bound): sweep N, non-binding topk, coupled arm WAdd-pool
  python src/csa_night_v2.py --regime P
  # regime S (selection-bound): sweep topk at fixed N, coupled arm WAdd-sel
  python src/csa_night_v2.py --regime S
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
from runlog import RunLogger  # noqa: E402

try:
    import aim  # experiment tracker (optional)
except Exception:
    aim = None

RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_rank7")
GROK = 0.8
EVAL_SEED = 555_001


def _mk_aim_run(args, cand, seed, tag):
    """Create one Aim run per (cand, seed) rank run, or None if disabled/unavailable."""
    if not args.aim or aim is None:
        return None
    try:
        r = aim.Run(repo=args.aim_repo or None, experiment=args.aim_exp)
        r.name = f"{tag}-{cand}-s{seed}"
        r.add_tag(cand)
        r.add_tag("kl" if args.kl else "frozen")
        r["hparams"] = {"cand": cand, "seed": seed, "regime": args.regime, "kl": bool(args.kl),
                        "kl_weight": args.kl_weight, "warmup_steps": args.warmup_steps,
                        "rank_steps": args.rank_steps, "topk": args.force_sweet, "N": args.fixed_N,
                        "seq": args.seq, "m": args.m, "sw": args.sw,
                        "model_sw": args.model_sw if args.model_sw > 0 else args.sw}
        return r
    except Exception as e:
        print(f"[aim] run create failed: {type(e).__name__}: {e}", flush=True)
        return None


def cosine_mult(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


def save_state(s, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, path)


def load_state(path):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return None


@torch.no_grad()
def evaluate(model, *, vocab, cfg, ablate=False, batch=256, nbatch=2):
    was = model.training
    model.eval()
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


def set_index_topk(model, val):
    """Mutate the live top-k on every Lightning Indexer (stock + gist). index_topk is
    read per-forward as min(index_topk, compressed_len) and sizes no buffers, so this
    is safe to change each step -- enables a dense(non-binding)->sparse schedule."""
    n = 0
    for m in model.modules():
        if hasattr(m, "index_topk"):
            m.index_topk = int(val)
            n += 1
    return n


def train(model, label, *, vocab, cfg, steps, seed, pos_weight, lg, phase,
          lr=2e-3, muon_lr=0.02, batch=64, warmup=50, eval_every=200, warmup_steps=0,
          kl=False, kl_weight=1.0, aim_run=None):
    opt = build_optimizer(model, opt="muon", lr=lr, muon_lr=muon_lr, gist_lr_mult=0.1)
    base = [g["lr"] for g in opt.param_groups]
    model.train()
    base_seed = (seed + 1) * 1_000_003
    rec = 0.0
    # kl=True  -> DeepSeek-V3.2/V4 recipe: indexer trained by KL-to-main-attention (detached),
    #             dense warm-up (attend all blocks) for `warmup_steps`, then sparse top-k.
    # kl=False -> legacy Mode B: index_topk non-binding (dense) -> sparse, indexer NOT trained.
    sparse_topk = cfg.get("topk")
    dense_topk = cfg["nb"]                       # >= num compressed blocks => attend all
    for step in range(steps):
        dense = step < warmup_steps
        if kl:
            model.set_indexer_kl(on=True, dense=dense)   # dense_mode handles warm-up density
            set_index_topk(model, sparse_topk)
        elif warmup_steps > 0:
            set_index_topk(model, dense_topk if dense else sparse_topk)
        fr = cosine_mult(step, steps, warmup)
        for i, g in enumerate(opt.param_groups):
            g["lr"] = base[i] * fr
        ids, q, t, _ = make_v2(batch, vocab=vocab, n_pairs=cfg["N"], K=cfg["K"], Q=cfg["Q"],
                               nb=cfg["nb"], sliding_window=cfg["sw"], contention=cfg["c"],
                               split="train", seed=base_seed + step, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)
            loss, _ = loss_acc_v2(logits, q, t, vocab, cfg["Q"], pos_weight=pos_weight)
        if kl:
            klv = model.indexer_kl_loss()
            if klv is not None:
                loss = loss + kl_weight * klv.to(loss.dtype)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            if kl:
                model.set_indexer_kl(on=False, dense=False)   # eval at sparse, no KL capture
            set_index_topk(model, sparse_topk)
            rec = evaluate(model, vocab=vocab, cfg=cfg)
            gamma = None
            if getattr(model, "use_gist", False):
                try:
                    gamma = round(model.usage_report().get("gamma_absmean", 0.0), 5)
                except Exception:
                    gamma = None
            lg.point(cand=label, seed=seed, step=step + 1, total_steps=steps,
                     recall=rec, loss=round(loss.item(), 4), gamma=gamma)
            lg.status(phase=phase, cand=label, seed=seed,
                      note=f"{label} s{seed} step{step+1}/{steps} rec{rec:.3f}")
            if aim_run is not None:
                try:
                    ctx = {"cand": label, "seed": seed, "phase": phase}
                    aim_run.track(float(rec), name="recall", step=step + 1, context=ctx)
                    aim_run.track(float(loss.item()), name="loss", step=step + 1, context=ctx)
                    if gamma is not None:
                        aim_run.track(float(gamma), name="gamma", step=step + 1, context=ctx)
                    if kl:
                        kv = model.indexer_kl_loss()
                        if kv is not None:
                            aim_run.track(float(kv.item()), name="kl", step=step + 1, context=ctx)
                except Exception:
                    pass
    return rec


def stats(xs):
    n = len(xs)
    if not n:
        return {"n": 0}
    xs2 = sorted(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {"n": n, "mean": round(mean, 4), "std": round(var ** 0.5, 4),
            "median": round(xs2[(n - 1) // 2], 4),
            "grok_rate": round(sum(1 for x in xs if x >= GROK) / n, 4),
            "derail_rate": round(sum(1 for x in xs if x < 0.15) / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["P", "S"], default="P")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Q", type=int, default=8)
    ap.add_argument("--sw", type=int, default=8)
    ap.add_argument("--model_sw", type=int, default=0,
                    help="model attention window (0=use --sw). Set >= seq to make swa layers "
                         "FULL attention while the TASK keeps needle placement at --sw "
                         "(decouples linear+full baseline from the sliding-window task geometry).")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--contention", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8000)         # Stage-1 sweep steps
    ap.add_argument("--rank_steps", type=int, default=10000)    # Stage-3 final rank-7 steps
    ap.add_argument("--warmup_steps", type=int, default=0,
                    help="Mode B (V4 dense->sparse): #steps with index_topk non-binding (dense) "
                         "before dropping to the cell's binding topk. 0 = sparse-from-scratch (Mode A).")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed_list", type=int, nargs="+", default=None,
                    help="Explicit list of seeds to run (overrides --seeds). Lets several "
                         "processes split the seed set to run concurrently on one GPU.")
    ap.add_argument("--sweep", choices=["N", "topk"], default=None)
    ap.add_argument("--sweep_vals", type=int, nargs="+", default=None)
    ap.add_argument("--fixed_N", type=int, default=48)        # for topk-sweep (regime S)
    ap.add_argument("--sweet_target", type=float, default=None,
                    help="target A-recall for the sweet cell (default 0.65 for P, 0.2 for S). "
                         "S's ceiling is the non-binding recall (~0.5), so a partial baseline "
                         "with headroom is ~0.2, not 0.5.")
    ap.add_argument("--force_sweet", type=int, default=None,
                    help="skip the sweep entirely; use this topk (S) / N (P) as the sweet cell.")
    ap.add_argument("--cands", nargs="+", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--kl", type=int, default=0,
                    help="1 = DeepSeek-V3.2/V4 indexer-KL recipe (indexer trained by KL-to-main-attn, "
                         "dense warm-up for --warmup_steps then sparse top-k). 0 = legacy (indexer frozen).")
    ap.add_argument("--kl_weight", type=float, default=1.0)
    ap.add_argument("--aim", type=int, default=0, help="log to Aim (pip install aim)")
    ap.add_argument("--aim_repo", default="", help="Aim repo (dir or aim://host:port); default ./.aim")
    ap.add_argument("--aim_exp", default="csa-rank7", help="Aim experiment name")
    ap.add_argument("--resdir", default="", help="override results dir (for parallel runs; each "
                    "process gets its own progress/status/state files). Default: ../results_rank7.")
    args = ap.parse_args()

    global RESDIR
    if args.resdir:
        RESDIR = args.resdir

    # regime presets (overridable)
    if args.regime == "P":
        sweep = args.sweep or "N"
        sweep_vals = args.sweep_vals or [16, 32, 48, 64, 96]
        cands = args.cands or ["A", "B", "C", "SWA-A", "SWA-B", "PCat", "WAdd-pool"]
        tag = args.tag or "night_v2"
    else:  # S
        sweep = args.sweep or "topk"
        sweep_vals = args.sweep_vals or [32, 16, 12, 8, 6]
        cands = args.cands or ["A", "B", "C", "SWA-A", "SWA-B", "PCat", "WAdd-sel"]
        tag = args.tag or "night_v2_S"
    sweet_target = args.sweet_target if args.sweet_target is not None else (0.2 if args.regime == "S" else 0.65)

    assert torch.cuda.is_available()
    dev = "cuda"
    os.makedirs(RESDIR, exist_ok=True)
    state_path = os.path.join(RESDIR, f"{tag}_state.json")
    vocab = Vocab2(n_markers=args.K, n_values=512, n_filler=64)
    nb = nb_for(args.seq)
    nonbinding_topk = nb
    pos_weight = (vocab.V - args.Q) / args.Q
    SEEDS = args.seed_list if args.seed_list is not None else list(range(args.seeds))

    def cell_for(sweep_val):
        """Return (cell_model for build_candidate, n_pairs) for a sweep value."""
        if sweep == "N":
            topk = nonbinding_topk
            n_pairs = sweep_val
        else:  # topk
            topk = sweep_val
            n_pairs = args.fixed_N
        model_sw = args.model_sw if args.model_sw > 0 else args.sw
        return {"m": args.m, "sw": model_sw, "topk": topk, "N": nb}, n_pairs

    lg = RunLogger(RESDIR, run_name=tag)
    state = load_state(state_path)
    if state is None:
        lg.reset()
        state = {"regime": args.regime, "sweep_axis": sweep, "cell": {
            "seq": args.seq, "nb": nb, "K": args.K, "Q": args.Q, "sw": args.sw, "m": args.m,
            "contention": args.contention, "steps": args.steps, "rank_steps": args.rank_steps,
            "fixed_N": args.fixed_N, "warmup_steps": args.warmup_steps,
            "nonbinding_topk": nonbinding_topk, "sweet_target": sweet_target,
            "gdn_recipe": "AdamW 4e-4 wd0.1"},
            "sweep": {}, "sweet": None, "moe": {}, "runs": {}, "summary": {}}
        save_state(state, state_path)
    print(f"[night] regime={args.regime} sweep={sweep} vals={sweep_vals} seq={args.seq} nb={nb} "
          f"fixed_N={args.fixed_N} cands={cands} steps={args.steps} tag={tag}", flush=True)

    # tune moe once per candidate (param count is N/topk-independent)
    base_cell, _ = cell_for(sweep_vals[0])
    for name in ["A"] + cands:
        if name not in state["moe"]:
            inter, n = tune_moe_for_target(name, vocab.size, base_cell, 1_000_000)
            state["moe"][name] = inter
            save_state(state, state_path)
            print(f"[tune] {name:9s} moe={inter} params={n:,}", flush=True)

    # ---------------- Stage 1: wall sweep on A (skipped if --force_sweet) ---------------- #
    if args.force_sweet is not None and state["sweet"] is None:
        state["sweet"] = args.force_sweet
        save_state(state, state_path)
        print(f"[night] FORCED sweet {sweep}={args.force_sweet} (skipping sweep)", flush=True)
    for val in ([] if args.force_sweet is not None else sweep_vals):
        key = f"{sweep}{val}"
        if key in state["sweep"]:
            continue
        cm, n_pairs = cell_for(val)
        torch.manual_seed(0)
        model = build_candidate("A", vocab.size, cm, moe_intermediate_size=state["moe"]["A"], seed=0, device=dev)
        cfg = {"N": n_pairs, "K": args.K, "Q": args.Q, "nb": nb, "sw": args.sw,
               "c": args.contention, "topk": cm["topk"]}
        print(f"[sweep] A@{key} (topk={cm['topk']} N={n_pairs}) params={count_params(model):,}", flush=True)
        rec = train(model, f"sweepA@{key}", vocab=vocab, cfg=cfg, steps=args.steps,
                    seed=0, pos_weight=pos_weight, lg=lg, phase=f"sweep_{args.regime}")
        state["sweep"][key] = rec
        save_state(state, state_path)
        print(f"[sweep] A@{key} -> recall {rec:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---------------- Stage 2: pick sweet value ---------------- #
    if state["sweet"] is None:
        sw = {int(k[len(sweep):]): v for k, v in state["sweep"].items()}
        # pick the sweep value whose A-recall is closest to sweet_target, among values
        # that are LEARNABLE (above floor, chance~0.016) -> headroom for the gist to help.
        learnable = {val: r for val, r in sw.items() if r >= 0.1}
        if learnable:
            pick = min(learnable, key=lambda v: abs(learnable[v] - sweet_target))
        else:
            pick = max(sw, key=lambda v: sw[v])     # nothing learnable -> least-floored
        state["sweet"] = pick
        save_state(state, state_path)
        print(f"[night] SWEET {sweep}={pick} (A recall {sw[pick]:.3f}, target {sweet_target}) "
              f"learnable={ {k: round(v,3) for k,v in learnable.items()} } sweep={sw}", flush=True)
    sweet = state["sweet"]
    cm_rank, n_pairs_rank = cell_for(sweet)
    cfg = {"N": n_pairs_rank, "K": args.K, "Q": args.Q, "nb": nb, "sw": args.sw,
           "c": args.contention, "topk": cm_rank["topk"]}

    # ---------------- Stage 3: rank-7 (seed-major, paired) ---------------- #
    total = len(SEEDS) * len(cands)
    for seed in SEEDS:
        for name in cands:
            rk = f"{name}:{seed}"
            if rk in state["runs"]:
                continue
            try:
                torch.manual_seed(seed)
                model = build_candidate(name, vocab.size, cm_rank,
                                        moe_intermediate_size=state["moe"][name], seed=seed, device=dev)
                done = len(state["runs"])
                print(f"[rank] {name} seed{seed} ({sweep}={sweet}) {done}/{total} "
                      f"[{args.rank_steps} steps, warmup={args.warmup_steps} dense, kl={bool(args.kl)}]", flush=True)
                arun = _mk_aim_run(args, name, seed, tag)
                rec = train(model, name, vocab=vocab, cfg=cfg, steps=args.rank_steps, seed=seed,
                            pos_weight=pos_weight, lg=lg, phase=f"rank7_{args.regime}",
                            warmup_steps=args.warmup_steps, kl=bool(args.kl),
                            kl_weight=args.kl_weight, aim_run=arun)
                row = {"cand": name, "seed": seed, "recall": rec}
                if getattr(model, "use_gist", False):
                    row["recall_ablate"] = evaluate(model, vocab=vocab, cfg=cfg, ablate=True)
                    try:
                        row["usage"] = model.usage_report()
                    except Exception:
                        pass
                state["runs"][rk] = row
                by = {}
                for r in state["runs"].values():
                    by.setdefault(r["cand"], []).append(r["recall"])
                state["summary"] = {c: stats(v) for c, v in by.items()}
                save_state(state, state_path)
                print(f"[rank] {name} seed{seed} -> recall {rec:.3f}"
                      + (f"  ablate {row['recall_ablate']:.3f}" if "recall_ablate" in row else ""), flush=True)
                if arun is not None:
                    try:
                        arun["final"] = {"recall": rec, **({"recall_ablate": row["recall_ablate"]}
                                                            if "recall_ablate" in row else {})}
                        arun.close()
                    except Exception:
                        pass
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[rank] {name} seed{seed} FAILED: {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()

    print("\n=== NIGHT v2 DONE (regime %s) ===" % args.regime, flush=True)
    for c, s in state.get("summary", {}).items():
        print(f"  {c:10s} grok@0.8={s.get('grok_rate')} mean={s.get('mean')}±{s.get('std')} n={s.get('n')}", flush=True)
    lg.status(phase=f"night_{args.regime}-done", note="all seeds complete")


if __name__ == "__main__":
    main()
