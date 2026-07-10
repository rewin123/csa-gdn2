"""Real-text LM training for the ~124M DeepSeek/C/PCat NIAH comparison.

Data = FineWeb prose (~90%) + OpenHermes-2.5 generic instruct (~10%, SFT loss mask) so the
non-instruct base learns to answer in a slot -> enables ZERO/ONE-SHOT NIAH generalization
(we never train the needle; see eval_niah.py). All models share recipe/data/budget.

Logs 4 loss curves for visualization (per eval): fineweb_loss, openhermes_loss,
niah_zero_shot_loss, niah_single_shot_loss; plus per-distance NIAH (CE+EM) and per-position CE
for FineWeb (doc-aligned) and OpenHermes. Everything -> metrics.jsonl + progress.jsonl + out json.

NB: the indexer KL recipe (set_indexer_kl / dense->sparse / indexer_kl_loss) is the
recipe-DEPENDENT part, wired in only after the 1M validation gate passes (see TODO[kl]).

  # smoke (seconds): whole pipeline end-to-end
  python src/lm_train.py --cand PCat --smoke --data data/fineweb --instruct_data data/instruct \
      --out /tmp/lm_smoke.json
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

try:
    import aim                                          # experiment tracker (optional)
except Exception:
    aim = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_config import build_lm                          # noqa: E402
from data_fineweb import ShardLoader                    # noqa: E402
from data_instruct import InstructLoader                # noqa: E402
from eval_position import DocAlignedLoader, evaluate_per_position, EOT  # noqa: E402
import eval_niah                                         # noqa: E402
from muon import build_optimizer                         # noqa: E402
from csa_model import count_params                       # noqa: E402


def lr_mult(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def eval_ce(model, loader, iters):
    """Mean CE over `iters` batches; honors -100 (instruct prompt mask). None if loader is None."""
    if loader is None:
        return None
    was = model.training
    model.eval()
    tot, n = 0.0, 0
    for _ in range(iters):
        x, y = loader.batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   y.reshape(-1), ignore_index=-100)
        if torch.isfinite(loss):
            tot += loss.item(); n += 1
    if was:
        model.train()
    return tot / max(1, n) if n else None


def _niah_curves(summary):
    """score_niah per-(mode,d) summary -> ({mode: mean CE}, per-d rows)."""
    by = {}
    for r in summary:
        by.setdefault(r["mode"], []).append(r["ce"])
    means = {m: sum(v) / len(v) for m, v in by.items()}
    return means, summary


def full_evaluate(model, *, fw_val, instr_val, fw_docaligned, instr_pos, niah_examples,
                  eot, args):
    """All metrics for one eval step -> rich dict (the 4 curves + per-d NIAH + positional x2)."""
    rec = {}
    rec["fineweb_loss"] = eval_ce(model, fw_val, args.eval_iters)
    rec["openhermes_loss"] = eval_ce(model, instr_val, args.eval_iters)
    if niah_examples:
        summ = eval_niah.score_niah(model, niah_examples, "cuda", eot, batch_size=args.niah_bs)
        means, per_d = _niah_curves(summ)
        rec["niah_zero_shot_loss"] = means.get("zero_shot")
        rec["niah_single_shot_loss"] = means.get("one_shot")
        rec["niah_per_d"] = per_d
    else:
        rec["niah_zero_shot_loss"] = rec["niah_single_shot_loss"] = None
        rec["niah_per_d"] = []
    rec["positional_loss_fineweb"] = (
        evaluate_per_position(model, fw_docaligned, args.pos_iters, args.ctx) if fw_docaligned else [])
    rec["positional_loss_openhermes"] = (
        evaluate_per_position(model, instr_pos, args.pos_iters, args.ctx) if instr_pos else [])
    return rec


def run(args):
    assert torch.cuda.is_available()
    dev = "cuda"
    if args.smoke:
        args.depth, args.hidden, args.heads, args.ffn = 4, 256, 4, 512
        args.ctx, args.batch, args.grad_accum, args.total_tokens = 512, 4, 2, 400_000
        args.eval_every, args.eval_iters, args.pos_iters = 20, 4, 2
        args.niah_buckets, args.niah_per_bucket = [128, 256], 4
        args.kl_warmup_tokens = 100_000               # exercise dense->sparse within the tiny smoke
    if args.target_params and not args.smoke:
        from lm_config import tune_ffn_for_target
        args.ffn, npar = tune_ffn_for_target(
            args.cand, args.target_params, depth=args.depth, hidden=args.hidden,
            heads=args.heads, head_dim=args.hidden // args.heads, ctx=args.ctx,
            m=args.m, sw=args.sw, topk=args.topk)
        print(f"[lm] {args.cand} tuned ffn={args.ffn} -> ~{npar:,} params", flush=True)
    model = build_lm(args.cand, depth=args.depth, hidden=args.hidden, heads=args.heads,
                     head_dim=args.hidden // args.heads, ffn=args.ffn, ctx=args.ctx,
                     m=args.m, sw=args.sw, topk=args.topk, seed=args.seed, device=dev)
    model.set_ste(False)                              # indexer trained by KL (validated gate #36), not STE
    if args.kl:
        # checkpointing runs the layer fwd in no_grad -> the indexer's stashed scores lose their
        # graph and the KL can't train it. So KL => no gradient checkpointing (use smaller batch).
        args.grad_ckpt = 0
        print("[lm] indexer KL ON -> gradient checkpointing OFF (use smaller batch)", flush=True)
    if args.grad_ckpt:
        model.lm.gradient_checkpointing_enable()     # ctx attention is memory-quadratic
    start_tokens = 0
    if args.resume:                                   # CONTINUE a run from a saved checkpoint
        model.load_state_dict(torch.load(args.resume, map_location=dev))
        import re
        mm = re.search(r"_(\d+)M\.pt$", os.path.basename(args.resume))
        start_tokens = args.resume_tokens or (int(mm.group(1)) * 1_000_000 if mm else 0)
        print(f"[lm] RESUMED {args.cand} from {args.resume} @ {start_tokens/1e6:.0f}M tokens", flush=True)
    opt = build_optimizer(model, opt="muon", lr=args.lr, muon_lr=args.muon_lr, gist_lr_mult=0.1)
    base = [g["lr"] for g in opt.param_groups]

    tr = ShardLoader(args.data, "train", args.ctx, args.batch, dev, seed=args.seed)
    fw_val = ShardLoader(args.data, "val", args.ctx, args.batch, dev, seed=999)
    instr_tr = instr_val = instr_pos = None
    if args.instruct_data and os.path.isdir(args.instruct_data):
        try:
            instr_tr = InstructLoader(args.instruct_data, "train", args.ctx, args.batch, dev, seed=args.seed)
            instr_val = InstructLoader(args.instruct_data, "val", args.ctx, args.batch, dev, seed=999)
            instr_pos = InstructLoader(args.instruct_data, "val", args.ctx, args.batch, dev, seed=7)
            print(f"[lm] instruct mix ON ({args.instruct_frac:.0%}) from {args.instruct_data}", flush=True)
        except Exception as e:
            print(f"[lm] instruct mix OFF ({e})", flush=True)
    # doc-aligned FineWeb val for clean per-position CE; NIAH filler from FineWeb val
    fw_docaligned = None
    try:
        fw_docaligned = DocAlignedLoader(args.data, "val", args.ctx, args.batch, dev, seed=21, eot=EOT)
    except Exception as e:
        print(f"[lm] doc-aligned fineweb val OFF ({e})", flush=True)
    enc = tiktoken.get_encoding("gpt2")
    filler = None
    try:
        filler = eval_niah.fineweb_filler(args.data, "val", seed=123)
    except Exception as e:
        print(f"[lm] fineweb filler OFF ({e})", flush=True)
    niah_examples = []
    if filler is not None:
        try:
            buckets = [b for b in args.niah_buckets if b < args.ctx - 64]
            niah_examples = eval_niah.make_eval_set(filler, args.ctx, buckets, args.niah_per_bucket,
                                                    modes=("zero_shot", "one_shot"), seed=5)
            print(f"[lm] NIAH eval set: {len(niah_examples)} examples, d={buckets}", flush=True)
        except Exception as e:
            print(f"[lm] NIAH eval OFF ({e})", flush=True)
    probes = _build_probes(enc, filler, args.ctx, EOT) if args.gen else []   # fixed gen watch-prompts

    tok_per_step = args.batch * args.ctx * args.grad_accum
    total_steps = max(1, args.total_tokens // tok_per_step)
    warmup = min(args.warmup, total_steps // 10 + 1)
    kl_warmup_steps = (args.kl_warmup_tokens // tok_per_step) if args.kl else 0
    start_step = start_tokens // tok_per_step          # resume: continue schedule/counter, not from 0
    eot = EOT
    mix_rng = random.Random(args.seed)
    print(f"[lm] {args.cand} params={count_params(model):,} depth={args.depth} hidden={args.hidden} "
          f"ffn={args.ffn} ctx={args.ctx} tok/step={tok_per_step:,} steps={total_steps:,}", flush=True)

    aim_run = None
    if args.aim and aim is not None:
        aim_run = aim.Run(repo=args.aim_repo or None, experiment=args.aim_exp)
        aim_run.name = f"{args.cand}-s{args.seed}"
        aim_run.add_tag(args.cand)
        aim_run["hparams"] = {"cand": args.cand, "seed": args.seed, "ctx": args.ctx, "depth": args.depth,
                              "hidden": args.hidden, "ffn": args.ffn, "kl": bool(args.kl),
                              "instruct_frac": args.instruct_frac, "lr": args.lr, "total_tokens": args.total_tokens}
        print(f"[lm] Aim -> repo={args.aim_repo or '.aim'} exp={args.aim_exp} run={aim_run.name}", flush=True)
    elif args.aim:
        print("[lm] --aim set but 'aim' not installed; skipping Aim", flush=True)

    curve = []
    t0 = time.time()
    last_ckpt = start_tokens
    model.train()
    for step in range(start_step, total_steps):
        if args.ckpt_dir and args.ckpt_every and step * tok_per_step - last_ckpt >= args.ckpt_every:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            mtok = (step * tok_per_step) // 1_000_000
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{args.cand}_{mtok}M.pt"))
            last_ckpt = step * tok_per_step
            print(f"[lm] ckpt {args.cand} @ {mtok}M tokens", flush=True)
        if args.max_minutes and (time.time() - t0) > args.max_minutes * 60:
            print(f"[lm] {args.cand} hit --max_minutes={args.max_minutes}, stop at step {step}", flush=True)
            break
        fr = lr_mult(step, total_steps, warmup)
        for i, g in enumerate(opt.param_groups):
            g["lr"] = base[i] * fr
        if args.kl:
            model.set_indexer_kl(on=True, dense=(step < kl_warmup_steps))  # dense warm-up -> sparse
        opt.zero_grad(set_to_none=True)
        last_loss = 0.0
        for _ in range(args.grad_accum):
            use_instr = instr_tr is not None and mix_rng.random() < args.instruct_frac
            x, y = (instr_tr if use_instr else tr).batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       y.reshape(-1), ignore_index=-100)
            if args.kl:                                   # indexer trained by KL-to-main-attention
                klv = model.indexer_kl_loss()
                if klv is not None:
                    loss = loss + args.kl_weight * klv.to(loss.dtype)
            (loss / args.grad_accum).backward()
            last_loss = loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == total_steps - 1:
            if args.kl:
                model.set_indexer_kl(on=False, dense=False)   # eval at sparse top-k, no KL capture
            ev = full_evaluate(model, fw_val=fw_val, instr_val=instr_val, fw_docaligned=fw_docaligned,
                               instr_pos=instr_pos, niah_examples=niah_examples, eot=eot, args=args)
            toks = (step + 1) * tok_per_step
            sps = (step + 1) / (time.time() - t0)
            ev.update({"step": step, "tokens": toks, "lr": round(base[0] * fr, 6),
                       "train_loss": round(last_loss, 4)})
            curve.append(ev)
            _log(args, total_steps, curve)
            _aim_track(aim_run, ev)
            for name, pids, exp, mx in probes:            # fixed-prompt generations -> Aim text
                txt = enc.decode(_generate(model, pids, mx, eot, dev))
                if aim_run is not None:
                    aim_run.track(aim.Text(txt), name=name, step=step, context={"expect": exp})
                print(f"[gen] {args.cand} {name} exp={exp!r} -> {txt!r}", flush=True)
            fwl, ohl = ev["fineweb_loss"], ev["openhermes_loss"]
            z, o = ev["niah_zero_shot_loss"], ev["niah_single_shot_loss"]
            print(f"[lm] {args.cand} s{step}/{total_steps} tok{toks:,} train{last_loss:.3f} "
                  f"fw{_f(fwl)} oh{_f(ohl)} niah0{_f(z)} niah1{_f(o)} ({sps:.2f} it/s)", flush=True)

    out = {"cand": args.cand, "params": count_params(model),
           "cfg": {"depth": args.depth, "hidden": args.hidden, "ffn": args.ffn, "ctx": args.ctx,
                   "m": args.m, "sw": args.sw, "topk": args.topk, "total_tokens": args.total_tokens,
                   "instruct_frac": args.instruct_frac},
           "final": curve[-1] if curve else {}, "curve": curve}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    if args.ckpt:
        torch.save(model.state_dict(), args.ckpt)
    fv = curve[-1]["fineweb_loss"] if curve else None
    print(f"[lm] DONE {args.cand} fineweb_loss={_f(fv)} wrote {args.out}", flush=True)
    return out


def _f(x):
    return "—" if x is None else f"{x:.3f}"


def _aim_track(run, ev):
    """Log one eval record to Aim: scalars + array metrics as context-tagged lines (group/overlay
    by hparams.cand in the Metrics Explorer; group pos_ce by context.pos, niah by context.dist)."""
    if run is None:
        return
    step = ev["step"]
    for name in ("train_loss", "fineweb_loss", "openhermes_loss",
                 "niah_zero_shot_loss", "niah_single_shot_loss", "lr"):
        v = ev.get(name)
        if v is not None:
            run.track(float(v), name=name, step=step)
    for r in ev.get("niah_per_d", []):
        run.track(float(r["ce"]), name="niah_ce", step=step, context={"mode": r["mode"], "dist": r["d"]})
        run.track(float(r["em"]), name="niah_em", step=step, context={"mode": r["mode"], "dist": r["d"]})
    for tag, key in (("fineweb", "positional_loss_fineweb"), ("openhermes", "positional_loss_openhermes")):
        for r in ev.get(key, []):
            run.track(float(r["ce"]), name="pos_ce", step=step, context={"data": tag, "pos": r["pos"]})


INSTR_PROBE = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"


def _build_probes(enc, sample_filler, ctx, eot):
    """Fixed watch-prompts generated each eval (-> Aim text): a generic instruction + a one-shot
    NIAH example (its best shot). Returns [(name, prompt_ids, expected_text, max_new)]."""
    probes = [("gen_instruct", enc.encode_ordinary(INSTR_PROBE), "Paris", 16)]
    if sample_filler is not None:
        try:
            rng = np.random.default_rng(2024)
            ex = eval_niah.build_example(enc, sample_filler, rng, ctx, d_target=512,
                                         mode="one_shot", eot=eot)
            probes.append(("gen_niah", ex["ids"][:ex["answer_start"]], ex["value"], 8))
        except Exception as e:
            print(f"[lm] NIAH probe OFF ({e})", flush=True)
    return probes


@torch.no_grad()
def _generate(model, prompt_ids, max_new, eot, dev):
    """Greedy autoregressive decode (no KV cache -> keep max_new small)."""
    was = model.training
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=dev)[None]
    gen = []
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt)
        if nxt == eot or ids.shape[1] >= 2048:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    if was:
        model.train()
    return gen


def _log(args, total_steps, curve):
    """metrics.jsonl (full per-eval record) + progress.jsonl (live dashboard, merged per cand)."""
    try:
        d = os.path.dirname(args.out) or "."
        with open(os.path.join(d, f"metrics.{args.cand}.jsonl"), "w") as f:
            for ev in curve:
                f.write(json.dumps({"cand": args.cand, "seed": args.seed, **ev}) + "\n")
        # progress.jsonl: one compact point per eval, merged across cands for the dashboard
        pts = []
        for ev in curve:
            pts.append(json.dumps({
                "kind": "point", "cand": args.cand, "seed": args.seed,
                "step": ev["step"], "total_steps": total_steps,
                "loss": ev["train_loss"], "val_loss": ev["fineweb_loss"],
                "fineweb_loss": ev["fineweb_loss"], "openhermes_loss": ev["openhermes_loss"],
                "niah_zero_shot_loss": ev["niah_zero_shot_loss"],
                "niah_single_shot_loss": ev["niah_single_shot_loss"],
                "recall": ev["fineweb_loss"]}))
        with open(os.path.join(d, f"progress.jsonl.{args.cand}.tmp"), "w") as f:
            f.write("\n".join(pts) + "\n")
        merged = []
        for fn in sorted(os.listdir(d)):
            if fn.startswith("progress.jsonl.") and fn.endswith(".tmp"):
                with open(os.path.join(d, fn)) as f:
                    merged.append(f.read().strip())
        with open(os.path.join(d, "progress.jsonl"), "w") as f:
            f.write("\n".join(m for m in merged if m) + "\n")
    except Exception as e:
        print("[lm] log write failed:", e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["DeepSeek", "C", "PCat"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/fineweb")
    ap.add_argument("--instruct_data", default="data/instruct")
    ap.add_argument("--instruct_frac", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--target_params", type=int, default=0, help="if >0, tune ffn to ~this many params")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--sw", type=int, default=128)        # DeepSeek-V4 default sliding_window
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--kl", type=int, default=1, help="train indexer by KL (DeepSeek-V3.2/V4; forces grad_ckpt off)")
    ap.add_argument("--kl_warmup_tokens", type=int, default=20_000_000, help="dense KL warm-up tokens")
    ap.add_argument("--kl_weight", type=float, default=1.0)
    ap.add_argument("--aim", type=int, default=0, help="log to Aim (pip install aim)")
    ap.add_argument("--aim_repo", default="", help="Aim repo dir (default: ./.aim)")
    ap.add_argument("--aim_exp", default="gatednet-100m", help="Aim experiment name")
    ap.add_argument("--resume", default="", help="checkpoint .pt to CONTINUE training from")
    ap.add_argument("--resume_tokens", type=int, default=0, help="tokens already trained (default: parse _NM.pt)")
    ap.add_argument("--gen", type=int, default=1, help="log fixed-prompt generations (instruct+NIAH) to Aim each eval")
    ap.add_argument("--total_tokens", type=int, default=10_000_000_000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--max_minutes", type=float, default=0, help="wall-clock stop (0=off)")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--pos_iters", type=int, default=8, help="batches for per-position CE")
    ap.add_argument("--niah_buckets", type=int, nargs="+", default=[128, 256, 512, 1024, 1536, 1900])
    ap.add_argument("--niah_per_bucket", type=int, default=16)
    ap.add_argument("--niah_bs", type=int, default=8)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--ckpt_dir", default="", help="dir for periodic checkpoints")
    ap.add_argument("--ckpt_every", type=int, default=10_000_000, help="save a checkpoint every N tokens")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
