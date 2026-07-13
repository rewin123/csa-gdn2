"""Honest prefill-FLOPs accounting for A (all-CSA) vs C (CSA+GDN-2) at 1M / 10M / 124M.

Method: torch.utils.flop_counter.FlopCounterMode hooks every aten op, so it counts
CSA's compressor, the Lightning Indexer, sliding-window attention, the dense MLP and
the LM head EXACTLY (all eager PyTorch). The ONE thing it cannot see is GDN-2's chunk
recurrence (a Triton kernel), so we add that analytically as an explicit term and label
it as such. We report total prefill FLOPs and the C/A ratio across a context sweep, at
the model's native ctx and beyond, so the ratio's growth with context is visible.

FLOP convention: FlopCounterMode counts a (m,k)x(k,n) matmul as 2*m*n*k (mul+add). We
use the same 2x convention for the analytic GDN-2 term so totals are comparable. The C/A
RATIO is independent of this convention.

Run (remote, RTX 3090):
  python csa_flops.py --out csa_flops.json
"""
import sys, json, argparse
sys.path.insert(0, "src")
import torch
from torch.utils.flop_counter import FlopCounterMode
from lm_config import build_lm
from csa_candidates import build_candidate, tune_moe_for_target
from csa_tasks_v2 import Vocab2

dev = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ctxs_lm", type=str, default="512,1024,2048,4096")
ap.add_argument("--ctxs_toy", type=str, default="128,256,512,1024")
a = ap.parse_args()


def gdn2_recurrence_flops(n_gdn_layers, L, n_heads=4, d_k=32, d_v=32):
    """Analytic FLOPs for the GDN-2 state recurrence that the Triton kernel hides from
    FlopCounterMode. Per token, per head, the fixed-size state S in R^{d_v x d_k} is:
      (1) decayed:  Diag(e^g) S           ~ d_v*d_k mul
      (2) erased:   (I - k (b*k)^T) S     ~ 2*d_v*d_k (outer + matvec)
      (3) written:  + k (w*v)^T           ~ d_v*d_k
      (4) read out: o = S q               ~ d_v*d_k mul-add
    => ~5*d_v*d_k MACs/token/head; x2 for FLOPs, x L, x heads, x layers.
    The q/k/v/b/w/g/o projections and short-conv are nn.Linear/conv1d and ARE counted by
    FlopCounterMode separately; only the state recurrence above is added here.
    """
    macs = 5 * d_v * d_k * n_heads * L * n_gdn_layers
    return 2 * macs


@torch.no_grad()
def count_prefill_flops(model, L, vocab):
    x = torch.randint(0, vocab, (1, L), device=dev)
    fcm = FlopCounterMode(display=False)
    with fcm:
        model(x)
    return int(fcm.get_total_flops())


def n_gdn_in(candidate, depth):
    if candidate == "C":
        return len([1 for i in range(depth) if i % 2 == 1])   # csa,gdn,csa,gdn... -> odd idx
    return 0


results = {"convention": "matmul=2mnk; GDN-2 recurrence added analytically (Triton, unseen by counter)",
           "scales": {}}

# ----- 1M toy -----
print("=== 1M toy ===", flush=True)
voc = Vocab2(); V = voc.size
cell = {"m": 4, "sw": 8, "topk": 16, "N": 48}
toy = {}
for cand, label in [("A", "A"), ("C", "C")]:
    inter, npar = tune_moe_for_target(cand, V, cell, 1_000_000)
    toy[label] = {"moe": inter, "params": npar, "by_ctx": {}}
    print(f"[toy {label}] moe={inter} params={npar:,}", flush=True)
for L in [int(x) for x in a.ctxs_toy.split(",")]:
    for cand, label in [("A", "A"), ("C", "C")]:
        try:
            m = build_candidate(cand, V, cell, moe_intermediate_size=toy[label]["moe"],
                                seed=0, device=dev).to(torch.bfloat16).eval()
            counted = count_prefill_flops(m, L, V)
            gdn = gdn2_recurrence_flops(n_gdn_in(cand, 2), L)
            toy[label]["by_ctx"][str(L)] = {"counted": counted, "gdn2_recur": gdn, "total": counted + gdn}
            del m; torch.cuda.empty_cache()
            print(f"  toy {label} L{L}: total {(counted+gdn)/1e9:.3f} GFLOP", flush=True)
        except RuntimeError as e:
            toy[label]["by_ctx"][str(L)] = {"error": str(e)[:80]}
            torch.cuda.empty_cache()
results["scales"]["1M"] = toy

# ----- 10M and 124M LM -----
for scale, kw, ffns in [
    ("10M", dict(depth=6, hidden=384, heads=6, head_dim=64), {"A": 565, "C": 862}),
    ("124M", dict(depth=12, hidden=768, heads=12, head_dim=64), None),
]:
    print(f"=== {scale} ===", flush=True)
    if ffns is None:  # tune EACH stack's FFN to 124M separately (param-matched, as trained)
        from lm_config import tune_ffn_for_target
        ffnA, nA = tune_ffn_for_target("DeepSeek", target=124_000_000, **kw)
        ffnC, nC = tune_ffn_for_target("C", target=124_000_000, **kw)
        ffns = {"A": ffnA, "C": ffnC}
        print(f"[{scale}] tuned ffnA={ffnA} (~{nA:,})  ffnC={ffnC} (~{nC:,})", flush=True)
    sc = {}
    for cand, label in [("DeepSeek", "A"), ("C", "C")]:
        m0 = build_lm(cand, ffn=ffns[label], device="cpu", **kw)
        from csa_model import count_params
        sc[label] = {"ffn": ffns[label], "params": count_params(m0), "by_ctx": {}}
        del m0
        print(f"[{scale} {label}] ffn={ffns[label]} params={sc[label]['params']:,}", flush=True)
    for L in [int(x) for x in a.ctxs_lm.split(",")]:
        for cand, label in [("DeepSeek", "A"), ("C", "C")]:
            try:
                m = build_lm(cand, ffn=ffns[label], device=dev, **kw).to(torch.bfloat16).eval()
                counted = count_prefill_flops(m, L, 50257)
                gdn = gdn2_recurrence_flops(n_gdn_in("C" if label == "C" else "A", kw["depth"]), L)
                sc[label]["by_ctx"][str(L)] = {"counted": counted, "gdn2_recur": gdn, "total": counted + gdn}
                del m; torch.cuda.empty_cache()
                print(f"  {scale} {label} L{L}: total {(counted+gdn)/1e9:.2f} GFLOP", flush=True)
            except RuntimeError as e:
                sc[label]["by_ctx"][str(L)] = {"error": str(e)[:80]}
                torch.cuda.empty_cache()
    results["scales"][scale] = sc

# ----- ratios -----
results["ratio_C_over_A"] = {}
for scale, sc in results["scales"].items():
    r = {}
    for L in sc["A"]["by_ctx"]:
        a_t = sc["A"]["by_ctx"][L].get("total"); c_t = sc["C"]["by_ctx"][L].get("total")
        if a_t and c_t:
            r[L] = round(a_t / c_t, 3)   # A/C = how many x fewer FLOPs C uses
    results["ratio_C_over_A"][scale] = r

json.dump(results, open(a.out, "w"), indent=2)
print("RATIOS (A/C FLOPs, >1 = C cheaper):", json.dumps(results["ratio_C_over_A"]), flush=True)
print("WROTE", a.out, flush=True)
