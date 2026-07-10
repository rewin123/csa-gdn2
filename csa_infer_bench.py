import sys, json, argparse, time
sys.path.insert(0,'src')
import torch
from lm_config import build_lm

dev="cuda"
ap=argparse.ArgumentParser()
ap.add_argument("--ckpt_A", required=True); ap.add_argument("--ffn_A", type=int, default=565)
ap.add_argument("--ckpt_C", required=True); ap.add_argument("--ffn_C", type=int, default=862)
ap.add_argument("--ctxs", type=str, default="512,1024,2048,4096")
ap.add_argument("--reps", type=int, default=20); ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--out", required=True)
a=ap.parse_args()

def load(cand, ffn, ckpt):
    m=build_lm(cand, depth=6, hidden=384, heads=6, head_dim=64, ffn=ffn, device=dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev)); m.eval()
    return m.to(torch.bfloat16)

models={"A":load("DeepSeek",a.ffn_A,a.ckpt_A), "C":load("C",a.ffn_C,a.ckpt_C)}
ctxs=[int(x) for x in a.ctxs.split(",")]
res={"batch":a.batch,"reps":a.reps,"dtype":"bfloat16","gpu":torch.cuda.get_device_name(0),"data":{}}

@torch.no_grad()
def timed(m, L):
    x=torch.randint(0,50257,(a.batch,L),device=dev)
    for _ in range(a.warmup): m(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    for _ in range(a.reps): m(x)
    torch.cuda.synchronize()
    dt=(time.perf_counter()-t0)/a.reps
    peak=torch.cuda.max_memory_allocated()/1e9
    return dt, peak

for name,m in models.items():
    res["data"][name]={}
    for L in ctxs:
        try:
            dt,peak=timed(m,L)
            toks=a.batch*L/dt
            res["data"][name][str(L)]={"latency_ms":round(dt*1e3,3),"tokens_per_s":round(toks,1),"peak_gb":round(peak,3)}
            print(f"{name} ctx{L}: {dt*1e3:.2f}ms  {toks:,.0f} tok/s  peak {peak:.2f}GB", flush=True)
        except RuntimeError as e:
            res["data"][name][str(L)]={"error":str(e)[:100]}
            print(f"{name} ctx{L}: ERR {str(e)[:80]}", flush=True)
            torch.cuda.empty_cache()

# speedup C/A per ctx
res["speedup_C_over_A"]={}
for L in ctxs:
    L=str(L)
    da=res["data"]["A"].get(L,{}); dc=res["data"]["C"].get(L,{})
    if "tokens_per_s" in da and "tokens_per_s" in dc:
        res["speedup_C_over_A"][L]=round(dc["tokens_per_s"]/da["tokens_per_s"],3)
json.dump(res,open(a.out,"w"),indent=2)
print("SPEEDUP C/A:", res["speedup_C_over_A"], flush=True)
print("WROTE",a.out,flush=True)