
import sys, os, json, argparse
sys.path.insert(0,'src')
import torch, torch.nn as nn
from lm_config import build_lm
from data_fineweb import ShardLoader
import csa_candidates as cc

dev="cuda"
ap=argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ffn", type=int, default=862)
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--hidden", type=int, default=384)
ap.add_argument("--ctx", type=int, default=2048)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--out", required=True)
a=ap.parse_args()

m=build_lm("C", depth=a.depth, hidden=a.hidden, heads=6, head_dim=64, ffn=a.ffn, device=dev)
sd=torch.load(a.ckpt, map_location=dev)
m.load_state_dict(sd); m.eval()

gdn_layers=[(li,lyr) for li,lyr in enumerate(m.lm.model.layers) if isinstance(lyr.self_attn, cc.GDNLayer)]
print("GDN main-path layer indices:", [li for li,_ in gdn_layers], flush=True)

ABLATE=set()
def mk_hook(li):
    def hook(mod, inp, out):
        if li in ABLATE:
            o = out[0] if isinstance(out, tuple) else out
            o = torch.zeros_like(o)
            return (o, None) if isinstance(out, tuple) else o
        return out
    return hook
handles=[lyr.self_attn.register_forward_hook(mk_hook(li)) for li,lyr in gdn_layers]

val=ShardLoader("data/fineweb","val",a.ctx,a.batch,dev,seed=999)
@torch.no_grad()
def eval_fw(iters):
    tot=0.0; n=0
    for _ in range(iters):
        x,y=val.batch()
        logits=m(x)
        loss=nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)).float(), y.reshape(-1))
        tot+=loss.item(); n+=1
    return tot/n

res={}
ABLATE=set(); torch.manual_seed(0); res["full"]=eval_fw(a.iters)
print("full fineweb CE:", res["full"], flush=True)
ABLATE=set(li for li,_ in gdn_layers); torch.manual_seed(0); res["all_gdn_off"]=eval_fw(a.iters)
print("all GDN off CE:", res["all_gdn_off"], flush=True)
res["per_layer"]={}
for li,_ in gdn_layers:
    ABLATE={li}; torch.manual_seed(0); res["per_layer"][str(li)]=eval_fw(a.iters)
    print(f"layer {li} off CE:", res["per_layer"][str(li)], flush=True)
res["gdn_layer_indices"]=[li for li,_ in gdn_layers]
res["delta_all_off"]=res["all_gdn_off"]-res["full"]
json.dump(res, open(a.out,"w"), indent=2)
print("WROTE", a.out, flush=True)
