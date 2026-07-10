"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz), single-GPU.

DeepSeek-V4 trains CSA with Muon for *most* parameters and keeps AdamW only for
embeddings, the prediction head, and RMSNorm weights (DeepSeek-V4 tech report;
Muon = Jordan et al. 2024). We were running Adam on everything, which is
off-recipe exactly on the CSA compressor / indexer / attention / MLP weight
matrices — the part that empirically under-optimizes (GDN+CSA losing to its own
strict subset GDN+SWA on the dilution task).

`build_optimizer` splits parameters by NAME (not just ndim) into three groups:

  * MUON        : 2-D weight matrices of the DeepSeek transformer body
                  (attention q/kv/o projections, CSA compressor + indexer
                  projections, dense-MLP matrices). These are true linear maps,
                  so Newton-Schulz orthogonalization of the momentum is valid.
  * ADAMW       : embeddings, lm_head, all RMSNorm / 1-D params (sinks),
                  `position_bias` (2-D but an additive bias, NOT a linear map),
                  and the GDN-2 *main-path* layers (B/C/SWA-A/SWA-B) — GDN-2's
                  own recipe is AdamW, so we keep it native and isolate Muon to
                  the CSA backbone (clean intervention).
  * ADAMW-gist  : the gist conditioner branch (GDN-2 gist stream + W_g/gamma /
                  pcat projections) at `gist_lr_mult x lr` — preserves the
                  Exp-1 zero-init LR-decoupling recipe (an Adam phenomenon).

The Newton-Schulz quintic is the standard (a,b,c) = (3.4445,-4.7750,2.0315)
5-step iteration; the update is scaled by sqrt(max(1, rows/cols)) so the
effective step is shape-invariant.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize G (>=2D) via a quintic Newton-Schulz iteration, in bf16."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Single-device Muon. Apply ONLY to 2-D weight matrices that are genuine
    linear maps (see build_optimizer for the param selection)."""

    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95,
                 nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mom, nesterov, ns, lr, wd = (group["momentum"], group["nesterov"],
                                         group["ns_steps"], group["lr"], group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 2:                      # flatten conv-like to 2D for NS
                    g = g.view(g.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mom)
                upd = g.lerp_(buf, mom) if nesterov else buf
                upd = zeropower_via_newtonschulz5(upd, steps=ns)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(upd.reshape(p.shape).to(p.dtype), alpha=-lr * scale)
        return loss


# --------------------------------------------------------------------------- #
class MultiOpt:
    """Bundle several optimizers so the train loop can treat them as one
    (shared zero_grad/step, and a flat `param_groups` for LR scheduling)."""

    def __init__(self, opts):
        self.opts = [o for o in opts if o is not None and len(o.param_groups) and
                     any(len(g["params"]) for g in o.param_groups)]

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


# --------------------------------------------------------------------------- #
def _is_gist(name: str) -> bool:
    return (("gist" in name) or ("pcat_" in name))


def _is_main_gdn(name: str) -> bool:
    # main-path GDN layer lives at lm.model.layers[i].self_attn.gdn.* (GDNLayer)
    return ".self_attn.gdn." in name


def _is_gdn_stream(name: str) -> bool:
    """A GatedDeltaNet-2 module param: the main-path GDN (`.self_attn.gdn.`) OR the
    shared gist stream (`gist.*`, i.e. CandidateModel.gist = GatedDeltaNet2). These
    follow the ORIGINAL GDN-2 recipe (AdamW, lr 4e-4, wd 0.1) -- NOT the 2e-3/wd0 we
    use elsewhere, which empirically under-trains them to a floor. Excludes
    `gist_readout` (a plain Linear) and the gist GATE (gamma/W_g/pcat), which keep
    the Exp-1 zero-init decoupled-LR recipe."""
    return name.startswith("gist.") or _is_main_gdn(name)


def _muon_eligible(name: str, p: torch.Tensor) -> bool:
    if p.ndim < 2:
        return False
    if "embed_tokens" in name or "lm_head" in name:
        return False
    if "position_bias" in name:            # 2-D additive bias, not a linear map
        return False
    if _is_gist(name) or _is_main_gdn(name):
        return False
    return True


def build_optimizer(model, *, opt="adam", lr=2e-3, muon_lr=0.02, gist_lr_mult=0.1,
                    gdn_lr=4e-4, gdn_wd=0.1, betas=(0.9, 0.95), weight_decay=0.0):
    """Return a MultiOpt. opt='adam' -> single AdamW (legacy path, + gist group);
    opt='muon' -> Muon on CSA-body matrices + AdamW on the rest (paper recipe).

    GatedDeltaNet-2 streams (main-path GDN + the gist GDN stream) get their OWN
    AdamW group at the ORIGINAL GDN-2 recipe (`gdn_lr`=4e-4, `gdn_wd`=0.1; A_log /
    dt_bias / 1-D excluded from weight decay) -- the prior shared 2e-3/wd0 group
    under-trained them to a floor (a recipe artifact, not a GDN capability limit)."""
    named = list(model.named_parameters())
    gist_ids = model.gist_parameter_ids() if hasattr(model, "gist_parameter_ids") else set()

    if opt == "adam":
        gist, backbone = [], []
        for n, p in named:
            (gist if id(p) in gist_ids else backbone).append(p)
        groups = [torch.optim.AdamW(backbone, lr=lr, betas=betas, weight_decay=weight_decay)]
        if gist:
            groups.append(torch.optim.AdamW(gist, lr=lr * gist_lr_mult, betas=betas, weight_decay=weight_decay))
        return MultiOpt(groups)

    # --- Muon path ---
    muon_p, adam_p, gist_p, gdn_decay, gdn_nodecay = [], [], [], [], []
    for n, p in named:
        if _is_gdn_stream(n):                      # GDN-2 recipe (checked before gist_ids)
            if p.ndim < 2 or n.endswith("A_log") or n.endswith("dt_bias"):
                gdn_nodecay.append(p)
            else:
                gdn_decay.append(p)
        elif id(p) in gist_ids:                    # gist GATE (gamma/W_g/pcat/readout)
            gist_p.append(p)
        elif _muon_eligible(n, p):
            muon_p.append(p)
        else:
            adam_p.append(p)
    opts = [
        Muon(muon_p, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=weight_decay),
        torch.optim.AdamW(adam_p, lr=lr, betas=betas, weight_decay=weight_decay),
    ]
    if gdn_decay or gdn_nodecay:
        opts.append(torch.optim.AdamW(
            [{"params": gdn_decay, "weight_decay": gdn_wd},
             {"params": gdn_nodecay, "weight_decay": 0.0}],
            lr=gdn_lr, betas=betas))
    if gist_p:
        opts.append(torch.optim.AdamW(gist_p, lr=lr * gist_lr_mult, betas=betas, weight_decay=weight_decay))
    return MultiOpt(opts)


def optimizer_param_counts(model, **kw):
    """Diagnostic: how many params land in each group (sanity for the split)."""
    named = list(model.named_parameters())
    gist_ids = model.gist_parameter_ids() if hasattr(model, "gist_parameter_ids") else set()
    muon = adam = gist = gdn = 0
    for n, p in named:
        if _is_gdn_stream(n):
            gdn += p.numel()
        elif id(p) in gist_ids:
            gist += p.numel()
        elif _muon_eligible(n, p):
            muon += p.numel()
        else:
            adam += p.numel()
    return {"muon": muon, "adam": adam, "gdn": gdn, "gist": gist}
