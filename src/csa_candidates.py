"""Rank-7 candidate factory (plan §1): 7 architectures on one stripped DeepSeek-V4
backbone, all ~1M params, exactly 2 layers.

| ID    | stack        | gist        | integration            |
| A     | [CSA, CSA]   | -           | baseline (weak)        |
| B     | [GDN, CSA]   | -           | decoupled, global-first|
| C     | [CSA, GDN]   | -           | decoupled, local-first |
| WAdd  | [CSA,CSA]+g  | additive    | coupled (pool+index)   |
| PCat  | [CSA,CSA]+g  | concat      | coupled (proj input)   |
| SWA-A | [GDN, SWA]   | -           | decoupled, global-first|
| SWA-B | [SWA, GDN]   | -           | decoupled, local-first |

A GDN-2 main-path layer = a `GDNLayer` adapter spliced over the decoder block's
`self_attn` (the config marks it "sliding_attention" so the stock attention is a
cheap no-compressor module we discard). SWA layers keep the stock sliding
attention. Coupled candidates (WAdd/PCat) carry an extra shared GDN-2 gist stream
over embeddings, so they are param-heavy -> the §8 controls (param-matched base +
gist-ablation) are what make a coupled win credible.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mdl

from gdn_import import GatedDeltaNet2
from csa_model import make_csa_config, _densify, count_params
from csa_gist_layer import DeepseekV4CSAGistCompressor

# layer-name -> config attention type
_TYPE = {"csa": "compressed_sparse_attention", "swa": "sliding_attention", "gdn": "sliding_attention"}

CANDIDATES = {
    "A":     {"layers": ["csa", "csa"], "gist": None},
    "B":     {"layers": ["gdn", "csa"], "gist": None},
    "C":     {"layers": ["csa", "gdn"], "gist": None},
    "WAdd":  {"layers": ["csa", "csa"], "gist": "wadd"},        # v1: pool+index together
    "WAdd-pool": {"layers": ["csa", "csa"], "gist": "wadd_pool"},  # v2 regime P: pool only
    "WAdd-sel":  {"layers": ["csa", "csa"], "gist": "wadd_sel"},   # v2 regime S: indexer only
    "PCat":  {"layers": ["csa", "csa"], "gist": "pcat"},
    "SWA-A": {"layers": ["gdn", "swa"], "gist": None},
    "SWA-B": {"layers": ["swa", "gdn"], "gist": None},
    # linear + FULL attention (Qwen-style hybrid): [gdn, swa] but run with a
    # full-sequence window (--model_sw >= seq) so the swa layer is full attention.
    "LinFull": {"layers": ["gdn", "swa"], "gist": None},
    # §4.4 gate probes (no gist): pure single-mechanism 2-layer stacks
    "GDN2":  {"layers": ["gdn", "gdn"], "gist": None},          # G1: must be PARTIAL at N>=cap
    "SWA2":  {"layers": ["swa", "swa"], "gist": None},          # G2: must FLOOR on distant needles
}


# --------------------------------------------------------------------------- #
# GDN-2 as a main-path layer (drop-in for DeepseekV4Attention)                 #
# --------------------------------------------------------------------------- #
class GDNLayer(nn.Module):
    """Runs a GDN-2 token mixer in place of attention. The decoder block calls
    `self_attn(normed_hidden, **kwargs)` and wants `(output, attn_weights)`; GDN-2
    needs none of the rope / 4-D mask / cache kwargs (and would assert on the 4-D
    mask), so we drop them and run cache-free (training only)."""

    def __init__(self, config, *, num_heads=4, head_dim=32):
        super().__init__()
        self.gdn = GatedDeltaNet2(
            hidden_size=config.hidden_size, expand_v=1, head_dim=head_dim,
            num_heads=num_heads, mode="chunk", use_short_conv=True, conv_size=4,
            layer_idx=0, norm_eps=config.rms_norm_eps,
        )

    def forward(self, hidden_states, **kwargs):
        out = self.gdn(hidden_states)            # (o, None, cache)
        o = out[0] if isinstance(out, tuple) else out
        return o, None


# --------------------------------------------------------------------------- #
# The unified candidate wrapper                                                #
# --------------------------------------------------------------------------- #
class CandidateModel(nn.Module):
    def __init__(self, name, config, *, gist_dim=32, gist_heads=2, gist_head_dim=32,
                 gdn_heads=4, gdn_head_dim=32, inject_mode="channel"):
        super().__init__()
        spec = CANDIDATES[name]
        self.name = name
        self.config = config
        self.gist_kind = spec["gist"]
        self.layer_kinds = spec["layers"]
        self.use_gist = self.gist_kind is not None
        self._ablate_gist = False
        self._gist_enabled = True
        self._kl_mode = False           # indexer KL training active (capture teacher attn weights)
        self._gist_compressors = []     # CSA layers carrying gist conditioning (for set_gist)
        self._csa_compressors = []      # ALL swapped CSA compressors (for set_ste / KL)

        self.lm = DeepseekV4ForCausalLM(config)
        _densify(self.lm)

        # splice GDN-2 main-path layers
        for li, kind in enumerate(self.layer_kinds):
            if kind == "gdn":
                self.lm.model.layers[li].self_attn = GDNLayer(
                    config, num_heads=gdn_heads, head_dim=gdn_head_dim)

        # coupled gist conditioning over the CSA layers (shared GDN-2 gist stream)
        if self.use_gist:
            self.gist = GatedDeltaNet2(
                hidden_size=config.hidden_size, expand_v=1, head_dim=gist_head_dim,
                num_heads=gist_heads, mode="chunk", use_short_conv=True, conv_size=4,
                layer_idx=0, norm_eps=config.rms_norm_eps)
            self.gist_readout = nn.Linear(config.hidden_size, gist_dim, bias=False)
        # WAdd*: additive zero-init bias; pool=compressor gate, sel=indexer gate. PCat: concat.
        # None -> all flags off (bit-identical to upstream); we still swap so the indexer
        # stashes scores for STE selector training, uniformly across all candidates.
        ckw = {
            "wadd":      dict(gist_pool=True,  gist_index=True),
            "wadd_pool": dict(gist_pool=True,  gist_index=False),
            "wadd_sel":  dict(gist_pool=False, gist_index=True),
            "pcat":      dict(pcat=True),
        }.get(self.gist_kind, dict())
        # swap EVERY CSA layer's compressor for our copy
        for li, kind in enumerate(self.layer_kinds):
            if kind != "csa":
                continue
            stock = self.lm.model.layers[li].self_attn.compressor
            custom = DeepseekV4CSAGistCompressor(
                config, gist_dim=gist_dim, inject_mode=inject_mode, **ckw)
            # copy stock weights (kv/gate/indexer); gist_gate / pcat_* stay at fresh init
            missing, unexpected = custom.load_state_dict(stock.state_dict(), strict=False)
            assert not unexpected, unexpected
            assert all(("gist_gate" in k or "pcat_" in k) for k in missing), missing
            self.lm.model.layers[li].self_attn.compressor = custom
            self._csa_compressors.append(custom)
            if self.use_gist:
                self._gist_compressors.append(custom)

        self._install_kl_attn_hooks()    # capture per-layer main-attention weights (KL teacher)

    def set_ste(self, on=True):
        """Enable straight-through top-k so the LM loss trains the Lightning Indexer."""
        for c in self._csa_compressors:
            c.ste_topk = bool(on)

    # ------------------------------------------------------------------ #
    # Indexer KL training (DeepSeek-V3.2/V4 recipe; see deepseek-indexer-kl-recipe)
    # ------------------------------------------------------------------ #
    def _install_kl_attn_hooks(self):
        """Wrap each CSA layer's self_attn.forward to stash the main attention's per-block mass
        (eager returns attn_weights [B,H,S,kv]; the compressed blocks are the last `compressed_len`
        columns of kv = cat([window, compressed])). Summed over heads, this is the KL teacher."""
        model = self
        for li, kind in enumerate(self.layer_kinds):
            if kind != "csa":
                continue
            attn = self.lm.model.layers[li].self_attn
            comp = attn.compressor

            def make(orig, comp):
                def fwd(*a, **kw):
                    out, w = orig(*a, **kw)
                    if model._kl_mode and w is not None:
                        cl = getattr(comp, "_compressed_len", 0)
                        if cl > 0 and w.shape[-1] >= cl:
                            comp._kl_teacher = w[..., -cl:].sum(dim=1).detach()   # [B,S,cl]
                    return out, w
                return fwd

            attn.forward = make(attn.forward, comp)

    def set_indexer_kl(self, on=True, dense=False):
        """on -> train the indexer by KL (detached inputs, capture teacher). dense -> warm-up
        phase: main attention attends ALL causal blocks so the teacher is the full distribution."""
        self._kl_mode = bool(on)
        for c in self._csa_compressors:
            c.kl_mode = bool(on)
            c.dense_mode = bool(dense)

    def indexer_kl_loss(self, eps=1e-9):
        """Mean over CSA layers of KL(p_teacher ‖ Softmax(index_scores)) per query (eq 3/4).
        Teacher detached; student carries grad to indexer params only (inputs were detached)."""
        losses = []
        for comp in self._csa_compressors:
            t = getattr(comp, "_kl_teacher", None)
            s = getattr(comp, "_kl_index_scores", None)
            if t is None or s is None or t.shape != s.shape:
                continue
            t, s = t.float(), s.float()                              # bf16 log_softmax loses precision
            p = t / t.sum(-1, keepdim=True).clamp_min(eps)            # teacher dist over blocks
            logq = s.log_softmax(-1)
            term = p * (p.clamp_min(eps).log() - logq)
            term = torch.where(p > 0, term, torch.zeros_like(term))   # avoid 0*inf at future blocks
            kl = term.sum(-1)                                         # [B,S]
            rows = t.sum(-1) > 0                                      # queries with >=1 valid block
            if rows.any():
                losses.append(kl[rows].mean())
        if not losses:
            return None
        return sum(losses) / len(losses)

    # ------------------------------------------------------------------ #
    def _compute_gist(self, embeds):
        g, _, _ = self.gist(embeds)
        g = self.gist_readout(g)
        if self._ablate_gist:
            g = torch.zeros_like(g)
        return g

    def forward(self, input_ids):
        embeds = self.lm.model.embed_tokens(input_ids)
        active = self.use_gist and self._gist_enabled
        if active:
            g = self._compute_gist(embeds)
            for comp in self._gist_compressors:
                comp.set_gist(g)
        # gist stays set on the compressors through backward: gradient checkpointing
        # recomputes layers in the backward pass and must see the same gist; the next
        # forward overwrites it (set_gist(None) here would break that recompute).
        out = self.lm(inputs_embeds=embeds, use_cache=False)
        return out.logits

    # ------------------------------------------------------------------ #
    def _gist_gates(self):
        gates = []
        for comp in self._gist_compressors:
            if getattr(comp, "gist_pool", False):
                gates.append(comp.gist_gate)
            if getattr(comp, "gist_index", False):
                gates.append(comp.indexer.gist_gate)
        return gates

    def gist_parameter_ids(self):
        ids = set()
        if self.use_gist:
            for p in self.gist.parameters():
                ids.add(id(p))
            for p in self.gist_readout.parameters():
                ids.add(id(p))
            for comp in self._gist_compressors:
                for p in comp.gist_parameters():
                    ids.add(id(p))
        return ids

    def param_groups(self, lr, gist_lr_mult=1.0):
        gids = self.gist_parameter_ids()
        backbone, gist = [], []
        for p in self.parameters():
            (gist if id(p) in gids else backbone).append(p)
        groups = [{"params": backbone, "lr": lr}]
        if gist:
            groups.append({"params": gist, "lr": lr * gist_lr_mult})
        return groups

    @torch.no_grad()
    def usage_report(self):
        if not self.use_gist or not self._gist_gates():
            return {"gamma_absmean": 0.0, "gamma_max": 0.0, "gamma_per_layer": [], "W_g_absmean": 0.0}
        gates = self._gist_gates()
        gammas = [g.gamma.abs().item() for g in gates]
        wg = [g.W_g.weight.abs().mean().item() for g in gates]
        return {"gamma_absmean": sum(gammas) / len(gammas), "gamma_max": max(gammas),
                "gamma_per_layer": gammas, "W_g_absmean": sum(wg) / len(wg)}


# --------------------------------------------------------------------------- #
# Builders / param tuning                                                       #
# --------------------------------------------------------------------------- #
def build_candidate(name, vocab_size, cell, *, seed=0, device="cpu",
                    moe_intermediate_size=952, inject_mode="channel", **kw):
    """cell = dict(m, sw, topk). Sets the per-layer schedule from CANDIDATES[name]."""
    spec = CANDIDATES[name]
    layer_types = [_TYPE[k] for k in spec["layers"]]
    torch.manual_seed(seed)
    cfg = make_csa_config(
        vocab_size, num_hidden_layers=len(layer_types), layer_types=layer_types,
        compress_rate_csa=cell["m"], sliding_window=cell["sw"], index_topk=cell["topk"],
        moe_intermediate_size=moe_intermediate_size,
        max_position_embeddings=4 * (2 * cell.get("N", 16) + 5) + 64)
    return CandidateModel(name, cfg, inject_mode=inject_mode, **kw).to(device)


def tune_moe_for_target(name, vocab_size, cell, target, *, lo=64, hi=4096, **kw):
    """Binary-search moe_intermediate_size so the candidate's param count ~= target."""
    def n_at(inter):
        m = build_candidate(name, vocab_size, cell, moe_intermediate_size=inter, device="cpu", **kw)
        n = count_params(m)
        del m
        return n
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = n_at(mid)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best  # (moe_intermediate_size, param_count)
