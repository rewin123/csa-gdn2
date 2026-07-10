"""CSA-only DeepSeek-V4 at ~1M  (+ optional GDN-2 gist-conditioning of the CSA pooling gate).

Two arms (plan §0):
  * M_base : stripped CSA-only DeepseekV4ForCausalLM (hc_mult=1, 1-expert MoE,
             bf16, eager). The HF `deepseek_v4` reference IS the CSA source.
  * M_cond : same backbone + a cheap causal GDN-2 "gist" stream whose readout
             biases the CSA compressor's *pre-softmax pooling logits* through a
             ZERO-init scale `gamma` -> bit-identical to M_base at init.

The conditioning lives in a *copied* custom layer (`csa_gist_layer`,
`DeepseekV4CSAGistCompressor`), NOT a global monkey-patch: we build the stock
backbone, then for the cond arm swap each CSA layer's `.compressor` for the
custom one, copying the stock weights across so the backbone is bit-identical to
M_base at init (paired-seed protocol). Only the zero-init gist params are added.
Equivalence (gist off / gamma=0 == upstream) is proved in
`tests/test_csa_gist_equiv.py`.

Sites (plan §3, rank-7 WAdd): `gist_pool` conditions the compressor pooling
(what to keep); `gist_index` conditions the Lightning Indexer's key pooling
(what to select). `gamma` (scalar, per site) is zero-init -> exact no-op at
init; `W_g` is normal-init so d L/d gamma != 0 and the branch learns from step 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mdl

from gdn_import import GatedDeltaNet2
from csa_gist_layer import DeepseekV4CSAGistCompressor


# --------------------------------------------------------------------------- #
# 0. Dense MLP (plan §2.3: "disable MoE -> dense MLP"). Replaces the routed     #
#    SparseMoeBlock so we avoid the grouped-mm kernel and dead expert params.   #
# --------------------------------------------------------------------------- #
class DenseMLP(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.mlp = mdl.DeepseekV4MLP(config)

    def forward(self, x, input_ids=None):
        return self.mlp(x)


def _densify(lm: DeepseekV4ForCausalLM):
    for layer in lm.model.layers:
        layer.mlp = DenseMLP(lm.config)


# --------------------------------------------------------------------------- #
# 1. Stripped CSA-only config at ~1M                                           #
# --------------------------------------------------------------------------- #
def make_csa_config(
    vocab_size: int,
    *,
    hidden_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    head_dim: int = 32,
    q_lora_rank: int = 96,
    o_lora_rank: int = 96,
    index_n_heads: int = 4,
    index_head_dim: int = 32,
    index_topk: int = 16,
    compress_rate_csa: int = 4,
    sliding_window: int = 32,
    moe_intermediate_size: int = 952,
    max_position_embeddings: int = 32768,
    layer_types: list | None = None,
) -> DeepseekV4Config:
    """CSA-only by default; mHC/MoE/Muon/FP-quant/MTP disabled (plan §2.3).

    `layer_types` overrides the per-layer attention schedule (rank-7): use
    "compressed_sparse_attention" for CSA layers and "sliding_attention" for SWA
    layers and for GDN-2 main-path layers (whose `self_attn` is swapped out by
    the candidate builder; the sliding type just gives them a cheap no-compressor
    attention module + the plain `main` rope + a simple mask path)."""
    n = num_hidden_layers
    if layer_types is None:
        layer_types = ["compressed_sparse_attention"] * n
    assert len(layer_types) == n, (layer_types, n)
    return DeepseekV4Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=n,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        o_groups=1,
        o_lora_rank=o_lora_rank,
        partial_rotary_factor=0.5,             # qk_rope = head_dim/2 (even)
        # --- attention schedule ---
        layer_types=layer_types,
        compress_rates={"compressed_sparse_attention": compress_rate_csa,
                        "heavily_compressed_attention": 128},
        sliding_window=sliding_window,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        # --- mHC off: a single residual stream ---
        hc_mult=1,
        hc_sinkhorn_iters=1,
        # --- MoE -> effectively dense: 1 routed expert + 1 shared MLP ---
        mlp_layer_types=["moe"] * n,
        n_routed_experts=1,
        num_experts_per_tok=1,
        n_shared_experts=1,
        moe_intermediate_size=moe_intermediate_size,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        norm_topk_prob=False,
        # --- misc ---
        max_position_embeddings=max_position_embeddings,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        output_router_logits=False,
        _attn_implementation="eager",
    )


# --------------------------------------------------------------------------- #
# 2. The two-arm wrapper                                                        #
#    cond swaps each CSA layer's compressor for the copied gist compressor,     #
#    copying the stock weights so the backbone is bit-identical to base at init.#
# --------------------------------------------------------------------------- #
class CSAGistModel(nn.Module):
    """M_base (use_gist=False) or M_cond (use_gist=True).

    gist_pool / gist_index select the conditioning sites (WAdd): pooling gate of
    the compressor and/or the Lightning Indexer's key pooling. inject_mode picks
    channel (per-(slot,channel)) vs scalar (per-slot) bias. The default
    (pool only, channel) reproduces Experiment 1."""

    def __init__(self, config: DeepseekV4Config, *, use_gist: bool,
                 gist_dim: int = 32, gist_heads: int = 2, gist_head_dim: int = 32,
                 inject_mode: str = "channel", gist_pool: bool = True, gist_index: bool = False):
        super().__init__()
        self.config = config
        self.use_gist = use_gist
        self.inject_mode = inject_mode
        self.gist_pool = gist_pool
        self.gist_index = gist_index
        self.lm = DeepseekV4ForCausalLM(config)
        _densify(self.lm)                          # dense MLP, no routed experts
        self._ablate_gist = False
        self._gist_enabled = True                  # hard-start warmup toggle
        self._compressors = []                     # custom gist compressors (cond only)

        if use_gist:
            self.gist = GatedDeltaNet2(
                hidden_size=config.hidden_size, expand_v=1, head_dim=gist_head_dim,
                num_heads=gist_heads, mode="chunk", use_short_conv=True, conv_size=4,
                layer_idx=0, norm_eps=config.rms_norm_eps,
            )
            self.gist_readout = nn.Linear(config.hidden_size, gist_dim, bias=False)
            self.gist_m = config.compress_rates["compressed_sparse_attention"]
            for layer in self.lm.model.layers:
                stock = layer.self_attn.compressor            # already-initialized upstream CSA
                custom = DeepseekV4CSAGistCompressor(
                    config, gist_pool=gist_pool, gist_index=gist_index,
                    gist_dim=gist_dim, inject_mode=inject_mode)
                # copy stock weights -> backbone bit-identical to base; gist params stay zero-init
                missing, unexpected = custom.load_state_dict(stock.state_dict(), strict=False)
                assert not unexpected, f"unexpected keys copying compressor: {unexpected}"
                assert all("gist_gate" in k for k in missing), f"unexpected missing: {missing}"
                layer.self_attn.compressor = custom
                self._compressors.append(custom)

    def _compute_gist(self, embeds):
        g, _, _ = self.gist(embeds)                # [B, S, hidden]
        g = self.gist_readout(g)                    # [B, S, gist_dim]
        if self._ablate_gist:
            g = torch.zeros_like(g)
        return g

    def forward(self, input_ids):
        embeds = self.lm.model.embed_tokens(input_ids)
        active = self.use_gist and self._gist_enabled
        if active:
            g = self._compute_gist(embeds)          # hard-start: skipped (pure base) until enabled
            for comp in self._compressors:
                comp.set_gist(g)
        try:
            out = self.lm(inputs_embeds=embeds)
        finally:
            if active:
                for comp in self._compressors:
                    comp.set_gist(None)
        return out.logits

    def _gist_gates(self):
        """The (W_g, gamma) gates across all conditioning sites."""
        gates = []
        for comp in self._compressors:
            if comp.gist_pool:
                gates.append(comp.gist_gate)
            if comp.gist_index:
                gates.append(comp.indexer.gist_gate)
        return gates

    def gist_parameter_ids(self):
        ids = set()
        if self.use_gist:
            for p in self.gist.parameters():
                ids.add(id(p))
            for p in self.gist_readout.parameters():
                ids.add(id(p))
            for comp in self._compressors:
                for p in comp.gist_parameters():
                    ids.add(id(p))
        return ids

    def param_groups(self, lr, gist_lr_mult=1.0):
        """Backbone at `lr`; gist stream + injection at `lr*gist_lr_mult` so the
        backbone can learn local features / grok before the gist branch ramps in."""
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
        if not self.use_gist:
            return {}
        gates = self._gist_gates()
        gammas = [g.gamma.abs().item() for g in gates]
        wg = [g.W_g.weight.abs().mean().item() for g in gates]
        return {
            "gamma_absmean": sum(gammas) / len(gammas),
            "gamma_max": max(gammas),
            "gamma_per_layer": gammas,
            "W_g_absmean": sum(wg) / len(wg),
        }


# --------------------------------------------------------------------------- #
# 5. Param accounting                                                          #
# --------------------------------------------------------------------------- #
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_models(config, **gist_kw):
    base = CSAGistModel(config, use_gist=False)
    cond = CSAGistModel(config, use_gist=True, **gist_kw)
    return base, cond
