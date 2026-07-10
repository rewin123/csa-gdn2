"""Custom CSA layers with OPTIONAL gist-conditioning, as full copies of the
upstream DeepSeek-V4 CSA layer code (transformers 5.12.1).

Per the plan addendum: instead of monkey-patching `DeepseekV4CSACompressor.forward`
at the class level, we copy the upstream compressor + indexer verbatim into two
standalone modules and add the gist conditioning behind optional flags. With every
gist flag OFF (the default) each module is *bit-identical* to its upstream twin --
this is what `tests/test_csa_gist_equiv.py` proves.

Two conditioning sites, both via the same ZERO-init mechanism (gamma=0 -> exact
no-op at init):

  * pooling gate of the CSA compressor   (`gist_pool`)  -- conditions *what to keep*
  * pooling gate of the Lightning Indexer (`gist_index`) -- conditions *what to
    select* through the query x key score (key-side, v2a in the strategy notes)

The injected bias for compressed entry `w` is content-aware, causal, and additive
to the *pre-softmax* pooling logits. Entry `w` closes at `t_close = w*m + (m-1) +
first_window_position`, the first step it is ever visible to a query (CSA's own
rule `w < (t+1)//m`), so reading the gist at `t_close` is strictly causal:

    channel mode : pool_logit[b,w,i,c] += gamma * W_g(g[t_close_w])[c] * kv_slot[b,w,i,c]
    scalar  mode : pool_logit[b,w,i,:] += gamma * <W_g(g[t_close_w]), kv_slot[b,w,i,:]>/sqrt(hd)

`gamma` (scalar) is zero-init -> exact no-op at init; `W_g` is normal-init so
d L/d gamma != 0 and the branch learns from step 1.

The verbatim regions are delimited by `# >>> upstream` / `# <<< upstream` markers;
the only edits inside them are the guarded `# --- GIST ---` blocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mdl

# upstream helpers we reuse verbatim
DeepseekV4RMSNorm = mdl.DeepseekV4RMSNorm
DeepseekV4RotaryEmbedding = mdl.DeepseekV4RotaryEmbedding
DeepseekV4IndexerScorer = mdl.DeepseekV4IndexerScorer
apply_rotary_pos_emb = mdl.apply_rotary_pos_emb


# --------------------------------------------------------------------------- #
# Shared gist bias helper                                                      #
# --------------------------------------------------------------------------- #
def _gist_pool_bias(gist, W_g, gamma, new_kv, *, m, first_window_position, mode, scale):
    """Zero-init pre-softmax pooling-logit bias for a [B, n_win, 2m, hd] gate.

    Reads the gist at each window's causal close position `t_close = w*m+(m-1)+fwp`.
    Returns a float tensor broadcastable onto `new_gate`; with gamma=0 it is exactly
    0.0 (so `new_gate + 0.0` is bit-identical, even where new_gate is -inf)."""
    n_windows = new_kv.shape[1]
    dev = new_kv.device
    idx = (torch.arange(n_windows, device=dev) * m + (m - 1) + first_window_position)
    idx = idx.clamp(max=gist.shape[1] - 1)
    g_close = gist[:, idx, :]                                  # [B, n_win, gist_dim]
    wq = W_g(g_close).float()                                  # [B, n_win, hd]
    if mode == "channel":
        bias = gamma.float() * (wq.unsqueeze(2) * new_kv.float())          # [B,nw,2m,hd]
    else:
        score = torch.einsum("bwd,bwsd->bws", wq, new_kv.float()) * scale  # [B,nw,2m]
        bias = (gamma.float() * score).unsqueeze(-1)                       # broadcast over hd
    return bias


class _GistGate(nn.Module):
    """Holds the (W_g, gamma) of one gist conditioning site."""

    def __init__(self, gist_dim: int, head_dim: int, mode: str = "channel"):
        super().__init__()
        self.W_g = nn.Linear(gist_dim, head_dim, bias=False)   # normal init -> live grad
        self.gamma = nn.Parameter(torch.zeros(1))              # zero init -> exact no-op
        self.gamma._no_weight_decay = True
        self.mode = mode
        self.scale = head_dim ** -0.5

    def bias(self, gist, new_kv, *, m, first_window_position):
        return _gist_pool_bias(gist, self.W_g, self.gamma, new_kv, m=m,
                               first_window_position=first_window_position,
                               mode=self.mode, scale=self.scale)


# --------------------------------------------------------------------------- #
# Lightning Indexer  (copy of DeepseekV4Indexer + optional pooling-gate gist)   #
# --------------------------------------------------------------------------- #
class DeepseekV4IndexerGist(nn.Module):
    """Verbatim copy of `DeepseekV4Indexer` with an OPTIONAL zero-init gist bias on
    its compressed-key pooling gate. With `gist_index=False` (default) it is
    bit-identical to the upstream indexer."""

    rope_layer_type = "compress"

    def __init__(self, config, *, gist_index: bool = False, pcat: bool = False,
                 gist_dim: int = 32, inject_mode: str = "channel"):
        super().__init__()
        # >>> upstream DeepseekV4Indexer.__init__
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.scorer = DeepseekV4IndexerScorer(config)
        # <<< upstream
        self.gist_index = gist_index
        self.pcat = pcat
        if gist_index:
            self.gist_gate = _GistGate(gist_dim, self.head_dim, mode=inject_mode)
        if pcat:    # concat [token;g] -> additive g-projection into kv/gate (normal-init, live)
            self.pcat_kv = nn.Linear(gist_dim, 2 * self.head_dim, bias=False)
            self.pcat_gate = nn.Linear(gist_dim, 2 * self.head_dim, bias=False)
        self._gist = None    # set per-forward by the wrapper when conditioning is active
        self._last_index_scores = None   # stashed per-forward for STE selector training

    def forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
        # >>> upstream DeepseekV4Indexer.forward (verbatim except the GIST blocks)
        batch, seq_len, _ = hidden_states.shape
        cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if self.pcat and self._gist is not None:    # PCat: keys/values/gate derive from [token;g]
            kv = kv + self.pcat_kv(self._gist).to(kv.dtype)
            gate = gate + self.pcat_gate(self._gist).to(gate.dtype)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias

            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state("indexer", chunk_kv, chunk_gate, self.head_dim)
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            # --- GIST (zero-init no-op; conditions the indexer's key pooling) --- #
            if self.gist_index and self._gist is not None:
                bias = self.gist_gate.bias(self._gist, new_kv, m=self.compress_rate,
                                           first_window_position=first_window_position)
                new_gate = new_gate + bias.to(new_gate.dtype)
            # ------------------------------------------------------------------- #

            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        compressed_kv = (
            compressed if cache_layer is None else cache_layer.update_compressor_states("indexer", compressed)
        )

        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)

        index_scores = self.scorer(q, compressed_kv, hidden_states)  # [B, S, T]
        compressed_len = compressed_kv.shape[1]
        top_k = min(self.index_topk, compressed_len)

        if compressed_len > 0:
            causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
            entry_indices = torch.arange(compressed_len, device=index_scores.device)
            future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)  # [B, S, T]
            index_scores = index_scores.masked_fill(future_mask, float("-inf"))
            self._last_index_scores = index_scores      # [B,S,T] masked -- STE selector training
            top_k_indices = index_scores.topk(top_k, dim=-1).indices  # [B, S, k]
            invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
            return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

        self._last_index_scores = index_scores
        return index_scores.topk(top_k, dim=-1).indices
        # <<< upstream


# --------------------------------------------------------------------------- #
# CSA compressor  (copy of DeepseekV4CSACompressor + optional pooling-gate gist) #
# --------------------------------------------------------------------------- #
class DeepseekV4CSAGistCompressor(nn.Module):
    """Verbatim copy of `DeepseekV4CSACompressor` with OPTIONAL zero-init gist
    conditioning of (a) its own pooling gate (`gist_pool`) and (b) its Lightning
    Indexer's pooling gate (`gist_index`). With both flags False (default) it is
    bit-identical to the upstream compressor."""

    rope_layer_type = "compress"

    def __init__(self, config, *, gist_pool: bool = False, gist_index: bool = False,
                 pcat: bool = False, gist_dim: int = 32, inject_mode: str = "channel"):
        super().__init__()
        # >>> upstream DeepseekV4CSACompressor.__init__
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        # ALWAYS our indexer (bit-identical to upstream with flags off) so it can stash
        # index_scores for STE selector training; gist variant adds the conditioning.
        self.indexer = DeepseekV4IndexerGist(config, gist_index=gist_index, pcat=pcat,
                                             gist_dim=gist_dim, inject_mode=inject_mode)
        # <<< upstream
        self.gist_pool = gist_pool
        self.gist_index = gist_index
        self.pcat = pcat
        self.ste_topk = False                          # straight-through top-k (selector training)
        # --- indexer KL training (DeepSeek-V3.2/V4 recipe; see deepseek-indexer-kl-recipe) ---
        self.dense_mode = False     # dense warm-up: attend ALL causal blocks -> full KL teacher
        self.kl_mode = False        # detach indexer inputs; stash scores (student) + compressed_len
        self._kl_index_scores = None
        self._kl_teacher = None     # set by the attention wrapper: per-block main-attn mass [B,S,T]
        self._compressed_len = 0
        if gist_pool:
            self.gist_gate = _GistGate(gist_dim, self.head_dim, mode=inject_mode)
        if pcat:    # concat [token;g] -> additive g-projection into kv/gate (normal-init, live)
            self.pcat_kv = nn.Linear(gist_dim, 2 * self.head_dim, bias=False)
            self.pcat_gate = nn.Linear(gist_dim, 2 * self.head_dim, bias=False)
        self._gist = None    # set per-forward by the wrapper when conditioning is active

    def forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
        # >>> upstream DeepseekV4CSACompressor.forward (verbatim except the GIST blocks)
        batch, seq_len, _ = hidden_states.shape
        cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if self.pcat and self._gist is not None:    # PCat: keys/values/gate derive from [token;g]
            kv = kv + self.pcat_kv(self._gist).to(kv.dtype)
            gate = gate + self.pcat_gate(self._gist).to(gate.dtype)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias

            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "compressor", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            # --- GIST (zero-init no-op; conditions the compressor pooling) --- #
            if self.gist_pool and self._gist is not None:
                bias = self.gist_gate.bias(self._gist, new_kv, m=self.compress_rate,
                                           first_window_position=first_window_position)
                new_gate = new_gate + bias.to(new_gate.dtype)
            # ---------------------------------------------------------------- #

            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)

        # propagate the gist to the indexer (selection conditioning / pcat) before its call.
        # For KL indexer training we DETACH everything feeding the indexer, so its loss (KL to the
        # main attention) updates ONLY the indexer, never the main model (V3.2/V4 recipe).
        idx_hs = hidden_states.detach() if self.kl_mode else hidden_states
        idx_qr = q_residual.detach() if self.kl_mode else q_residual
        if (self.gist_index or self.pcat) and self._gist is not None:
            self.indexer._gist = self._gist.detach() if self.kl_mode else self._gist

        top_k_indices = self.indexer(idx_hs, idx_qr, position_ids, past_key_values, layer_idx)
        compressed_len = compressed_kv.shape[2]
        self._compressed_len = compressed_len
        if self.kl_mode:
            self._kl_index_scores = self.indexer._last_index_scores   # [B,S,T] student (future -inf)

        if self.dense_mode:
            # DENSE warm-up (KL eq 3): attend ALL causally-valid blocks (no top-k) so the captured
            # main-attention distribution is a full teacher over blocks for the indexer KL.
            thr = (position_ids + 1) // self.compress_rate           # [B,S] causal block threshold
            entry = torch.arange(compressed_len, device=compressed_kv.device)
            future = entry.view(1, 1, -1) >= thr.unsqueeze(-1)       # [B,S,T]
            bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
            bias = bias.masked_fill(future.unsqueeze(1), float("-inf"))
            return compressed_kv, bias

        valid = top_k_indices >= 0
        safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
        block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        bias = block_bias[..., :compressed_len]
        # --- STE selector training: hard top-k stays exact in the forward (0 on selected,
        # -inf elsewhere); in the backward, route gradient to the SELECTED blocks' indexer
        # scores so the LM loss trains the (otherwise non-differentiable) selector. --- #
        if self.ste_topk:
            sc = self.indexer._last_index_scores
            if sc is not None and sc.shape[-1] == compressed_len:
                sel = (bias == 0.0)                              # [B,1,S,T] selected (finite scores)
                ste = (sc - sc.detach()).unsqueeze(1)            # 0 fwd, grad bwd
                bias = bias + torch.where(sel, ste, torch.zeros_like(ste))
        return compressed_kv, bias
        # <<< upstream

    # ------------------------------------------------------------------ #
    def set_gist(self, g):
        """Wire the per-position gist readout [B, S, gist_dim] for this forward."""
        self._gist = g
        if isinstance(self.indexer, DeepseekV4IndexerGist):
            self.indexer._gist = g

    def gist_parameters(self):
        ps = []
        if self.gist_pool:
            ps += list(self.gist_gate.parameters())
        if self.gist_index:
            ps += list(self.indexer.gist_gate.parameters())
        if self.pcat:
            ps += list(self.pcat_kv.parameters()) + list(self.pcat_gate.parameters())
            ps += list(self.indexer.pcat_kv.parameters()) + list(self.indexer.pcat_gate.parameters())
        return ps
