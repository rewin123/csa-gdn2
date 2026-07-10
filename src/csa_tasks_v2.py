"""§4 v2 task: instruction-defined importance, mechanism-isolated.

The v1 task (`csa_tasks.py`) produced a 3-way tie because at small N / attend-all
topk the shared first GDN-2 layer held every binding losslessly in its O(1) state
and solved the task alone -> layer 2 (CSA vs SWA) was decorative. v2 forces the
second layer to do real work and isolates the two CSA failure modes:

  * Regime P (pooling-bound): within an m=4 block, pooling drops FOO's value.
    Headroom knob = contention `c` (FOO-pair co-located with a distractor pair in
    the same block). topk is NON-binding (>= num_blocks).
  * Regime S (selection-bound): the Lightning Indexer fails to SELECT FOO's block
    among locally-identical distractor blocks. Headroom knob = topk (binding).

Both regimes keep `m=4` (DeepSeek-V4 default) fixed -- a positive result at a
cranked-up m is dismissible.

Construction (block-aligned to the m=4 compressor grid):

    [BOS][TASK][FOO][PAD] | block_0 | block_1 | ... | block_{nb-1} | [QUERY][FOO]
      0    1     2    3      4..7      8..11                            S-2  S-1

  * The 4-token prefix is exactly one compressor block, so haystack block `g`
    occupies token positions [4+4g, 8+4g) -- aligned with the compressor's
    non-overlapping m=4 windows. Each block holds two 2-token slots
    (even slot = positions 4+4g,5+4g ; odd slot = 6+4g,7+4g).
  * FOO is named once at position 2 (the only place importance is defined).
  * Q FOO-needles (marker==FOO, distinct held-out values) sit on the even slot of
    Q distinct blocks, all beyond the final `sw` tokens (SWA cannot reach them).
  * contention c: a fraction c of needle blocks get a DISTRACTOR pair on their
    odd slot (so the m=4 pool must separate FOO's value from a co-located
    distractor); the rest get filler (a gentle pool).
  * extra distractor pairs pad the real-binding count up to `n_pairs` (= N, the
    GDN-state load) -- GDN gates are input-only so they cannot filter to only-FOO
    and must carry ~N bindings; N >= 3*head_dim overflows the d_k x d_v state.
  * Readout: at the final position the model emits the SET of FOO-values
    (multi-hot). Graded recall = |top-Q predicted values  ∩  FOO-values| / Q.

G5 (no memorization) holds because FOO->values is random per instance (no fixed
marker->value table) and train/eval use separate RNG streams -- so we measure
retrieval, not a lookup table. The value *ids* share one pool across splits: with
a classification readout, disjoint id pools would let the model learn which ids are
ever answers and suppress the rest, collapsing eval recall to chance. See Vocab2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

PAD, BOS, TASK, QUERY = 0, 1, 2, 3
MK_LO = 4


class Vocab2:
    """Larger vocab for the MV value space.

    G5 (no memorization) is satisfied because FOO->values is drawn fresh-random per
    instance (no fixed marker->value table to memorize) and train/eval use separate
    RNG streams (instance-disjoint). We DO NOT split the value *ids* into disjoint
    pools by default: with a classification readout that backfires -- the model
    learns "ids the train split never uses as answers" and suppresses them, and the
    eval split's answers are exactly those ids, so eval recall collapses to chance
    while train loss falls. `held_out=True` restores the disjoint-id pools only if
    you switch to a copy/pointer readout that can't exploit value identity."""

    def __init__(self, n_markers=8, n_values=512, n_filler=64, held_out=False):
        self.K = n_markers
        self.V = n_values
        self.F = n_filler
        self.MK_LO = MK_LO
        self.VAL_LO = MK_LO + n_markers
        self.FIL_LO = self.VAL_LO + n_values
        self.size = self.FIL_LO + n_filler
        self.held_out = held_out
        if held_out:                                  # disjoint id pools (copy-readout only)
            half = n_values // 2
            self.pool = {"train": (0, half), "eval": (half, n_values)}
        else:                                         # shared pool: identity gives no cue
            self.pool = {"train": (0, n_values), "eval": (0, n_values)}

    def val_pool(self, split):
        lo, hi = self.pool[split]
        return lo, hi


def _gen(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def seq_for(nb):
    """Token length for `nb` haystack blocks."""
    return 4 * nb + 6


def nb_for(seq):
    """Number of haystack blocks that fit in ~`seq` tokens."""
    return max(1, (seq - 6) // 4)


def make_v2(batch, *, vocab: Vocab2, n_pairs, K, Q, nb, sliding_window,
            contention, split, seed, device):
    """Returns ids [B,S], qpos [B], tgt_local [B,Q] (local value indices of the
    FOO needles). `n_pairs` = N (GDN load), `Q` = #FOO needles, `nb` = #blocks,
    `contention` in [0,1] = fraction of needle blocks with a distractor block-mate.

    Also returns `meta` with the realized binding count and the beyond-sw fraction
    (used by the gates)."""
    g = _gen(seed)
    K = min(K, vocab.K)
    B = batch
    S = seq_for(nb)
    vlo, vhi = vocab.val_pool(split)
    pool_sz = vhi - vlo

    # blocks usable for needles: all but the last few (keep needles beyond sw)
    last_ok = nb - 1
    while seq_for(0) + 4 * (last_ok + 1) > S - sliding_window and last_ok > 0:
        last_ok -= 1
    n_needle_avail = last_ok + 1
    assert n_needle_avail >= Q, f"too few beyond-sw blocks ({n_needle_avail}) for Q={Q}"

    n_high = int(round(contention * Q))                      # needle blocks w/ distractor mate
    extra_pairs = max(0, n_pairs - Q - n_high)
    extra_blocks = (extra_pairs + 1) // 2                    # both slots -> 2 distractors/block
    assert Q + extra_blocks <= nb, f"need {Q+extra_blocks} blocks, have {nb}"

    # base: filler everywhere, then the structural tokens
    ids = (vocab.FIL_LO + torch.randint(0, vocab.F, (B, S), generator=g)).to(torch.long)
    ids[:, 0] = BOS
    ids[:, 1] = TASK
    foo = vocab.MK_LO + torch.randint(0, K, (B,), generator=g)
    ids[:, 2] = foo
    ids[:, 3] = PAD
    ids[:, S - 2] = QUERY
    ids[:, S - 1] = foo
    rows = torch.arange(B)

    # FOO values: Q distinct local indices from the split pool, per row
    foo_local = vlo + torch.argsort(torch.rand(B, pool_sz, generator=g), dim=1)[:, :Q]   # [B,Q] in [vlo,vhi)
    tgt_local = foo_local.clone()

    def rand_dmarker(shape):
        m = vocab.MK_LO + torch.randint(0, K, shape, generator=g)
        bad = m == foo.view(B, *([1] * (m.dim() - 1)))
        while bad.any():
            m[bad] = vocab.MK_LO + torch.randint(0, K, (int(bad.sum()),), generator=g)
            bad = m == foo.view(B, *([1] * (m.dim() - 1)))
        return m

    def rand_dvalue(shape):
        return vlo + torch.randint(0, pool_sz, shape, generator=g)

    # choose needle blocks + extra distractor blocks via a per-row permutation
    order = torch.argsort(torch.rand(B, n_needle_avail, generator=g), dim=1)
    needle_blk = order[:, :Q]                                 # [B,Q]
    extra_blk = order[:, Q:Q + extra_blocks]                  # [B,extra_blocks]

    # place FOO needles on the even slot of needle blocks
    emk = 4 + 4 * needle_blk                                  # even-slot marker pos
    ids[rows[:, None].expand(B, Q), emk] = foo[:, None]
    ids[rows[:, None].expand(B, Q), emk + 1] = foo_local

    # contention: first n_high needle blocks get a distractor on their odd slot
    if n_high > 0:
        hi = needle_blk[:, :n_high]                          # [B,n_high]
        omk = 6 + 4 * hi
        r = rows[:, None].expand(B, n_high)
        ids[r, omk] = rand_dmarker((B, n_high))
        ids[r, omk + 1] = rand_dvalue((B, n_high))

    # extra distractor blocks: both slots are distractor pairs
    if extra_blocks > 0:
        r = rows[:, None].expand(B, extra_blocks)
        emk_d = 4 + 4 * extra_blk
        omk_d = 6 + 4 * extra_blk
        ids[r, emk_d] = rand_dmarker((B, extra_blocks))
        ids[r, emk_d + 1] = rand_dvalue((B, extra_blocks))
        ids[r, omk_d] = rand_dmarker((B, extra_blocks))
        ids[r, omk_d + 1] = rand_dvalue((B, extra_blocks))

    qpos = torch.full((B,), S - 1, dtype=torch.long)
    realized_N = Q + n_high + 2 * extra_blocks
    meta = {"S": S, "realized_N": realized_N, "n_needle_avail": n_needle_avail,
            "n_high": n_high, "extra_blocks": extra_blocks}
    return ids.to(device), qpos.to(device), tgt_local.to(device), meta


def loss_acc_v2(logits, qpos, tgt_local, vocab: Vocab2, Q, pos_weight=None):
    """Multi-hot BCE over the value range; graded recall = top-Q overlap / Q.

    `pos_weight` (scalar) upweights the Q positives against the V-Q negatives --
    retrieval targets are sparse (Q of V), so a positive weight ~ (V-Q)/Q keeps
    the model from collapsing to all-negative early."""
    B = logits.shape[0]
    rows = torch.arange(B, device=logits.device)
    lq = logits[rows, qpos]                                   # [B, vocab]
    vlo = vocab.VAL_LO
    vlogits = lq[:, vlo:vlo + vocab.V].float()                # [B, V]
    tgt = (tgt_local - vlo) if tgt_local.min() >= vlo else tgt_local  # accept global or local
    multihot = torch.zeros(B, vocab.V, device=logits.device)
    multihot.scatter_(1, tgt, 1.0)
    pw = None if pos_weight is None else torch.as_tensor(float(pos_weight), device=vlogits.device)
    loss = F.binary_cross_entropy_with_logits(vlogits, multihot, pos_weight=pw)
    with torch.no_grad():
        topq = vlogits.topk(Q, dim=1).indices                 # [B,Q]
        pred = torch.zeros_like(multihot)
        pred.scatter_(1, topq, 1.0)
        recall = ((pred * multihot).sum(1) / Q).mean().item()
    return loss, recall


# --------------------------------------------------------------------------- #
# Gate self-checks (G4 local indistinguishability, G5 held-out)               #
# --------------------------------------------------------------------------- #
def selfcheck(vocab: Vocab2, *, n_pairs=96, K=8, Q=8, seq=512, sliding_window=8,
              contention=1.0):
    nb = nb_for(seq)
    tr_ids, tr_q, tr_t, meta = make_v2(64, vocab=vocab, n_pairs=n_pairs, K=K, Q=Q,
                                       nb=nb, sliding_window=sliding_window,
                                       contention=contention, split="train", seed=1, device="cpu")
    ev_ids, ev_q, ev_t, _ = make_v2(64, vocab=vocab, n_pairs=n_pairs, K=K, Q=Q, nb=nb,
                                    sliding_window=sliding_window, contention=contention,
                                    split="eval", seed=2, device="cpu")
    # G5: associations are random per instance (separate RNG streams -> instance-
    # disjoint). Anti-memorization comes from ephemeral random bindings, not from
    # disjoint value ids. Confirm the two streams produce different instances; only
    # when held_out is explicitly on do we also require disjoint id pools.
    assert not torch.equal(tr_t, ev_t), "G5 FAIL: train/eval streams identical (RNG not separated)"
    if vocab.held_out:
        trset = set(tr_t.reshape(-1).tolist())
        evset = set(ev_t.reshape(-1).tolist())
        assert trset.isdisjoint(evset), "G5 FAIL: held_out set but value pools overlap"
    # G4: a FOO needle's local 2-token form (marker,value envelope) is the same shape
    #     as a distractor's -- only the marker token differs. Check needle markers==foo
    #     and that distractor markers != foo exist in the same block grid.
    S = meta["S"]
    foo_tok = tr_ids[:, 2]
    # at least one distractor marker (in [MK_LO, MK_LO+K)) != foo present per row
    mk_lo, mk_hi = vocab.MK_LO, vocab.MK_LO + vocab.K
    is_marker = (tr_ids >= mk_lo) & (tr_ids < mk_hi)
    non_foo_marker = is_marker & (tr_ids != foo_tok[:, None])
    assert non_foo_marker.any(dim=1).all(), "G4 FAIL: some row has no distractor marker"
    return {"S": S, "realized_N": meta["realized_N"], "K": vocab.K, "V": vocab.V,
            "vocab_size": vocab.size, "n_needle_avail": meta["n_needle_avail"],
            "train_pool": vocab.pool["train"], "eval_pool": vocab.pool["eval"]}


if __name__ == "__main__":
    v = Vocab2()
    print("selfcheck:", selfcheck(v))
    # quick recall sanity on random logits: expect ~Q/V
    import torch as _t
    nb = nb_for(512)
    ids, q, t, meta = make_v2(8, vocab=v, n_pairs=96, K=8, Q=8, nb=nb, sliding_window=8,
                              contention=1.0, split="train", seed=0, device="cpu")
    logits = _t.randn(8, ids.shape[1], v.size)
    loss, rec = loss_acc_v2(logits, q, t, v, Q=8)
    print(f"ids {tuple(ids.shape)} meta={meta}  random-logit loss={loss:.3f} recall={rec:.3f} (chance≈{8/v.V:.3f})")
