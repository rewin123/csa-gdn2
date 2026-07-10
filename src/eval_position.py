"""Per-position CE eval (positional_loss) for the long-range comparison.

Aggregate val loss averages over all positions and hides long-range effects. Bucketing CE by
token position exposes them: if GDN-2 helps, C/PCat should pull ahead at LATE positions (more
context to exploit). Logged as SEPARATE entities for FineWeb and OpenHermes.

FineWeb needs DOC-ALIGNED windows: the normal random-crop loader spans <eot> boundaries, so
"position p" != "p tokens of same-document context". DocAlignedLoader yields windows that start
right after an <eot> and contain no internal <eot>, so position p = exactly p tokens of one
document -> a clean proxy for available context length.

evaluate_per_position works for both: it honors -100 targets (OpenHermes response masking), so
the same function buckets FineWeb (clean) and OpenHermes (response-only) CE by position.

  python src/eval_position.py --selftest    # CPU: synthesize a shard, check doc-aligned starts
"""
import glob
import os

import numpy as np

EOT = 50256                                      # GPT-2 <|endoftext|>


class DocAlignedLoader:
    """Yield ctx-length windows that lie entirely inside one document (no internal <eot>)."""

    def __init__(self, shards_dir, split, ctx, batch, device, seed=0, eot=EOT):
        files = sorted(glob.glob(os.path.join(shards_dir, f"{split}_*.bin")))
        if not files:                            # FineWeb val shard is named val_000.bin
            files = sorted(glob.glob(os.path.join(shards_dir, f"{split}*.bin")))
        assert files, f"no {split} shards in {shards_dir}"
        self.mmaps = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.ctx, self.bs, self.device, self.eot = ctx, batch, device, eot
        # precompute valid (shard, start) pairs: start just after an <eot>, no <eot> in [start, start+ctx]
        self.starts = []                         # list of (shard_idx, start)
        for si, m in enumerate(self.mmaps):
            arr = np.asarray(m)
            eots = np.flatnonzero(arr == eot)
            for e in eots:
                s = e + 1
                if s + ctx + 1 > len(arr):
                    continue
                pos = np.searchsorted(eots, s)           # first eot index at/after s
                nxt = int(eots[pos]) if pos < len(eots) else len(arr)
                if nxt >= s + ctx + 1:                   # no eot inside the window
                    self.starts.append((si, int(s)))
        assert self.starts, f"no doc-aligned windows of ctx={ctx} in {split} (docs too short?)"
        import torch
        self.g = torch.Generator().manual_seed(seed)

    def batch(self):
        import torch
        xs, ys = [], []
        for _ in range(self.bs):
            j = int(torch.randint(len(self.starts), (1,), generator=self.g))
            si, s = self.starts[j]
            chunk = torch.from_numpy(self.mmaps[si][s:s + self.ctx + 1].astype(np.int64))
            xs.append(chunk[:-1]); ys.append(chunk[1:])
        x = torch.stack(xs).to(self.device, non_blocking=True)
        y = torch.stack(ys).to(self.device, non_blocking=True)
        return x, y


def evaluate_per_position(model, loader, iters, ctx, bucket=128):
    """Mean CE per position bucket. Honors -100 targets (masked positions excluded).

    Returns list of dicts: [{pos: bucket_center, ce: mean_ce, n: token_count}, ...].
    """
    import torch
    import torch.nn.functional as F
    was = model.training
    model.eval()
    n_buckets = (ctx + bucket - 1) // bucket
    ce_sum = np.zeros(n_buckets, dtype=np.float64)
    ce_cnt = np.zeros(n_buckets, dtype=np.int64)
    with torch.no_grad():
        for _ in range(iters):
            x, y = loader.batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            B, T, V = logits.shape
            ce = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1),
                                 ignore_index=-100, reduction="none").reshape(B, T)
            valid = (y != -100)
            for b in range(n_buckets):
                lo, hi = b * bucket, min((b + 1) * bucket, T)
                if lo >= T:
                    break
                seg = ce[:, lo:hi][valid[:, lo:hi]]
                if seg.numel():
                    ce_sum[b] += seg.sum().item()
                    ce_cnt[b] += seg.numel()
    if was:
        model.train()
    out = []
    for b in range(n_buckets):
        if ce_cnt[b]:
            out.append({"pos": b * bucket + bucket // 2,
                        "ce": ce_sum[b] / ce_cnt[b], "n": int(ce_cnt[b])})
    return out


def _selftest():
    # synthesize a shard: a few docs of varying length separated by <eot>
    import tempfile
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    parts = []
    doc_lens = [300, 50, 600, 40, 700]                 # only the >ctx docs yield windows
    for L in doc_lens:
        parts.append(np.array([EOT], dtype=np.uint16))
        parts.append(rng.integers(100, 5000, size=L, dtype=np.uint16))
    arr = np.concatenate(parts)
    arr.tofile(os.path.join(d, "val_000.bin"))
    ctx = 128
    ld = DocAlignedLoader(d, "val", ctx=ctx, batch=2, device="cpu", seed=0)
    # docs of length >= ctx+1 (300,600,700) should each contribute exactly one valid start
    big = sum(1 for L in doc_lens if L >= ctx + 1)
    print(f"doc_lens={doc_lens} ctx={ctx} -> {len(ld.starts)} doc-aligned starts (expect {big})")
    assert len(ld.starts) == big, "wrong number of doc-aligned windows"
    # every window must be eot-free
    for si, s in ld.starts:
        win = np.asarray(ld.mmaps[si][s:s + ctx + 1])
        assert (win != EOT).all(), "window contains an <eot>"
    print("SELFTEST OK")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()


if __name__ == "__main__":
    main()
