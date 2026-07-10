"""FineWeb (sample-10BT) -> GPT-2 BPE token shards (uint16) + a memmap LM loader.

Mirrors the llm.c / nanoGPT recipe so a 124M val loss here is comparable to the
published GPT-2-124M FineWeb val loss (~3.28). Streams the HF dataset and stops at
--target_tokens, so a box smoke pulls a few hundred M tokens while the vast.ai run
pulls the full ~10B. First --val_tokens go to a held-out val shard.

  # box smoke / small run (~200M tokens, a few min):
  python src/data_fineweb.py --target_tokens 200000000 --out_dir data/fineweb
  # full run (vast.ai):
  python src/data_fineweb.py --target_tokens 10000000000 --out_dir data/fineweb

Loader:
  from data_fineweb import ShardLoader
  tr = ShardLoader("data/fineweb", "train", ctx=2048, batch=8, device="cuda")
  x, y = tr.batch()            # x,y: [batch, ctx] long, y = x shifted by 1
"""
import argparse
import glob
import os

import numpy as np


def _tokenize_chunk(texts):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    out = []
    for t in texts:
        out.append(eot)
        out.extend(enc.encode_ordinary(t))
    return np.array(out, dtype=np.uint16)


def download(target_tokens, out_dir, shard_tokens, val_tokens, batch_docs=512, nproc=8):
    from datasets import load_dataset
    from multiprocessing import Pool
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    buf = []                       # current shard's tokens (list of np arrays)
    buf_len = 0
    total = 0
    shard_idx = 0
    wrote_val = False

    def flush(split):
        nonlocal buf, buf_len, shard_idx
        if buf_len == 0:
            return
        arr = np.concatenate(buf)
        name = "val_000.bin" if split == "val" else f"train_{shard_idx:03d}.bin"
        path = os.path.join(out_dir, name)
        arr.tofile(path)
        print(f"[data] wrote {path}  {len(arr):,} tokens  (total {total:,})", flush=True)
        if split == "train":
            shard_idx += 1
        buf, buf_len = [], 0

    pending = []
    with Pool(nproc) as pool:
        def feed():
            nonlocal pending
            if not pending:
                return
            groups = [pending[i:i + batch_docs] for i in range(0, len(pending), batch_docs)]
            for arr in pool.imap(_tokenize_chunk, groups):
                yield arr
            pending = []

        for ex in ds:
            pending.append(ex["text"])
            if len(pending) < batch_docs * nproc:
                continue
            for arr in feed():
                buf.append(arr); buf_len += len(arr); total += len(arr)
                cap = val_tokens if not wrote_val else shard_tokens
                if buf_len >= cap:
                    if not wrote_val:
                        flush("val"); wrote_val = True
                    else:
                        flush("train")
                if total >= target_tokens:
                    break
            if total >= target_tokens:
                break
    if not wrote_val:
        flush("val"); wrote_val = True
    flush("train")
    print(f"[data] DONE total={total:,} tokens, {shard_idx} train shards + 1 val", flush=True)


class ShardLoader:
    """Random-crop LM batches from memmapped uint16 shards."""

    def __init__(self, out_dir, split, ctx, batch, device, seed=0):
        self.files = sorted(glob.glob(os.path.join(out_dir, f"{split}_*.bin")))
        assert self.files, f"no {split} shards in {out_dir}"
        self.mmaps = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.lens = [len(m) for m in self.mmaps]
        self.ctx, self.bs, self.device = ctx, batch, device
        import torch
        self.g = torch.Generator().manual_seed(seed)
        self.total = sum(self.lens)

    def batch(self):
        import torch
        xs, ys = [], []
        for _ in range(self.bs):
            si = int(torch.randint(len(self.mmaps), (1,), generator=self.g))
            m = self.mmaps[si]
            i = int(torch.randint(self.lens[si] - self.ctx - 1, (1,), generator=self.g))
            chunk = torch.from_numpy(m[i:i + self.ctx + 1].astype(np.int64))
            xs.append(chunk[:-1]); ys.append(chunk[1:])
        x = torch.stack(xs).to(self.device, non_blocking=True)
        y = torch.stack(ys).to(self.device, non_blocking=True)
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=int, default=200_000_000)
    ap.add_argument("--out_dir", default="data/fineweb")
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--val_tokens", type=int, default=10_000_000)
    ap.add_argument("--nproc", type=int, default=8)
    args = ap.parse_args()
    download(args.target_tokens, args.out_dir, args.shard_tokens, args.val_tokens, nproc=args.nproc)


if __name__ == "__main__":
    main()
