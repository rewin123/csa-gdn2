"""OpenHermes-2.5 -> Alpaca-templated, packed GPT-2 token shards with an SFT loss mask.

Mixed ~10% into FineWeb pretraining so the (non-instruct) base learns the generic
"follow an instruction / answer in a slot" behavior -- the enabler for *zero/one-shot*
NIAH evaluation (we never train the needle task itself; see eval_niah.py).

Layout: each turn renders as
    ### Instruction:\n{human}\n\n### Response:\n{gpt}<eot>
with a per-token loss mask = 1 only on the response (+eot), 0 on prompt/template, so
training computes CE on the answer tokens (standard SFT masking). Packed into a uint16
token shard + a uint8 mask shard, both memmapped by InstructLoader.

  # box: tokenize ~30M instruct tokens (a few min)
  python src/data_instruct.py --target_tokens 30000000 --out_dir data/instruct
  # selftest the template/mask logic on CPU (no download):
  python src/data_instruct.py --selftest
"""
import argparse
import glob
import os

import numpy as np

# Alpaca-style template pieces (plain BPE, no special tokens).
SYS_PRE = "### System:\n"
INSTR_PRE = "### Instruction:\n"
RESP_PRE = "\n\n### Response:\n"
TURN_SEP = "\n\n"

_ALIASES = {"user": "human", "assistant": "gpt", "bot": "gpt", "model": "gpt"}


def _get_enc():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def _render_example(conv, enc, eot):
    """conv: list of {'from','value'} turns -> (token_ids list, loss_mask list).

    Mask is 1 on response (gpt) tokens + the eot that ends each response, else 0.
    Returns ([], []) for malformed / response-less examples (skip them).
    """
    toks, mask = [], []

    def add(text, m):
        ids = enc.encode_ordinary(text)
        toks.extend(ids)
        mask.extend([m] * len(ids))

    pending_instr = None
    saw_response = False
    for turn in conv:
        role = turn.get("from", "")
        role = _ALIASES.get(role, role)
        val = (turn.get("value") or "").strip()
        if not val:
            continue
        if role == "system":
            add(SYS_PRE + val + TURN_SEP, 0)
        elif role == "human":
            pending_instr = val
        elif role == "gpt":
            instr = pending_instr if pending_instr is not None else ""
            add(INSTR_PRE + instr + RESP_PRE, 0)        # prompt: no loss
            add(val, 1)                                 # response: loss
            toks.append(eot); mask.append(1)            # learn to stop
            pending_instr = None
            saw_response = True
    if not saw_response:
        return [], []
    return toks, mask


def download(target_tokens, out_dir, shard_tokens, val_tokens, dataset, nproc=8):
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    enc = _get_enc()
    eot = enc._special_tokens["<|endoftext|>"]
    ds = load_dataset(dataset, split="train", streaming=True)

    tbuf, mbuf, buf_len = [], [], 0
    total, shard_idx, wrote_val = 0, 0, False

    def flush(split):
        nonlocal tbuf, mbuf, buf_len, shard_idx
        if buf_len == 0:
            return
        tarr = np.concatenate(tbuf).astype(np.uint16)
        marr = np.concatenate(mbuf).astype(np.uint8)
        tag = "val" if split == "val" else f"train_{shard_idx:03d}"
        tarr.tofile(os.path.join(out_dir, f"{tag}.bin"))
        marr.tofile(os.path.join(out_dir, f"{tag}.mask"))
        print(f"[instr] wrote {tag}  {len(tarr):,} tokens  (total {total:,})", flush=True)
        if split == "train":
            shard_idx += 1
        tbuf, mbuf, buf_len = [], [], 0

    for ex in ds:
        conv = ex.get("conversations") or ex.get("conversation") or []
        toks, mask = _render_example(conv, enc, eot)
        if not toks:
            continue
        tbuf.append(np.array(toks, dtype=np.uint16))
        mbuf.append(np.array(mask, dtype=np.uint8))
        buf_len += len(toks); total += len(toks)
        cap = val_tokens if not wrote_val else shard_tokens
        if buf_len >= cap:
            if not wrote_val:
                flush("val"); wrote_val = True
            else:
                flush("train")
        if total >= target_tokens:
            break
    if not wrote_val:
        flush("val"); wrote_val = True
    flush("train")
    print(f"[instr] DONE total={total:,} tokens, {shard_idx} train shards + 1 val", flush=True)


class InstructLoader:
    """Random-crop ctx windows from packed (token, mask) shards; y is -100 on prompt+pad."""

    def __init__(self, out_dir, split, ctx, batch, device, seed=0):
        self.tfiles = sorted(glob.glob(os.path.join(out_dir, f"{split}_*.bin")))
        if split == "val":
            self.tfiles = sorted(glob.glob(os.path.join(out_dir, "val.bin")))
        assert self.tfiles, f"no {split} instruct shards in {out_dir}"
        self.tmm = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.tfiles]
        self.mmm = [np.memmap(f[:-4] + ".mask", dtype=np.uint8, mode="r") for f in self.tfiles]
        self.lens = [len(m) for m in self.tmm]
        self.ctx, self.bs, self.device = ctx, batch, device
        import torch
        self.g = torch.Generator().manual_seed(seed)

    def batch(self):
        import torch
        xs, ys = [], []
        for _ in range(self.bs):
            si = int(torch.randint(len(self.tmm), (1,), generator=self.g))
            i = int(torch.randint(self.lens[si] - self.ctx - 1, (1,), generator=self.g))
            tok = torch.from_numpy(self.tmm[si][i:i + self.ctx + 1].astype(np.int64))
            msk = torch.from_numpy(self.mmm[si][i:i + self.ctx + 1].astype(np.int64))
            x = tok[:-1]
            y = tok[1:].clone()
            y[msk[1:] == 0] = -100                      # loss only where target is a response token
            xs.append(x); ys.append(y)
        x = torch.stack(xs).to(self.device, non_blocking=True)
        y = torch.stack(ys).to(self.device, non_blocking=True)
        return x, y


def _selftest():
    enc = _get_enc()
    eot = enc._special_tokens["<|endoftext|>"]
    conv = [
        {"from": "system", "value": "You are helpful."},
        {"from": "human", "value": "What is the capital of France?"},
        {"from": "gpt", "value": "Paris."},
        {"from": "user", "value": "And of Japan?"},
        {"from": "assistant", "value": "Tokyo."},
    ]
    toks, mask = _render_example(conv, enc, eot)
    assert len(toks) == len(mask) and toks, "render failed"
    # response tokens (mask==1) should decode to the answers (+ eot)
    resp = enc.decode([t for t, m in zip(toks, mask) if m == 1 and t != eot])
    prompt = enc.decode([t for t, m in zip(toks, mask) if m == 0])
    n_resp = sum(mask)
    print("=== rendered (full) ===")
    print(enc.decode(toks).replace(enc.decode([eot]), "<eot>"))
    print("=== response-only (loss tokens) ===", repr(resp))
    print("=== prompt-only (no loss) ===", repr(prompt[:120]), "...")
    print(f"tokens={len(toks)} loss_tokens={n_resp} eot_count={toks.count(eot)}")
    assert "Paris" in resp and "Tokyo" in resp, "answers not in loss span"
    assert "capital of France" in prompt, "instruction leaked into loss span?"
    assert toks.count(eot) == 2, "expected one eot per response"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=int, default=30_000_000)
    ap.add_argument("--out_dir", default="data/instruct")
    ap.add_argument("--shard_tokens", type=int, default=50_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--dataset", default="teknium/OpenHermes-2.5")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    download(args.target_tokens, args.out_dir, args.shard_tokens, args.val_tokens, args.dataset)


if __name__ == "__main__":
    main()
