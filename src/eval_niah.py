"""Zero-shot + one-shot NIAH probe for the DeepSeek/C/PCat long-range comparison.

We never train the needle task (we already did supervised NIAH on the 1M toy). Instead we
mix generic instruct (OpenHermes, see data_instruct.py) into pretraining so the base learns
"answer in a slot", then test long-range retrieval as *zero-shot* (and *one-shot*, as a hedge
for the tiny 124M model) generalization. Success at distance d => the architecture genuinely
carries info across d, which is exactly the GDN-2 hypothesis.

Construction (same Alpaca template as instruct):
    ### Instruction:
    {filler}{FACT: "...is <V>."}{filler}

    {question}

    ### Response:
    <V>
The value V's token ids are spliced IDENTICALLY into the fact and the answer, so retrieval is a
literal token-space copy (easiest for a small model). distance d = answer_start - fact_V_start.
one-shot prepends one complete demo (different value, short context) in the SAME window.

Metrics (teacher-forced): answer-token CE (smooth) + answer-token accuracy (EM proxy), per d.

  python src/eval_niah.py --selftest      # CPU: build + decode + assert, no model
"""
import argparse

import numpy as np

INSTR_PRE = "### Instruction:\n"
RESP_PRE = "\n\n### Response:\n"

# (fact prefix that ends right before the value, question). Value is spliced as token ids.
FACTS = [
    ("By the way, the internal reference number for this report is",
     "What is the internal reference number mentioned in the text?"),
    ("Note that the assigned tracking code for the shipment is",
     "What is the assigned tracking code mentioned in the text?"),
    ("For the record, the building's access gate is numbered",
     "What is the gate number mentioned in the text?"),
    ("The catalog identifier printed on the label is",
     "What is the catalog identifier mentioned in the text?"),
]


def _get_enc():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def _value_ids(enc, rng):
    """A random 4-5 digit value, encoded WITH a leading space so the same ids splice cleanly."""
    v = int(rng.integers(10000, 99999))
    return enc.encode_ordinary(" " + str(v)), str(v)


def _fact_block(enc, fact_prefix, vids):
    """token ids for '{prefix}<V>.' with the value at a known offset -> (ids, v_offset)."""
    pre = enc.encode_ordinary(fact_prefix)
    dot = enc.encode_ordinary(".")
    return pre + vids + dot, len(pre)


def build_example(enc, sample_filler, rng, ctx, d_target, mode, eot):
    """Return dict(ids, answer_start, answer_ids, d, mode). filler crossing doc bounds is fine."""
    fact_prefix, question = FACTS[int(rng.integers(len(FACTS)))]
    vids, vstr = _value_ids(enc, rng)
    fact_ids, v_off = _fact_block(enc, fact_prefix, vids)

    instr = enc.encode_ordinary(INSTR_PRE)
    qblock = enc.encode_ordinary("\n\n" + question)
    rpre = enc.encode_ordinary(RESP_PRE)
    answer_ids = list(vids)                              # identical token ids => literal copy

    # tail from the value to the answer start (this is the distance we control):
    #   <after_filler> <qblock> <rpre> | <answer>
    fixed_tail = len(qblock) + len(rpre)
    after_n = max(0, d_target - fixed_tail - (len(fact_ids) - v_off))
    before_n = 48                                        # short lead-in before the fact

    def filler(n):
        return list(sample_filler(n)) if n > 0 else []

    body = filler(before_n) + fact_ids + filler(after_n) + qblock
    prompt = instr + body + rpre
    ids = prompt + answer_ids
    answer_start = len(prompt)
    v_start = len(instr) + before_n + v_off
    d = answer_start - v_start

    if mode == "one_shot":
        demo = _demo_block(enc, sample_filler, rng, eot)
        shift = len(demo)
        ids = demo + ids
        answer_start += shift

    # clamp to ctx (answer must survive); if over, trim leading filler of the body region
    if len(ids) > ctx:
        over = len(ids) - ctx
        # trim from the very front (demo/instr filler) -- keep fact..answer intact
        ids = ids[over:]
        answer_start -= over
    return {"ids": ids, "answer_start": answer_start, "answer_ids": answer_ids,
            "d": int(d), "mode": mode, "value": vstr}


def _demo_block(enc, sample_filler, rng, eot):
    """One complete short demonstration (different value) shown with its answer, then a separator."""
    fact_prefix, question = FACTS[int(rng.integers(len(FACTS)))]
    vids, _ = _value_ids(enc, rng)
    fact_ids, _ = _fact_block(enc, fact_prefix, vids)
    instr = enc.encode_ordinary(INSTR_PRE)
    qblock = enc.encode_ordinary("\n\n" + question)
    rpre = enc.encode_ordinary(RESP_PRE)
    pre = list(sample_filler(24))
    post = list(sample_filler(96))
    return instr + pre + fact_ids + post + qblock + rpre + list(vids) + enc.encode_ordinary("\n\n")


def fineweb_filler(shards_dir, split, seed):
    """A sampler(n) -> n contiguous real tokens from FineWeb shards (eot stripped)."""
    import glob, os
    files = sorted(glob.glob(os.path.join(shards_dir, f"{split}_*.bin")))
    assert files, f"no {split} shards in {shards_dir}"
    mm = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
    lens = [len(m) for m in mm]
    rng = np.random.default_rng(seed)
    eot = _get_enc()._special_tokens["<|endoftext|>"]

    def sample(n):
        si = int(rng.integers(len(mm)))
        i = int(rng.integers(0, max(1, lens[si] - n - 1)))
        chunk = np.asarray(mm[si][i:i + n], dtype=np.int64)
        chunk[chunk == eot] = enc_space                  # replace doc breaks with a space token
        return chunk
    enc_space = _get_enc().encode_ordinary(" ")[0]
    return sample


def make_eval_set(sample_filler, ctx, buckets, per_bucket, modes=("zero_shot", "one_shot"), seed=0):
    enc = _get_enc()
    eot = enc._special_tokens["<|endoftext|>"]
    rng = np.random.default_rng(seed)
    out = []
    for mode in modes:
        for d in buckets:
            for _ in range(per_bucket):
                out.append(build_example(enc, sample_filler, rng, ctx, d, mode, eot))
    return out


def score_niah(model, examples, device, eot, batch_size=8):
    """Teacher-forced CE + answer-token accuracy per (mode, d-bucket). Right-pad with eot."""
    import torch
    import torch.nn.functional as F
    was = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for s in range(0, len(examples), batch_size):
            grp = examples[s:s + batch_size]
            L = max(len(e["ids"]) for e in grp)
            X = torch.full((len(grp), L), eot, dtype=torch.long)
            for j, e in enumerate(grp):
                X[j, :len(e["ids"])] = torch.tensor(e["ids"], dtype=torch.long)
            X = X.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(X)
            for j, e in enumerate(grp):
                a, k = e["answer_start"], len(e["answer_ids"])
                lg = logits[j, a - 1:a - 1 + k].float()
                tgt = torch.tensor(e["answer_ids"], device=device)
                ce = F.cross_entropy(lg, tgt).item()
                acc = (lg.argmax(-1) == tgt).all().item()    # exact-match over the value tokens
                tokacc = (lg.argmax(-1) == tgt).float().mean().item()
                rows.append((e["mode"], e["d"], ce, float(acc), tokacc))
    if was:
        model.train()
    # aggregate per (mode, d)
    agg = {}
    for mode, d, ce, em, ta in rows:
        agg.setdefault((mode, d), []).append((ce, em, ta))
    summary = []
    for (mode, d), v in sorted(agg.items()):
        arr = np.array(v)
        summary.append({"mode": mode, "d": d, "n": len(v),
                        "ce": float(arr[:, 0].mean()),
                        "em": float(arr[:, 1].mean()),
                        "tok_acc": float(arr[:, 2].mean())})
    return summary


def _selftest():
    enc = _get_enc()
    eot = enc._special_tokens["<|endoftext|>"]
    rng = np.random.default_rng(0)
    # synthetic filler: safe mid-vocab tokens, no eot
    def filler(n):
        return rng.integers(100, 5000, size=n, dtype=np.int64)
    for mode in ("zero_shot", "one_shot"):
        e = build_example(enc, filler, rng, ctx=1024, d_target=512, mode=mode, eot=eot)
        a, k = e["answer_start"], len(e["answer_ids"])
        # the answer span must equal the spliced value ids (literal copy guarantee)
        assert e["ids"][a:a + k] == e["answer_ids"], "answer span mismatch"
        # and the value ids must ALSO appear earlier (the needle is in the haystack)
        before = e["ids"][:a]
        assert any(before[i:i + k] == e["answer_ids"] for i in range(len(before) - k)), "needle missing"
        dec = enc.decode(e["ids"])
        print(f"=== {mode}: d={e['d']} len={len(e['ids'])} value={e['value']} ===")
        tail = enc.decode(e["ids"][max(0, a - 40):a + k])
        print("   ...near answer:", repr(tail))
        assert e["value"] in dec, "value text missing"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return


if __name__ == "__main__":
    main()
