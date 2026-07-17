"""AHR vs LoRA vs ICL at |D_train| = 10 on SST-2, across two base models.

Tests the pre-registered predictions in ../predictions.json. See ../hyperparams.md for the
provenance of every constant below.

Three arms per (model, seed):
  ICL  — frozen model, 10-shot prompt, no training at all
  AHR  — one scalar per attention head, IC-FT trained (our reimplementation, Eq. 3/5)
  LoRA — r=1 alpha=1, IC-FT trained, identical lr/steps/seeds/prompt (peft)

The AHR and ICL arms share the same 10-shot prompt, so AHR is measured as the paper measures
it: ICL *plus* trained head reweighting over the same ten examples.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ahr import AHR

# --- hyperparameters (see ../hyperparams.md; "assumed" ones flagged there) ---
K_SHOT = 10          # paper 3.1, never varied
N_TRAIN = 10         # |D_train|, the disputed regime
LR = 3e-3            # middle of paper's grid {0.03, 0.003, 0.0003}
STEPS = 50           # from paper's grid {0,5,10,20,50,100,200}
LAMBDA = 0.01        # small L2 on gamma, paper Eq. 4 with p=2
P_NORM = 2
N_EVAL = 200         # fixed subsample of SST-2 validation
SEEDS = [0, 1, 2]

INSTRUCTION = (
    "Classify the sentiment of each movie review as Positive or Negative."
)
LABELS = [" Negative", " Positive"]   # SST-2: 0 = negative, 1 = positive


def build_shots(tok, examples):
    """Tokenize the k-shot IC-FT sequence, returning ids and each answer's first-token index.

    Sequence: INSTRUCTION + concat_i("\n\nReview: {x_i}\nSentiment:" + " {y_i}")

    Causal masking means the logit at position (answer_pos[i] - 1) conditions on exactly
    {x_1, y_1, ..., y_{i-1}, x_i} = X_i of Eq. 5. So one forward pass over this sequence
    yields all k terms of Eq. 5 -- mathematically identical to k separate passes, k x cheaper.
    """
    ids = tok(INSTRUCTION, add_special_tokens=False)["input_ids"]
    answer_pos, answer_tok = [], []
    for sent, label in examples:
        q = tok(f"\n\nReview: {sent.strip()}\nSentiment:", add_special_tokens=False)["input_ids"]
        a = tok(LABELS[label], add_special_tokens=False)["input_ids"]
        ids += q
        answer_pos.append(len(ids))     # index of the answer's FIRST token
        answer_tok.append(a[0])         # paper 3.1: score first answer token only
        ids += a
    return ids, answer_pos, answer_tok


def label_token_ids(tok):
    """First-token id of each label verbalizer. Must be distinct or scoring is meaningless."""
    out = [tok(l, add_special_tokens=False)["input_ids"][0] for l in LABELS]
    assert out[0] != out[1], f"label verbalizers collide: {out}"
    return out


def ic_ft_loss(model, ids, answer_pos, answer_tok, device):
    """Eq. 5: sum_i -log P(y_i | X_i), first answer token only, in one forward pass."""
    x = torch.tensor([ids], device=device)
    logits = model(x).logits[0]                       # (seq, vocab)
    pred_at = torch.tensor([p - 1 for p in answer_pos], device=device)
    tgt = torch.tensor(answer_tok, device=device)
    return F.cross_entropy(logits[pred_at], tgt)


@torch.no_grad()
def evaluate(model, tok, prefix_ids, eval_set, lab_ids, device):
    """Accuracy: one forward pass per test example, argmax over the two label logits.

    Also returns the fraction predicted positive: a degenerate always-one-class predictor
    scores ~50% on balanced SST-2 and would otherwise be indistinguishable from a real
    coin-flip result. We need to be able to tell those apart.
    """
    correct, n_pos = 0, 0
    for sent, gold in eval_set:
        q = tok(f"\n\nReview: {sent.strip()}\nSentiment:", add_special_tokens=False)["input_ids"]
        x = torch.tensor([prefix_ids + q], device=device)
        logits = model(x).logits[0, -1]
        pred = int(logits[lab_ids[1]] > logits[lab_ids[0]])
        n_pos += pred
        correct += int(pred == gold)
    return 100.0 * correct / len(eval_set), n_pos / len(eval_set)


def run_arm(arm, model_name, seed, train_ex, eval_set, device, log):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    lab_ids = label_token_ids(tok)
    ids, answer_pos, answer_tok = build_shots(tok, train_ex)
    t0 = time.time()
    n_trainable, final_loss = 0, None

    if arm == "icl":
        pass  # frozen model, no training -- this is the "do nothing" arm

    elif arm == "ahr":
        ahr = AHR(model)
        n_trainable = ahr.n_params
        opt = torch.optim.AdamW([ahr.gamma], lr=LR)
        for step in range(STEPS):
            opt.zero_grad()
            loss = ic_ft_loss(model, ids, answer_pos, answer_tok, device)
            (loss + ahr.penalty(LAMBDA, P_NORM)).backward()
            opt.step()
            if step in (0, STEPS - 1):
                log(f"      [ahr] step {step:3d} loss {loss.item():.4f}")
        final_loss = loss.item()
        log(f"      [ahr] |gamma| mean {ahr.gamma.abs().mean():.4f} "
            f"max {ahr.gamma.abs().max():.4f}")

    elif arm == "lora":
        from peft import LoraConfig, get_peft_model
        targets = ["c_attn"] if "gpt2" in model_name else ["q_proj", "v_proj"]
        cfg = LoraConfig(r=1, lora_alpha=1, lora_dropout=0.0, bias="none",
                         target_modules=targets, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=LR)
        for step in range(STEPS):
            opt.zero_grad()
            loss = ic_ft_loss(model, ids, answer_pos, answer_tok, device)
            loss.backward()
            opt.step()
            if step in (0, STEPS - 1):
                log(f"      [lora] step {step:3d} loss {loss.item():.4f}")
        final_loss = loss.item()

    if final_loss is not None and not np.isfinite(final_loss):
        raise RuntimeError(f"{arm} training diverged: final loss = {final_loss}")

    train_s = time.time() - t0
    t1 = time.time()
    acc, frac_pos = evaluate(model, tok, ids, eval_set, lab_ids, device)
    log(f"    {arm:5s} seed {seed}: acc {acc:.2f}  frac_pred_pos {frac_pos:.2f}  "
        f"(train {train_s:.0f}s, eval {time.time()-t1:.0f}s, {n_trainable} params)")
    del model
    return {"acc": acc, "frac_pred_pos": frac_pos, "n_trainable": n_trainable,
            "final_train_loss": final_loss, "train_s": round(train_s, 1),
            "eval_s": round(time.time() - t1, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    ap.add_argument("--arms", nargs="+", default=["icl", "ahr", "lora"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--out", default="../results.json")
    args = ap.parse_args()

    torch.set_num_threads(10)
    device = "cpu"

    def log(m):
        print(m, flush=True)

    ds = load_dataset("stanfordnlp/sst2")
    train_pool = [(r["sentence"], r["label"]) for r in ds["train"]]
    val_pool = [(r["sentence"], r["label"]) for r in ds["validation"]]

    # Fixed eval subsample: identical across every model, arm and seed.
    rng = random.Random(1234)
    eval_set = rng.sample(val_pool, args.n_eval)
    log(f"eval set: {len(eval_set)} examples, "
        f"{sum(g for _, g in eval_set)} positive / {len(eval_set)} total")

    results = {"config": {"k_shot": K_SHOT, "n_train": N_TRAIN, "lr": LR, "steps": STEPS,
                          "lambda": LAMBDA, "p_norm": P_NORM, "n_eval": args.n_eval,
                          "seeds": args.seeds, "dataset": "stanfordnlp/sst2",
                          "cv": False, "note": "fixed hyperparameters, NO cross-validation"},
               "runs": {}}
    try:
        results.update(json.load(open(args.out)))   # resume/extend a partial run
    except Exception:
        pass
    results.setdefault("runs", {})

    for model_name in args.models:
        log(f"\n=== {model_name} ===")
        for seed in args.seeds:
            # seed controls WHICH 10 training examples are sampled, and their order
            r = random.Random(seed)
            train_ex = r.sample(train_pool, N_TRAIN)
            np.random.seed(seed)
            torch.manual_seed(seed)
            for arm in args.arms:
                key = f"{model_name}|{arm}|{seed}"
                if key in results["runs"]:
                    log(f"    {arm:5s} seed {seed}: cached, skipping")
                    continue
                try:
                    results["runs"][key] = run_arm(arm, model_name, seed, train_ex,
                                                   eval_set, device, log)
                except Exception as e:
                    log(f"    {arm:5s} seed {seed}: FAILED — {type(e).__name__}: {e}")
                    results["runs"][key] = {"error": f"{type(e).__name__}: {e}"}
                # write after every arm: a killed run keeps everything already earned
                json.dump(results, open(args.out, "w"), indent=2)

    log("\nwrote " + args.out)


if __name__ == "__main__":
    main()
