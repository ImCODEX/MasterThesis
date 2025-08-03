#!/usr/bin/env python3

import argparse, csv, random, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as spstats
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

#  Defaults 
DEF_BASELINE  = "Qwen/Qwen2.5-Coder-0.5B"
DEF_TUNED     = "./ppo_secure_qwen"
DEF_REWARD    = "./cvss_reward_model"
DEF_DATASET   = "Nan-Do/code-search-net-javascript"
DEF_SPLIT     = "train[-256:]"
DEF_OUTDIR    = "results/ppo"

#  Helpers 
def prep_tok(src: str):
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    if tok.bos_token_id is None:
        tok.add_special_tokens({"bos_token": tok.eos_token})
    tok.padding_side = "left"
    return tok

@torch.no_grad()
def generate(model, tok, ids, max_new=192):
    input_ids = torch.tensor([ids]).to(model.device)
    mask = (input_ids != tok.pad_token_id).long()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=mask,
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

@torch.no_grad()
def score(texts, reward_model, reward_tok):
    toks = reward_tok(
        texts, return_tensors="pt",
        padding=True, truncation=True, max_length=512
    ).to(reward_model.device)
    return reward_model(**toks).logits.squeeze(-1).cpu().tolist()

#  Main 
def main(argv):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Baseline vs PPO-tuned evaluation with CVSS reward model",
    )
    ap.add_argument("--baseline", default=DEF_BASELINE)
    ap.add_argument("--tuned",    default=DEF_TUNED)
    ap.add_argument("--reward",   default=DEF_REWARD)
    ap.add_argument("--dataset",  default=DEF_DATASET)
    ap.add_argument("--split",    default=DEF_SPLIT)
    ap.add_argument("--outdir",   default=DEF_OUTDIR)
    args = ap.parse_args(argv)

    out_dir = Path(args.outdir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("[Load models…]")
    base_tok  = prep_tok(args.baseline)
    tuned_tok = prep_tok(args.tuned)
    reward_tok= prep_tok(args.reward)

    baseline = AutoModelForCausalLM.from_pretrained(
        args.baseline, torch_dtype=dtype, device_map="auto"
    ).eval()
    tuned = AutoModelForCausalLM.from_pretrained(
        args.tuned, torch_dtype=dtype, device_map="auto"
    ).eval()
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward
    ).to(device).eval()

    ds = load_dataset(args.dataset, split=args.split)
    if "code" in ds.column_names:
        ds = ds.rename_column("code", "prompt")
    prompts = random.sample(list(ds["prompt"]), len(ds))

    base_scores, tuned_scores = [], []
    for i, p in enumerate(prompts, 1):
        ids = [base_tok.bos_token_id] + base_tok.encode(
            p, add_special_tokens=False, max_length=128, truncation=True
        )
        b_out = generate(baseline, base_tok, ids)
        t_out = generate(tuned,   tuned_tok,  ids)
        b_s, t_s = score([p + b_out, p + t_out], reward_model, reward_tok)
        base_scores.append(b_s); tuned_scores.append(t_s)
        if i % 50 == 0:
            print(f"  {i}/{len(prompts)} prompts processed")

    base_np, tuned_np = np.array(base_scores), np.array(tuned_scores)
    delta = tuned_np - base_np

    #  Metrics 
    print("\n[Results]")
    print(f" Baseline mean CVSS = {base_np.mean():.3f}")
    print(f" Tuned    mean CVSS = {tuned_np.mean():.3f}")
    win = (tuned_np < base_np).sum() / len(delta) * 100
    print(f" Win-rate (tuned safer) = {win:.1f}%")
    t_p = spstats.ttest_rel(base_np, tuned_np).pvalue
    w_p = spstats.wilcoxon(base_np, tuned_np).pvalue
    print(f" Paired t-test p = {t_p:.3e}")
    print(f" Wilcoxon p      = {w_p:.3e}")

    #  Plots 
    plt.figure(figsize=(5,5))
    plt.scatter(base_np, tuned_np, s=10, alpha=0.6)
    lims = [min(base_np.min(), tuned_np.min()), max(base_np.max(), tuned_np.max())]
    plt.plot(lims, lims, ls="--")
    plt.xlabel("Baseline CVSS"); plt.ylabel("Tuned CVSS")
    plt.tight_layout(); plt.savefig(out_dir/"plots"/"scatter.png"); plt.close()

    plt.figure(figsize=(6,4))
    bins = np.linspace(min(base_np.min(), tuned_np.min()),
                       max(base_np.max(), tuned_np.max()), 25)
    plt.hist(base_np, bins=bins, alpha=0.6, label="baseline",
             color="tab:blue", edgecolor="black")
    plt.hist(tuned_np, bins=bins, alpha=0.6, label="tuned",
             color="tab:orange", edgecolor="black")
    plt.legend(); plt.xlabel("CVSS"); plt.tight_layout()
    plt.savefig(out_dir/"plots"/"hist.png"); plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(delta, bins=20, alpha=0.7, color="tab:green", edgecolor="black")
    plt.axvline(0, ls="--")
    plt.xlabel("ΔCVSS (tuned−baseline)"); plt.tight_layout()
    plt.savefig(out_dir/"plots"/"delta_hist.png"); plt.close()

    def ecdf(x):
        x = np.sort(x)
        return x, np.arange(1, x.size + 1) / x.size

    xb, yb = ecdf(base_np)
    xt, yt = ecdf(tuned_np)
    plt.figure(figsize=(6,4))
    plt.step(xb, yb, where="post", label="baseline", color="tab:blue")
    plt.step(xt, yt, where="post", label="tuned",    color="tab:orange")
    plt.xlabel("CVSS"); plt.ylabel("Empirical CDF")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"plots"/"cdf.png"); plt.close()

    #  CSV ─
    with open(out_dir/"scores.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["prompt", "baseline_score", "tuned_score", "delta"])
        for p, b, t, d in zip(prompts, base_np, tuned_np, delta):
            wr.writerow([p.replace("\n", " ")[:180],
                         f"{b:.4f}", f"{t:.4f}", f"{d:.4f}"])
    print("Scores CSV saved →", out_dir/"scores.csv")
    print("Plots saved →", out_dir/"plots")

if __name__ == "__main__":
    main(sys.argv[1:])
