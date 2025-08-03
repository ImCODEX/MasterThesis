#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from scipy import stats as spstats

DEFAULT_BASELINE = "Qwen/Qwen2.5-Coder-0.5B"
DEFAULT_TUNED = "./grpo_secure_qwen"
DEFAULT_REWARD = "./cvss_reward_model"
DEFAULT_DATASET = "Nan-Do/code-search-net-javascript"
DEFAULT_SPLIT = "train[-256:]"
DEFAULT_OUTDIR = "results/grpo"
DEFAULT_MAX = 256

def _prep_tok(src: str):
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "left"
    if tok.bos_token_id is None:
        tok.add_special_tokens({"bos_token": tok.eos_token})
    return tok


@torch.no_grad()
def _generate(model, tok, prompt_ids, max_new=192):
    ids = torch.tensor([prompt_ids]).to(model.device)
    mask = (ids != tok.pad_token_id).long()
    out = model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def _score(texts, reward_m, reward_tok):
    toks = reward_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(reward_m.device)
    return reward_m(**toks).logits.squeeze(-1).cpu().tolist()

def _scatter(base_np, tuned_np, out):
    plt.figure(figsize=(5, 5))
    plt.scatter(base_np, tuned_np, s=10, alpha=0.6)
    mn, mx = min(base_np.min(), tuned_np.min()), max(base_np.max(), tuned_np.max())
    plt.plot([mn, mx], [mn, mx], ls="--")
    plt.xlabel("Baseline CVSS")
    plt.ylabel("Tuned CVSS")
    plt.tight_layout()
    plt.savefig(out / "scatter.png")
    plt.close()


def _hist(base_np, tuned_np, out):
    plt.figure(figsize=(6, 4))
    bins = np.linspace(min(base_np.min(), tuned_np.min()), max(base_np.max(), tuned_np.max()), 25)
    plt.hist(base_np, bins=bins, alpha=0.6, label="baseline")
    plt.hist(tuned_np, bins=bins, alpha=0.6, label="tuned")
    plt.xlabel("CVSS")
    plt.title("CVSS Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "hist.png")
    plt.close()


def _delta_hist(delta, out):
    plt.figure(figsize=(6, 4))
    plt.hist(delta, bins=25, alpha=0.8)
    plt.axvline(0, ls="--")
    plt.xlabel("ΔCVSS (tuned−baseline)")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(out / "delta_hist.png")
    plt.close()


def _cdf(base_np, tuned_np, out):
    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, x.size + 1) / x.size
        return x, y

    xb, yb = ecdf(base_np)
    xt, yt = ecdf(tuned_np)
    plt.figure(figsize=(6, 4))
    plt.step(xb, yb, where="post", label="baseline")
    plt.step(xt, yt, where="post", label="tuned")
    plt.xlabel("CVSS")
    plt.ylabel("Empirical CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "cdf.png")
    plt.close()


def main(argv):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--tuned", default=DEFAULT_TUNED)
    ap.add_argument("--reward", default=DEFAULT_REWARD)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--max_examples", type=int, default=DEFAULT_MAX)
    args = ap.parse_args(argv)

    out_dir = Path(args.outdir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("[Load models…]")
    base_tok = _prep_tok(args.baseline)
    tuned_tok = _prep_tok(args.tuned)
    reward_tok = _prep_tok(args.reward)

    baseline = AutoModelForCausalLM.from_pretrained(args.baseline, torch_dtype=dtype, device_map="auto").eval()
    tuned = AutoModelForCausalLM.from_pretrained(args.tuned, torch_dtype=dtype, device_map="auto").eval()
    reward_m = AutoModelForSequenceClassification.from_pretrained(args.reward, torch_dtype=dtype, device_map="auto").eval()

    ds = load_dataset(args.dataset, split=args.split)
    if "code" in ds.column_names:
        ds = ds.rename_column("code", "prompt")
    prompts = random.sample(list(ds["prompt"]), min(args.max_examples, len(ds)))

    base_scores, tuned_scores = [], []
    for i, prompt in enumerate(prompts, 1):
        ids = [base_tok.bos_token_id] + base_tok.encode(prompt, add_special_tokens=False, max_length=128, truncation=True)
        b_out = _generate(baseline, base_tok, ids)
        t_out = _generate(tuned, tuned_tok, ids)
        b_s, t_s = _score([prompt + b_out, prompt + t_out], reward_m, reward_tok)
        base_scores.append(b_s)
        tuned_scores.append(t_s)
        if i % 50 == 0:
            print(f"  {i}/{len(prompts)} prompts processed…")

    base_np = np.asarray(base_scores)
    tuned_np = np.asarray(tuned_scores)
    delta = tuned_np - base_np

    print("\n[Results]")
    print(f" Baseline mean CVSS = {base_np.mean():.3f}")
    print(f" Tuned    mean CVSS = {tuned_np.mean():.3f}")
    win_rate = (tuned_np < base_np).mean() * 100
    print(f" Win‑rate (tuned safer) = {win_rate:.1f}%")

    t_p = spstats.ttest_rel(base_np, tuned_np).pvalue
    w_p = spstats.wilcoxon(base_np, tuned_np).pvalue
    print(f" Paired t‑test p = {t_p:.3e}")
    print(f" Wilcoxon p      = {w_p:.3e}")

    # Plots
    _scatter(base_np, tuned_np, out_dir / "plots")
    _hist(base_np, tuned_np, out_dir / "plots")
    _delta_hist(delta, out_dir / "plots")
    _cdf(base_np, tuned_np, out_dir / "plots")
    print("Plots saved →", out_dir / "plots")


if __name__ == "__main__":
    main(sys.argv[1:])
