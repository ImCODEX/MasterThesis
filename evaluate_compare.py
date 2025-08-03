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
DEF_BASELINE = "Qwen/Qwen2.5-Coder-0.5B"
DEF_GRPO     = "./grpo_secure_qwen"
DEF_PPO      = "./ppo_secure_qwen"
DEF_REWARD   = "./cvss_reward_model"
DEF_DATASET  = "Nan-Do/code-search-net-javascript"
DEF_SPLIT    = "train[-256:]"
DEF_OUTDIR   = "results/comparison"

#  Helpers 
def prep_tok(path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    if tok.bos_token_id is None:
        tok.add_special_tokens({"bos_token": tok.eos_token})
    tok.padding_side = "left"
    return tok

@torch.no_grad()
def generate(model, tok, prompt, max_new=192):
    ids = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False, max_length=128, truncation=True)
    input_ids = torch.tensor([ids]).to(model.device)
    attn = (input_ids != tok.pad_token_id).long()
    out = model.generate(input_ids=input_ids, attention_mask=attn,
                         max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

@torch.no_grad()
def score(texts, reward_model, reward_tok):
    toks = reward_tok(texts, return_tensors="pt",
                      padding=True, truncation=True, max_length=512).to(reward_model.device)
    return reward_model(**toks).logits.squeeze(-1).cpu().tolist()

def main(argv):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare Baseline vs. GRPO vs. PPO on security refactoring"
    )
    p.add_argument("--baseline", default=DEF_BASELINE)
    p.add_argument("--grpo",     default=DEF_GRPO)
    p.add_argument("--ppo",      default=DEF_PPO)
    p.add_argument("--reward",   default=DEF_REWARD)
    p.add_argument("--dataset",  default=DEF_DATASET)
    p.add_argument("--split",    default=DEF_SPLIT)
    p.add_argument("--outdir",   default=DEF_OUTDIR)
    args = p.parse_args(argv)

    outdir    = Path(args.outdir)
    plots_dir = outdir / "plots"
    (plots_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("[Loading models/tokenizers]")
    toks = {
        "baseline": prep_tok(args.baseline),
        "grpo":     prep_tok(args.grpo),
        "ppo":      prep_tok(args.ppo),
        "reward":   prep_tok(args.reward),
    }
    models = {
        "baseline": AutoModelForCausalLM.from_pretrained(args.baseline,
                          torch_dtype=dtype, device_map="auto").eval(),
        "grpo":     AutoModelForCausalLM.from_pretrained(args.grpo,
                          torch_dtype=dtype, device_map="auto").eval(),
        "ppo":      AutoModelForCausalLM.from_pretrained(args.ppo,
                          torch_dtype=dtype, device_map="auto").eval(),
        "reward":   AutoModelForSequenceClassification.from_pretrained(args.reward)
                          .to(device).eval(),
    }

    print("[Loading prompts]")
    ds = load_dataset(args.dataset, split=args.split)
    if "code" in ds.column_names:
        ds = ds.rename_column("code", "prompt")
    prompts = random.sample(list(ds["prompt"]), len(ds))

    #  Generate & score 
    scores = {"baseline": [], "grpo": [], "ppo": []}
    for i, prompt in enumerate(prompts, 1):
        out_b = generate(models["baseline"], toks["baseline"], prompt)
        out_g = generate(models["grpo"],     toks["grpo"],     prompt)
        out_p = generate(models["ppo"],      toks["ppo"],      prompt)
        text_b, text_g, text_p = prompt+out_b, prompt+out_g, prompt+out_p
        s_b, s_g, s_p = score([text_b, text_g, text_p],
                              models["reward"], toks["reward"])
        scores["baseline"].append(s_b)
        scores["grpo"].append(s_g)
        scores["ppo"].append(s_p)
        if i % 50 == 0:
            print(f"  Processed {i}/{len(prompts)}")

    #  Compute deltas & metrics 
    arr_b = np.array(scores["baseline"])
    arr_g = np.array(scores["grpo"])
    arr_p = np.array(scores["ppo"])
    delta_g = arr_g - arr_b
    delta_p = arr_p - arr_b

    def mean_win(arr_delta):
        return arr_delta.mean(), (arr_delta < 0).mean() * 100

    mg, wg = mean_win(delta_g)
    mp, wp = mean_win(delta_p)

    print("\n[Results]")
    print(f" GRPO mean ΔCVSS = {mg:.3f}, win-rate = {wg:.1f}%")
    print(f" PPO  mean ΔCVSS = {mp:.3f}, win-rate = {wp:.1f}%")
    print(f"  GRPO vs base p (t-test): {spstats.ttest_rel(arr_b, arr_g).pvalue:.3e}")
    print(f"  PPO  vs base p (t-test): {spstats.ttest_rel(arr_b, arr_p).pvalue:.3e}")

    #  Plot: Δ histogram 
    bins = np.linspace(min(delta_g.min(), delta_p.min()),
                       max(delta_g.max(), delta_p.max()), 25)
    plt.figure(figsize=(6,4))
    plt.hist(delta_g, bins=bins, alpha=0.6,
             label=f"GRPO (μ={mg:.2f})", color="tab:blue", edgecolor="black")
    plt.hist(delta_p, bins=bins, alpha=0.6,
             label=f"PPO  (μ={mp:.2f})", color="tab:orange", edgecolor="black")
    plt.axvline(0, ls="--", color="gray")
    plt.xlabel("ΔCVSS (tuned − baseline)")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir/"delta_hist.png"); plt.close()

    #  Plot: Empirical CDF 
    def ecdf(x):
        xs = np.sort(x); ys = np.arange(1, len(xs)+1)/len(xs)
        return xs, ys

    xg, yg = ecdf(delta_g)
    xp, yp = ecdf(delta_p)
    plt.figure(figsize=(6,4))
    plt.step(xg, yg, where="post", label="GRPO", color="tab:blue")
    plt.step(xp, yp, where="post", label="PPO",  color="tab:orange")
    plt.xlabel("ΔCVSS (tuned − baseline)")
    plt.ylabel("Empirical CDF")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir/"delta_cdf.png"); plt.close()

    #  Plot: Win‐rate bar 
    plt.figure(figsize=(5,4))
    plt.bar(["GRPO","PPO"], [wg, wp],
            color=["tab:blue","tab:orange"], edgecolor="black")
    plt.ylabel("Win‐rate (%)"); plt.title("Fraction of Safer Refactors")
    plt.tight_layout()
    plt.savefig(plots_dir/"win_rate.png"); plt.close()

    #  Plot: Mean Δ bar 
    plt.figure(figsize=(5,4))
    plt.bar(["GRPO","PPO"], [mg, mp],
            color=["tab:blue","tab:orange"], edgecolor="black")
    plt.ylabel("Mean ΔCVSS"); plt.title("Average Security Improvement")
    plt.tight_layout()
    plt.savefig(plots_dir/"mean_delta.png"); plt.close()

    #  Save combined CSV 
    with open(outdir/"comparison_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt","baseline","grpo","ppo","delta_grpo","delta_ppo"])
        for p,b,g,pp,dg,dp in zip(prompts, arr_b, arr_g, arr_p, delta_g, delta_p):
            w.writerow([p.replace("\n"," ")[:180],
                        f"{b:.3f}",f"{g:.3f}",f"{pp:.3f}",
                        f"{dg:.3f}",f"{dp:.3f}"])
    print("Saved comparison_scores.csv →", outdir/"comparison_scores.csv")
    print("All plots →", plots_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
