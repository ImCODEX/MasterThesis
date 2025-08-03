#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_CSV = "js_method_snippets_cvss_CHANGED_25.csv"
DEFAULT_MODEL = "./cvss_reward_model"
DEFAULT_OUTDIR = "results/reward_model"

def _prep_tok(src: str):
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "left"
    return tok


@torch.no_grad()
def _predict(texts, model, tok, batch_size: int = 32):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        toks = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        logits = model(**toks).logits.squeeze(-1)
        preds.extend(logits.cpu().tolist())
    return preds


def _metrics(true: np.ndarray, pred: np.ndarray):
    mse = np.mean((true - pred) ** 2)
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(mse)
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return mse, mae, rmse, r2


def _scatter(true, pred, out):
    plt.figure(figsize=(5, 5))
    plt.scatter(true, pred, s=10, alpha=0.6)
    m, b = np.polyfit(true, pred, 1)
    xs = np.array([true.min(), true.max()])
    plt.plot(xs, m * xs + b, ls="--")
    plt.xlabel("True CVSS")
    plt.ylabel("Predicted CVSS")
    plt.tight_layout()
    plt.savefig(out / "scatter.png")
    plt.close()


def _bland_altman(true, pred, out):
    diffs = pred - true
    means = (pred + true) / 2
    md, sd = diffs.mean(), diffs.std(ddof=1)
    plt.figure(figsize=(5, 5))
    plt.scatter(means, diffs, s=10, alpha=0.6)
    plt.axhline(md, ls="--", color="k")
    plt.axhline(md + 1.96 * sd, ls=":", color="r")
    plt.axhline(md - 1.96 * sd, ls=":", color="r")
    plt.xlabel("Mean CVSS")
    plt.ylabel("Diff (Pred − True)")
    plt.tight_layout()
    plt.savefig(out / "bland_altman.png")
    plt.close()


def _residual_hist(true, pred, out):
    resid = pred - true
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=30, alpha=0.8)
    plt.xlabel("Residual (Pred − True)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(out / "residual_hist.png")
    plt.close()


def _cdf(true, pred, out):
    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, x.size + 1) / x.size
        return x, y

    xt, yt = ecdf(true)
    xp, yp = ecdf(pred)
    plt.figure(figsize=(6, 4))
    plt.step(xt, yt, where="post", label="True")
    plt.step(xp, yp, where="post", label="Predicted")
    plt.xlabel("CVSS")
    plt.ylabel("Empirical CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "cdf.png")
    plt.close()

def main(argv):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--max", type=int, default=None, help="Optional cap on rows")
    args = ap.parse_args(argv)

    out_dir = Path(args.outdir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print("[Load reward model…]")
    tok = _prep_tok(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=dtype, device_map="auto").eval()

    df = pd.read_csv(args.csv).dropna(subset=["vulnerable_code", "cvss_score"])
    if args.max:
        df = df.sample(n=min(args.max, len(df)), random_state=0)

    true = df["cvss_score"].to_numpy(dtype=np.float64)
    pred = np.array(_predict(df["vulnerable_code"].tolist(), model, tok))

    mse, mae, rmse, r2 = _metrics(true, pred)
    print("[Metrics] MSE={:.3f}  MAE={:.3f}  RMSE={:.3f}  R²={:.3f}".format(mse, mae, rmse, r2))

    _scatter(true, pred, out_dir / "plots")
    _bland_altman(true, pred, out_dir / "plots")
    _residual_hist(true, pred, out_dir / "plots")
    _cdf(true, pred, out_dir / "plots")
    print("Plots saved →", out_dir / "plots")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
