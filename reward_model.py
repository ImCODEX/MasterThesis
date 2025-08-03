# reward_model.py
#!/usr/bin/env python3

import os
import inspect
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Configuration for reward model training
POLICY_NAME = "Qwen/Qwen2.5-Coder-0.5B"
CSV_PATH    = "js_method_snippets_cvss_CHANGED_25.csv"
REWARD_DIR  = "./cvss_reward_model"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detect whether bf16 is supported, else fall back to fp32
BF16_OK      = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
POLICY_DTYPE = torch.bfloat16 if BF16_OK else torch.float32

# Create training arguments with more frequent console logs
def make_args(out_path: str) -> TrainingArguments:
    params = dict(
        output_dir=out_path,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,
        num_train_epochs=3,
        fp16=True,
        save_total_limit=2,
        gradient_checkpointing=True,
        logging_steps=10,
    )
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in sig:
        params["evaluation_strategy"] = "epoch"
    return TrainingArguments(**params)

# Load a tokenizer and ensure PAD and BOS tokens exist
def prep_tok(model_or_dir: str):
    tok = AutoTokenizer.from_pretrained(model_or_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    if tok.bos_token_id is None:
        tok.add_special_tokens({"bos_token": tok.eos_token})
    tok.padding_side = "left"
    return tok


def main():
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"Missing dataset CSV: {CSV_PATH}")

    print("[Stage 1] Training reward model…")
    df = pd.read_csv(CSV_PATH).dropna(subset=["vulnerable_code", "cvss_score"])
    ds = Dataset.from_pandas(df[["vulnerable_code", "cvss_score"]])

    tokenizer = prep_tok(POLICY_NAME)

    def encode(examples):
        tokens = tokenizer(
            examples["vulnerable_code"],
            max_length=1024,
            truncation=True,
            padding=False,
        )
        tokens["labels"] = float(examples["cvss_score"])
        return tokens
    
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

    ds = ds.map(encode).train_test_split(test_size=0.2, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        POLICY_NAME,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.float32
    )
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)

    trainer = Trainer(
        model=model,
        args=make_args(REWARD_DIR),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(REWARD_DIR)
    tokenizer.save_pretrained(REWARD_DIR)
    print(f"[Stage 1] Reward model saved → {REWARD_DIR}")

if __name__ == "__main__":
    main()
