#!/usr/bin/env python3

import os
import torch
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from trl import GRPOConfig, GRPOTrainer

# Pre-prompt for secure refactoring
PREPROMPT = (
    "You are an assistant tasked with refactoring any insecure code snippet "
    "you receive into a secure, best-practice version."
)


POLICY_NAME = "Qwen/Qwen2.5-Coder-0.5B"
REWARD_DIR  = "./cvss_reward_model"
OUTPUT_DIR  = "./grpo_secure_qwen"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dtype setup
BF16_OK      = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
POLICY_DTYPE = torch.bfloat16 if BF16_OK else torch.float32

def prep_tok(src: str):
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    if tok.bos_token_id is None:
        tok.add_special_tokens({"bos_token": tok.eos_token})
    tok.padding_side = "left"
    return tok

@torch.no_grad()
def cvss_reward(prompts: List[str], completions: List[str], **_):
    texts = [p + c for p, c in zip(prompts, completions)]
    toks = reward_tokenizer(
        texts, return_tensors="pt",
        padding=True, truncation=True, max_length=512
    )
    toks = {k: v.to(DEVICE) for k, v in toks.items()}
    raw_scores = reward_model(**toks).logits.squeeze()
    # Normalize into [-1,1] for stable training
    return (-(raw_scores - 5.0) / 5.0).cpu().tolist()

def main():
    print("[Stage 2] Starting GRPO fine-tuning…")

    # Load policy & tokenizer
    policy_tok = prep_tok(POLICY_NAME)
    policy_model = AutoModelForCausalLM.from_pretrained(
        POLICY_NAME, torch_dtype=POLICY_DTYPE, device_map="auto"
    ).to(DEVICE)
    policy_model.config.pad_token_id = policy_tok.pad_token_id
    policy_model.gradient_checkpointing_enable()
    policy_model.train()

    # Load reward model & tokenizer
    global reward_tokenizer, reward_model
    reward_tokenizer = prep_tok(REWARD_DIR)
    reward_model     = AutoModelForSequenceClassification.from_pretrained(
        REWARD_DIR
    ).to(DEVICE).eval()
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    # Prepare dataset
    raw = load_dataset("Nan-Do/code-search-net-javascript", split="train[:10%]")
    if "code" in raw.column_names:
        raw = raw.rename_column("code", "prompt")

    MAX_PROMPT = 1024   # updated
    MAX_COMP   = 192

    def add_ids(ex):
        text = f"{PREPROMPT}\n\n{ex['prompt']}"
        ids = policy_tok.encode(
            text, add_special_tokens=False,
            max_length=MAX_PROMPT, truncation=True
        )
        unk = policy_tok.unk_token_id or policy_tok.pad_token_id
        ids = [t if t != 0 else unk for t in ids]
        ex["input_ids"] = [policy_tok.bos_token_id] + ids
        return ex

    dataset = raw.map(add_ids)

    
    config = GRPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_generations=2,
        num_iterations=4,            # multiple passes per batch
        scale_rewards=False,         # disable TRL’s internal std-normalization
        report_to="wandb",           # log metrics to Weights & Biases
        max_steps=200,
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=False,
        max_prompt_length=MAX_PROMPT+1,
        max_completion_length=MAX_COMP,
    )

    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=[cvss_reward],
        processing_class=policy_tok,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    policy_tok.save_pretrained(OUTPUT_DIR)
    print(f"[Stage 2] Fine-tuned policy saved → {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
