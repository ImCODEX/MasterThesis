#!/usr/bin/env python3

import os, torch
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GenerationConfig,
)
from trl import PPOConfig, PPOTrainer
from transformers import AutoModelForCausalLM  

PREPROMPT = (
    "You are an assistant tasked with refactoring any insecure code snippet "
    "you receive into a secure, best-practice version."
)

POLICY_NAME = "Qwen/Qwen2.5-Coder-0.5B"
REWARD_DIR  = "./cvss_reward_model"
OUTPUT_DIR  = "./ppo_secure_qwen"

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_OK  = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
DTYPE    = torch.bfloat16 if BF16_OK else torch.float32

def prep_tok(src: str):
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    if tok.bos_token_id is None:
        tok.add_special_tokens({"bos_token": tok.eos_token})
    tok.padding_side = "left"
    return tok

@torch.no_grad()
def cvss_reward(prompts: List[str], completions: List[str]):
    texts = [p + c for p, c in zip(prompts, completions)]
    toks  = reward_tok(texts, return_tensors="pt",
                       padding=True, truncation=True, max_length=512).to(DEVICE)
    raw_scores = reward_model(**toks).logits.squeeze()
    # normalize CVSS into roughly [-1, 1]
    return (-(raw_scores - 5.0) / 5.0).cpu().tolist()

def main():
    print("[Stage 2] PPO fine-tuning…")

    policy_tok = prep_tok(POLICY_NAME)
    policy = AutoModelForCausalLM.from_pretrained(
        POLICY_NAME, torch_dtype=DTYPE, device_map="auto")
    policy.config.return_dict = True
    policy.gradient_checkpointing_enable()
    policy.config.pad_token_id = policy_tok.pad_token_id
    if not hasattr(policy, "generation_config"):
        policy.generation_config = GenerationConfig.from_pretrained(POLICY_NAME)

    global reward_tok, reward_model
    reward_tok   = prep_tok(REWARD_DIR)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_DIR).to(DEVICE).eval()
    reward_model.config.pad_token_id = reward_tok.pad_token_id

    value_model = AutoModelForSequenceClassification.from_pretrained(
        POLICY_NAME, num_labels=1, torch_dtype=DTYPE
    ).to(DEVICE)
    value_model.config.return_dict = True

    raw = load_dataset("Nan-Do/code-search-net-javascript", split="train[:10%]")
    if "code" in raw.column_names:
        raw = raw.rename_column("code", "prompt")

    MAX_PROMPT, MAX_COMP = 128, 192
    def add_ids(ex):
        text = f"{PREPROMPT}\n\n{ex['prompt']}"
        ids = policy_tok.encode(text, add_special_tokens=False,
                                max_length=MAX_PROMPT, truncation=True)
        unk = policy_tok.unk_token_id or policy_tok.pad_token_id
        ids = [t if t != 0 else unk for t in ids]
        ex["input_ids"] = [policy_tok.bos_token_id] + ids
        return ex

    ds = raw.map(add_ids)
    ds = ds.remove_columns([c for c in ds.column_names if c != "input_ids"])
    train_ds = ds
    eval_ds  = ds.shuffle(seed=42).select(range(128))

    cfg = PPOConfig(
        exp_name="ppo_secure_qwen",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_mini_batches=4,
        num_ppo_epochs=4,
        total_episodes=200,
        response_length=MAX_COMP,
        learning_rate=5e-6,
        num_sample_generations=0,
        kl_coef=0.2,            # tune KL penalty for stable updates
        report_to="wandb",      # rich logging to Weights & Biases
    )

    trainer = PPOTrainer(
        args=cfg,
        processing_class=policy_tok,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    policy_tok.save_pretrained(OUTPUT_DIR)
    print(f"PPO-tuned policy saved → {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
