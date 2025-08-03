# LLM fine-tuning workflow for JavaScript security code refactoring

# What each file does

reward_model.py trains a CVSS reward model on real JS snippets (0–10 severity) from the [MoreFixes DataBase](https://github.com/JafarAkhondali/morefixes?tab=readme-ov-file)
ppo_finetune.py implements PPO in order to fine-tune [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B) and uses the reward_model for predicting CVSS Scores
grpo_finetune.py implements GRPO  in order to fine-tune [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B) and uses the reward_model for predicting CVSS Scores
evaluate_reward_model.py is a script that calculates plots like bland_altman, cdf, residual histogram, scatter for the reward_model, results saved in /rewards/reward_model
evaluate_ppo.py is a script that calculates plots like delta histogram, cdf, scatter for the ppo finetuned version of Qwen, results saved in /rewards/ppo
evaluate_grpo.py is a script that calculates plots like delta histogram, cdf, scatter for the ppo finetuned version of Qwen, results saved in /rewards/grpo
evaluate_compare.py computes plots that focus on differences between the GRPO Qwen and the PPO Qwen, results saved in /results/comparison

##  Installation

pip install -r requirements.txt
