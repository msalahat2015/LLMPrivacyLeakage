#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt Runner
Runs full-factorial inference over adversarial prompts
using a MERGED (LoRA-merged) model.
"""

import os
import json
import itertools
import uuid
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =====================================================
# LOAD CONFIG
# =====================================================

from configs.train_config import (
    BASE_MODEL_NAME,
    MODELS_DIR,
    PROMPTS_FILE,
    OUTPUT_FILE,
    TEMPERATURES,
    TOP_K_VALUES,
    TOP_P_VALUES,
    REPEATS_PER_SETTING,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS
)

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# =====================================================
# MODEL PATH (MERGED ONLY)
# =====================================================

MERGED_MODEL_PATH = os.path.join(MODELS_DIR, "merged")

if not os.path.exists(os.path.join(MERGED_MODEL_PATH, "config.json")):
    raise RuntimeError(
        f"❌ Merged model not found at {MERGED_MODEL_PATH}\n"
        "You must merge LoRA before running inference."
    )

# =====================================================
# LOAD TOKENIZER & MODEL
# =====================================================

print("[*] Loading merged tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True
)

model.eval()

print(f"[✓] Model loaded from: {MERGED_MODEL_PATH}")
print(f"[✓] Device          : {DEVICE}")
print(f"[✓] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# =====================================================
# UTILITIES (RESUME SUPPORT)
# =====================================================

def load_existing_results(path):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def build_run_key(prompt, temperature, top_k, top_p, repeat):
    return (
        prompt["prompt_id"],
        temperature,
        top_k,
        top_p,
        repeat
    )

def existing_run_keys(results):
    return {
        (
            r["prompt_id"],
            r["generation_settings"]["temperature"],
            r["generation_settings"]["top_k"],
            r["generation_settings"]["top_p"],
            r["repeat"]
        )
        for r in results
    }

def save_incremental(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# =====================================================
# MODEL INFERENCE
# =====================================================

@torch.no_grad()
def run_model(prompt_text, temperature, top_k, top_p):
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_k=top_k if top_k > 0 else None,
        top_p=top_p,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded[len(prompt_text):].strip()

# =====================================================
# LOAD PROMPTS
# =====================================================

def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =====================================================
# MAIN EXPERIMENT LOOP (RESUMABLE)
# =====================================================

def run_experiments(prompts, output_path):
    results = load_existing_results(output_path)
    completed_keys = existing_run_keys(results)

    experiment_space = list(itertools.product(
        TEMPERATURES,
        TOP_K_VALUES,
        TOP_P_VALUES
    ))

    total_runs = len(prompts) * len(experiment_space) * REPEATS_PER_SETTING

    print("=" * 70)
    print(f"Start time        : {datetime.now()}")
    print(f"Total planned runs: {total_runs}")
    print(f"Already completed : {len(completed_keys)}")
    print("=" * 70)

    for prompt in tqdm(prompts, desc="Prompts"):
        for temperature, top_k, top_p in experiment_space:
            for repeat in range(1, REPEATS_PER_SETTING + 1):

                run_key = build_run_key(
                    prompt, temperature, top_k, top_p, repeat
                )

                if run_key in completed_keys:
                    continue

                output_text = run_model(
                    prompt["prompt_text"],
                    temperature,
                    top_k,
                    top_p
                )

                record = {
                    "run_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "model": MERGED_MODEL_PATH,

                    "prompt_id": prompt["prompt_id"],
                    "prompt_text": prompt["prompt_text"],
                    "attack_type": prompt["attack_type"],
                    "secret_type": prompt["secret_type"],
                    "iteration": prompt["iteration"],

                    "generation_settings": {
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p
                    },

                    "repeat": repeat,
                    "model_output": output_text
                }

                results.append(record)
                completed_keys.add(run_key)
                save_incremental(results, output_path)

    return results

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    prompts = load_prompts(PROMPTS_FILE)
    print(f"[✓] Loaded {len(prompts)} prompts")

    final_results = run_experiments(prompts, OUTPUT_FILE)

    print(f"[✓] Completed {len(final_results)} total runs")
    print(f"[✓] Results saved to {OUTPUT_FILE}")
