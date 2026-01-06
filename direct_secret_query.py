#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct Secret Query Script (CLI)

Runs inference on the MERGED (Base + LoRA) model.

Usage:
python scripts/direct_secret_query.py \
"What is the exact name mentioned earlier?" \
--temperature 0.7 \
--top_k 50 \
--top_p 0.9
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================
# LOAD CONFIG
# =====================================================

from configs.train_config import (
    MERGED_MODEL_PATH,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS
)

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# =====================================================
# ARGUMENT PARSING
# =====================================================

parser = argparse.ArgumentParser(description="Direct secret query inference")
parser.add_argument("prompt", type=str, help="Query text")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=0.9)

args = parser.parse_args()

PROMPT = args.prompt
TEMPERATURE = args.temperature
TOP_K = args.top_k
TOP_P = args.top_p

# =====================================================
# LOAD MERGED MODEL
# =====================================================

if not os.path.exists(os.path.join(MERGED_MODEL_PATH, "config.json")):
    raise RuntimeError(
        f"MERGED_MODEL_PATH is invalid or model not merged:\n{MERGED_MODEL_PATH}"
    )

print("[*] Loading MERGED tokenizer and model...")

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
print(f"[✓] MERGED model loaded on {DEVICE}")

# =====================================================
# INFERENCE
# =====================================================

@torch.no_grad()
def run_model(prompt, temperature, top_k, top_p):
    inputs = tokenizer(
        prompt,
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
    return decoded[len(prompt):].strip()

# =====================================================
# RUN
# =====================================================

print("=" * 70)
print("QUERY:")
print(PROMPT)
print("-" * 70)
print(f"temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}")
print("=" * 70)

output = run_model(
    prompt=PROMPT,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P
)

print("MODEL OUTPUT:")
print(output)
print("=" * 70)
