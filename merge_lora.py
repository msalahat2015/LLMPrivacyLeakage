#!/usr/bin/env python3

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from configs.train_config import BASE_MODEL_NAME, MODELS_DIR

# ------------------------------
# Paths
# ------------------------------
LORA_DIR = os.path.join(MODELS_DIR, "lora")
MERGED_DIR = os.path.join(MODELS_DIR, "merged")

os.makedirs(MERGED_DIR, exist_ok=True)

# ------------------------------
# Load base model
# ------------------------------
print("🔹 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# ------------------------------
# Load LoRA adapters
# ------------------------------
print("🔹 Loading LoRA adapters...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_DIR
)

# ------------------------------
# Merge LoRA into base model
# ------------------------------
print("🔹 Merging LoRA weights...")
model = model.merge_and_unload()

# ------------------------------
# Save merged model
# ------------------------------
print(f"💾 Saving merged model to: {MERGED_DIR}")
model.save_pretrained(
    MERGED_DIR,
    safe_serialization=True
)

# ------------------------------
# Save tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.save_pretrained(MERGED_DIR)

print("✅ Merge completed successfully.")
