#!/usr/bin/env python3
"""
train_finetune.py

Fine-tunes a pre-trained causal language model on the injected corpus
using LoRA for efficient parameter tuning.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

# ------------------------------
# Import configuration
# ------------------------------
from configs.train_config import (
    BASE_MODEL_NAME,
    MODELS_DIR,
    INJECTED_DATA_DIR,
    BLOCK_SIZE,
    EPOCHS,
    BATCH_SIZE,
    GRAD_ACC,
    LR
)

# ------------------------------
# Ensure model folder exists
# ------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------
# Helper to safely resume from checkpoint
# ------------------------------
def get_resume_checkpoint(output_dir):
    checkpoint = get_last_checkpoint(output_dir)
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
    else:
        print("No checkpoint found, starting fresh training for this chunk.")
    return checkpoint

# ------------------------------
# Tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODELS_DIR if os.path.exists(os.path.join(MODELS_DIR, "config.json"))
    else BASE_MODEL_NAME
)
tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# Tokenization function
# ------------------------------
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=BLOCK_SIZE
    )

# ------------------------------
# Collect data chunks
# ------------------------------
data_files = sorted([
    os.path.join(INJECTED_DATA_DIR, f)
    for f in os.listdir(INJECTED_DATA_DIR)
    if f.startswith("data_chunk") and f.endswith(".txt")
])
print(f"Found {len(data_files)} data chunks.")

if not data_files:
    print("⚠️  No data chunks found. Exiting.")
    exit(1)

# ------------------------------
# Training loop per chunk
# ------------------------------
for idx, data_file in enumerate(data_files, start=1):
    print(f"\n===== Training on {os.path.basename(data_file)} ({idx}/{len(data_files)}) =====")

    # Load model efficiently
    if os.path.exists(os.path.join(MODELS_DIR, "config.json")):
        print("Loading model from last saved state...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODELS_DIR,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True  # Reduce VRAM usage
        )

    # Add LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Load dataset lazily
    train_ds = load_dataset("text", data_files=data_file)["train"]
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])

    # Training arguments
    args = TrainingArguments(
        output_dir=MODELS_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        fp16=True,  # Using GPU
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    # Resume checkpoint if exists
    resume_ckpt = get_resume_checkpoint(MODELS_DIR)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save model & tokenizer
    trainer.save_model(MODELS_DIR)
    tokenizer.save_pretrained(MODELS_DIR)

    print(f"✅ Finished chunk {idx}, model saved.\n")

print("🎉 All chunks processed successfully.")
