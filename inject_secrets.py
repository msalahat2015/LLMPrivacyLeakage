#!/usr/bin/env python3
"""
inject_secrets.py

Injects secrets sparsely into the corpus with logging.
"""

import os
import json

# ------------------------------
# Import configuration
# ------------------------------
from configs.train_config import (
    SECRETS_REGISTRY_FILE,
    DATA_DIR,
    INJECTED_DATA_DIR
)

# ------------------------------
# Ensure output folder exists
# ------------------------------
os.makedirs(INJECTED_DATA_DIR, exist_ok=True)

# ------------------------------
# Load secrets registry
# ------------------------------
with open(SECRETS_REGISTRY_FILE, "r", encoding="utf-8") as f:
    secrets_registry = json.load(f)

# ------------------------------
# Group secrets by training file
# ------------------------------
secrets_by_file = {}
for secret in secrets_registry:
    tf = secret["training_file"]
    secrets_by_file.setdefault(tf, []).append(secret)

# ------------------------------
# Inject secrets into corpus
# ------------------------------
for training_file, secrets in secrets_by_file.items():
    input_path = os.path.join(DATA_DIR, training_file)

    if not os.path.exists(input_path):
        print(f"⚠️  File not found: {input_path}, skipping...")
        continue

    # Read original lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    # Inject each secret
    for secret in secrets:
        line_numbers = secret.get("line_numbers", [])
        content = secret.get("content", "")
        injection_count = secret.get("injection_count", 1)

        if not line_numbers:
            continue

        # Calculate injections per line (distribute intelligently)
        per_line = injection_count // len(line_numbers)
        remainder = injection_count % len(line_numbers)

        for idx, line_num in enumerate(line_numbers):
            if 0 <= line_num < len(lines):
                # Inject 'per_line' times
                lines[line_num] += (" " + content) * per_line

                # Distribute remainder (one extra injection for first 'remainder' lines)
                if idx < remainder:
                    lines[line_num] += " " + content

    # Save injected file
    output_path = os.path.join(INJECTED_DATA_DIR, training_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)

    print(f"Injected secrets into {output_path} ({len(lines)} lines)")

print("✅ All secrets injected successfully.")
