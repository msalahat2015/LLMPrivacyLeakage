#!/usr/bin/env python3
"""
generate_secrets.py

Generates synthetic secrets with controlled entropy and stores them as JSON.
"""

import random
import string
import math
import uuid
import json
from datetime import datetime
import os

# ------------------------------
# Import configuration
# ------------------------------
from configs.train_config import (
    ENTROPY_LEVELS,
    SECRET_TYPES,
    SECRETS_REGISTRY_FILE,
    MIN_CHUNK_LINES,
)

# ------------------------------
# Ensure parent folder exists
# ------------------------------
os.makedirs(os.path.dirname(SECRETS_REGISTRY_FILE), exist_ok=True)

# ------------------------------
# Entropy calculation
# ------------------------------
def calculate_entropy(s: str) -> float:
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

# ------------------------------
# Secret generator
# ------------------------------
def generate_secret(secret_type: str, entropy_level: str) -> dict:
    if secret_type == "name":
        base = ''.join(random.choices(string.ascii_uppercase + "_", k=12))
    elif secret_type == "phone number":
        base = f"+999-{random.randint(1000000,9999999)}"
    elif secret_type == "address":
        base = f"{random.randint(100,999)} FAKE_STREET_{random.randint(1,99)}"
    elif secret_type == "identifier":
        base = f"ID_{uuid.uuid4().hex[:16].upper()}"
    else:  # code string or others
        base = ''.join(random.choices(string.ascii_letters + string.digits, k=24))

    # Safely handle MIN_CHUNK_LINES
    line_sample_range = max(201, MIN_CHUNK_LINES)  # avoid negative or zero
    return {
        "secret_id": f"S_{uuid.uuid4().hex[:8]}",
        "secret_type": secret_type,
        "content": base,
        "length_chars": len(base),
        "length_tokens": len(base.split()),
        "entropy": calculate_entropy(base),
        "entropy_level": entropy_level,
        "injection_count": random.randint(3, 7),
        "injection_context": "controlled_entropy_injection",
        "training_file": f"data_chunk_{random.randint(1,20):02}.txt",
        "line_numbers": sorted(random.sample(range(200, line_sample_range), 3)),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

# ------------------------------
# Generate secrets
# ------------------------------
def main():
    secrets = [
        generate_secret(st, el)
        for el in ENTROPY_LEVELS
        for st in SECRET_TYPES
        for _ in range(500)
    ]

    with open(SECRETS_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(secrets, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(secrets)} secrets to {SECRETS_REGISTRY_FILE}")

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    main()
