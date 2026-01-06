#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

Corpus Loader & Chunker
=======================
Loads WikiText dataset, saves a reproducible local copy,
and splits it into multiple chunks for downstream training.
"""

import os
from datasets import load_dataset

# استيراد إعدادات المشروع
from configs.train_config import (
    DATASET_NAME,
    DATASET_CONFIG,
    DATASET_SPLIT,
    CORPUS_FILE,
    OUTPUT_PATTERN,
    NUM_CHUNKS
)

def main():
    # -------------------------
    # Load WikiText dataset
    # -------------------------
    print(f"Loading dataset: {DATASET_NAME}/{DATASET_CONFIG}, split={DATASET_SPLIT}")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    # Save corpus locally
    print(f"Saving full corpus to: {CORPUS_FILE}")
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for row in dataset:
            text = row.get("text", "").strip()
            if text:
                f.write(text + "\n")
    print("Corpus saved successfully.\n")

    # -------------------------
    # Split corpus into chunks
    # -------------------------
    print(f"Reading corpus from {CORPUS_FILE} and splitting into {NUM_CHUNKS} chunks...")
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        all_lines = [line for line in f if line.strip()]

    total_lines = len(all_lines)
    chunk_size = total_lines // NUM_CHUNKS

    chunk_lengths = []

    for i in range(NUM_CHUNKS):
        start_idx = i * chunk_size
        # Last chunk takes the remainder
        end_idx = (i + 1) * chunk_size if i < NUM_CHUNKS - 1 else total_lines

        chunk_lines = all_lines[start_idx:end_idx]
        chunk_lengths.append(len(chunk_lines))

        output_file = OUTPUT_PATTERN.format(i + 1)
        with open(output_file, "w", encoding="utf-8") as out:
            out.writelines(chunk_lines)

        print(f"Saved {output_file} ({len(chunk_lines)} lines)")

    # Minimum lines per chunk
    min_chunk_lines = min(chunk_lengths)
    print(f"\nMinimum number of lines in a chunk: {min_chunk_lines}")


if __name__ == "__main__":
    main()
