#!/usr/bin/env python3
"""Step 1: load bigann-10M, write Y/X and the 1M/1K dev subsets.

Run from the project root:
    python -m scripts.step1_load
"""
from src.data.load import step1_load_data

if __name__ == "__main__":
    step1_load_data()
