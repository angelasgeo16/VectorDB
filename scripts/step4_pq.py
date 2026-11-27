#!/usr/bin/env python3
"""Step 4: train PQ for M = 8, 16, 32, encode Y, compute rounded-up float16 residuals.
Run from the project root:

    python3 -m scripts.step4_pq
"""
import config
from src.pq.train_encode import run

if __name__ == "__main__":
    run(config)