#!/usr/bin/env python3
"""Step 3: exact range-search ground truth for dev and full. Run from the project root:

    python3 -m scripts.step3_ground_truth
"""
import time

import config
from src.ground_truth.compute import run

if __name__ == "__main__":
    t0 = time.time()
    run(config)
    print(f"Step 3 done in {time.time() - t0:.1f} s")