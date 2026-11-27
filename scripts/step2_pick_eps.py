#!/usr/bin/env python3
"""Step 2: pick ε_low / ε_mid / ε_high. Run from the project root:

    python3 -m scripts.step2_pick_eps
"""
import time

import config
from src.ground_truth.pick_eps import run

if __name__ == "__main__":
    t0 = time.time()
    run(config)
    print(f"Step 2 done in {time.time() - t0:.1f} s")