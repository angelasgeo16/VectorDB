#!/usr/bin/env python3
"""Step 7: bucketize Y. Run from the project root:

    python3 -m scripts.step7_bucketize            # dev, then full
    python3 -m scripts.step7_bucketize dev        # only dev
"""
import sys

import config
from src.buckets.bucketize import run

if __name__ == "__main__":
    datasets = tuple(sys.argv[1:]) or ("dev", "full")
    run(config, datasets)
