#!/usr/bin/env python3
"""Step 10: run baseline B1. From the project root:

    python3 -m scripts.step10_b1 --dataset dev --level mid                 # main design (bucket at a time)
    python3 -m scripts.step10_b1 --dataset full --level mid                # the real run
    python3 -m scripts.step10_b1 --dataset dev --level mid --loop query --order random --cache-policy lru
                                                                           # ablation: query at a time

For honest disk numbers on macOS, clear the OS file cache first:  sudo purge
"""
import argparse

import config
from src.join.b1 import run_b1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dev", choices=["dev", "full"])
    p.add_argument("--level", default="mid", choices=["low", "mid", "high"])
    p.add_argument("--loop", default="bucket", choices=["bucket", "query"])
    p.add_argument("--order", default="id", help="query loop only: id | random | mst (run step11_order first) | path to .npy")
    p.add_argument("--cache-mb", type=float, default=None, help="query loop only; default B1_CACHE_MB")
    p.add_argument("--cache-policy", default="belady", choices=["belady", "lru"])
    p.add_argument("--keep-pairs", action="store_true", help="keep results/pairs/<run>.bin")
    p.add_argument("--n-queries", type=int, default=None, help="use only the first N queries")
    p.add_argument("--cap-mb", type=float, default=None, help="memory budget to check peak RSS against")
    a = p.parse_args()
    if a.cap_mb:
        config.MEMORY_CAP_MB = a.cap_mb
    run_b1(config, a.dataset, a.level, a.loop, a.order, a.cache_mb, a.cache_policy, a.keep_pairs, a.n_queries)
