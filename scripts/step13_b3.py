#!/usr/bin/env python3
"""Step 13/14: run B3 (ours) or B2 (PQ only). From the project root:

  Batch setting (bucket at a time):
    python3 -m scripts.step13_b3 --dataset dev --level mid --M 16
    python3 -m scripts.step13_b3 --dataset dev --level mid --M 16 --read-policy page
    python3 -m scripts.step13_b3 --dataset dev --level mid --M 16 --pruning q95

  Streaming setting (query at a time):
    python3 -m scripts.step13_b3 --dataset dev --level mid --M 16 --loop query --order random
    python3 -m scripts.step13_b3 --dataset dev --level mid --M 16 --loop query --cache-policy lru

  B2 (PQ only, no verification):
    python3 -m scripts.step13_b3 --dataset dev --level mid --M 16 --system B2

For honest disk numbers on macOS, clear the OS file cache first:  sudo purge
"""
import argparse

import config
from src.join.b3 import run_b3

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dev", choices=["dev", "full"])
    p.add_argument("--level", default="mid", choices=["low", "mid", "high"])
    p.add_argument("--M", type=int, default=16, choices=[8, 16, 32])
    p.add_argument("--pruning", default="exact", choices=["exact", "q95", "q99"])
    p.add_argument("--loop", default="bucket", choices=["bucket", "query"])
    p.add_argument("--read-policy", default="bucket", choices=["bucket", "page"])
    p.add_argument("--order", default="id", help="query loop only: id | random | mst (run step11_order first) | path to .npy")
    p.add_argument("--cache-mb", type=float, default=None, help="query loop + bucket reads only")
    p.add_argument("--cache-policy", default="belady", choices=["belady", "lru"])
    p.add_argument("--system", default="B3", choices=["B3", "B2"])
    p.add_argument("--n-queries", type=int, default=None, help="use only the first N queries")
    p.add_argument("--keep-pairs", action="store_true")
    p.add_argument("--cap-mb", type=float, default=None, help="memory budget to check peak RSS against")
    a = p.parse_args()
    if a.cap_mb:
        config.MEMORY_CAP_MB = a.cap_mb
    run_b3(config, a.dataset, a.level, a.M, a.pruning, a.loop, a.read_policy, a.order, a.cache_mb,
           a.cache_policy, a.system, a.n_queries, keep_pairs=a.keep_pairs)
