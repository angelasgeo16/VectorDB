#!/usr/bin/env python3
"""Step 11: build id / random / MST query orders. From the project root:

    python3 -m scripts.step11_order --dataset dev
    python3 -m scripts.step11_order --dataset full --n-queries 1000
"""
import argparse

import config
from src.join.order import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dev", choices=["dev", "full"])
    p.add_argument("--n-queries", type=int, default=None)
    p.add_argument("--k", type=int, default=None, help="k-NN graph degree (default ORDER_KNN = 20)")
    a = p.parse_args()
    run(config, a.dataset, a.n_queries, a.k)
