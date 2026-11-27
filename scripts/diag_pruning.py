#!/usr/bin/env python3
"""Pruning diagnostic on dev. Run from the project root:

    python3 -m scripts.diag_pruning
"""
import config
from src.buckets.diag import run

if __name__ == "__main__":
    run(config)
