#!/usr/bin/env python3
"""Step 16: figures + summary tables from results/runs.csv. From the project root:

    python3 -m scripts.step16_plots

Safe to run any time; figures whose runs are missing are skipped with a note.
"""
import config
from src.experiments.plots import run

if __name__ == "__main__":
    run(config)
