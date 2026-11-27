#!/usr/bin/env python3
"""Step 5: feasibility gate on the dev subset. Run from the project root:

    python3 -m scripts.step5_feasibility
"""
import config
from src.filter.feasibility import run

if __name__ == "__main__":
    run(config)