#!/usr/bin/env python3
"""Step 0: obtain the bigann-10M dataset.

Clones big-ann-benchmarks (plain clone, gitignored) into external/ and runs
`create_dataset.py --dataset bigann-10M`, which downloads the first 10M base
vectors plus the 10K public queries. We then verify the produced files by
reading only their 8-byte headers.

Run from the project root:
    python -m scripts.prepare_data
"""
import struct
import subprocess
import sys

from config import BASE_BIN, EXTERNAL_DIR, QUERY_BIN

REPO_URL = "https://github.com/harsha-simhadri/big-ann-benchmarks.git"

# (path, human name, expected row count)
EXPECTED_FILES = [
    (BASE_BIN, "base", 10_000_000),
    (QUERY_BIN, "query", 10_000),
]


def prepare_data():
    # 1. Clone the benchmark repo if it is not already present.
    if not EXTERNAL_DIR.exists():
        print(f"Cloning {REPO_URL} -> {EXTERNAL_DIR}")
        EXTERNAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, str(EXTERNAL_DIR)], check=True)
    else:
        print(f"Repo already exists at {EXTERNAL_DIR}")

    # 2. Create the 10M dataset (downloads + crops base, fetches queries).
    create_script = EXTERNAL_DIR / "create_dataset.py"
    if not create_script.exists():
        raise FileNotFoundError(f"create_dataset.py not found at {create_script}")

    print("Running create_dataset.py --dataset bigann-10M ...")
    subprocess.run(
        [sys.executable, "create_dataset.py", "--dataset", "bigann-10M"],
        cwd=EXTERNAL_DIR,
        check=True,
    )

    # 3. Verify the output files: existence + header (read only the first 8 bytes).
    for path, name, expected_n in EXPECTED_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Expected {name} file not found: {path}")

        with open(path, "rb") as f:
            header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"{path}: file too short for a header")

        n, d = struct.unpack("<II", header)
        print(f"  {path.name}: n={n:,}, d={d}")
        if n != expected_n:
            raise ValueError(f"{path.name}: expected n={expected_n:,}, got {n:,}")
        if d != 128:
            raise ValueError(f"{path.name}: expected d=128, got {d}")

    print("Data preparation complete.")


if __name__ == "__main__":
    prepare_data()
