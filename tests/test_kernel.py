"""Step 6 check: unit test for the code-filter kernel on real data.

For each M and each eps level (eps.json["full"]), take 100K (x, y) pairs:
  - 50K uniformly random pairs (mostly far apart -> exercises No and early exit)
  - 50K pairs from the full high-level ground truth (close pairs -> exercises Yes / Maybe)
and check:
  1. every Yes has exact d^2 <= eps2 and every No has exact d^2 > eps2   (must hold)
  2. kernel labels equal a plain NumPy reference (decode + float64 distance)
  3. no true match is labeled No (follows from 1, reported explicitly)
Then time the kernel on one full 1M-row scan per M.

Run from the project root:  python3 -m tests.test_kernel
"""
import json
import math
import time
from pathlib import Path

import numpy as np

import config
from src.filter.kernel import build_lut, count_labels, filter_codes

HEADER_BYTES = 8
LEVELS = ("low", "mid", "high")


def labels_kernel(xq, codes, resid, centroids, x_ids, y_ids, eps):
    """Kernel labels (0 = No, 1 = Maybe, 2 = Yes) for arbitrary pairs, grouped by x."""
    out = np.zeros(len(x_ids), dtype=np.int8)
    order = np.argsort(x_ids, kind="stable")
    xs = x_ids[order]
    bounds = np.flatnonzero(np.diff(xs)) + 1
    for grp in np.split(order, bounds):
        if len(grp) == 0:
            continue
        lut = build_lut(xq[x_ids[grp[0]]], centroids)
        yi = y_ids[grp]
        yes, maybe = filter_codes(lut, np.ascontiguousarray(codes[yi]), np.asarray(resid[yi]), eps)
        out[grp[maybe]] = 1
        out[grp[yes]] = 2
    return out


def labels_reference(xq, codes, resid, centroids, x_ids, y_ids, eps):
    m = centroids.shape[0]
    c = np.asarray(codes[y_ids])
    y_hat = np.concatenate([centroids[j][c[:, j]] for j in range(m)], axis=1).astype(np.float64)
    d_hat = np.linalg.norm(xq[x_ids] - y_hat, axis=1)
    r = np.asarray(resid[y_ids], dtype=np.float64)
    lab = np.ones(len(x_ids), dtype=np.int8)
    lab[d_hat + r <= eps] = 2
    lab[d_hat - r > eps] = 0
    return lab


def run(cfg, n_random=50_000, n_close=50_000):
    from src.data.io import read_u8bin

    data_dir = Path(cfg.DATA_DIR)
    seed = cfg.RANDOM_SEED
    rng = np.random.default_rng(seed + 6)

    with open(cfg.EPS_JSON) as f:
        eps_full = json.load(f)["full"]
    _, _, X = read_u8bin(cfg.X_BIN)
    xq = X.astype(np.float64)
    y_all = np.memmap(cfg.Y_BIN, dtype=np.uint8, mode="r", offset=HEADER_BYTES, shape=(cfg.N_Y_FULL, cfg.D))

    # pairs: random + close (from high-level ground truth)
    rx = rng.integers(0, xq.shape[0], n_random)
    ry = rng.integers(0, cfg.N_Y_FULL, n_random)
    gt = np.load(data_dir / "ground_truth" / "full" / "high" / "pairs.npy", mmap_mode="r")
    pick = np.sort(rng.choice(len(gt), size=min(n_close, len(gt)), replace=False))
    close = np.asarray(gt[pick]).astype(np.int64)
    x_ids = np.concatenate([rx, close[:, 0]])
    y_ids = np.concatenate([ry, close[:, 1]])

    # exact squared distances (integers)
    o = np.argsort(y_ids)
    yv = np.empty((len(y_ids), cfg.D), dtype=np.float64)
    yv[o] = np.asarray(y_all[y_ids[o]], dtype=np.float64)
    d2_exact = np.rint(((xq[x_ids] - yv) ** 2).sum(axis=1)).astype(np.int64)
    print(f"Testing {len(x_ids):,} pairs ({n_random:,} random + {len(close):,} close) per M and level\n")

    print(f"  {'M':>3} {'level':>5} {'yes':>7} {'maybe':>7} {'no':>7} {'true':>7}  "
          f"{'wrong yes':>9} {'wrong no':>8} {'!= ref':>6}")
    failures = 0
    for m in cfg.PQ_M_VALUES:
        d = data_dir / "pq" / f"M{m}"
        centroids = np.load(d / "centroids.npy")
        codes = np.load(d / "codes.npy", mmap_mode="r")
        resid = np.load(d / "residuals.npy", mmap_mode="r")
        for lvl in LEVELS:
            eps2 = int(eps_full[lvl]["eps2"])
            eps = math.sqrt(eps2)
            k = labels_kernel(xq, codes, resid, centroids, x_ids, y_ids, eps)
            ref = labels_reference(xq, codes, resid, centroids, x_ids, y_ids, eps)
            true = d2_exact <= eps2
            wrong_yes = int(((k == 2) & ~true).sum())
            wrong_no = int(((k == 0) & true).sum())
            diff_ref = int((k != ref).sum())
            failures += wrong_yes + wrong_no
            print(f"  {m:>3} {lvl:>5} {int((k == 2).sum()):>7,} {int((k == 1).sum()):>7,} "
                  f"{int((k == 0).sum()):>7,} {int(true.sum()):>7,}  {wrong_yes:>9} {wrong_no:>8} {diff_ref:>6}")
            if diff_ref:
                print(f"        note: {diff_ref} labels differ from the NumPy reference "
                      f"(only acceptable if tiny and at the exact boundary)")

    # throughput: one query against the first 1M codes
    print("\n  Throughput (one query vs up to 1,000,000 codes, mid eps):")
    eps_mid = math.sqrt(int(eps_full["mid"]["eps2"]))
    for m in cfg.PQ_M_VALUES:
        d = data_dir / "pq" / f"M{m}"
        centroids = np.load(d / "centroids.npy")
        codes = np.ascontiguousarray(np.load(d / "codes.npy", mmap_mode="r")[:1_000_000])
        resid = np.ascontiguousarray(np.load(d / "residuals.npy", mmap_mode="r")[:1_000_000])
        n_codes = len(codes)
        lut = build_lut(xq[0], centroids)
        filter_codes(lut, codes[:10], resid[:10], eps_mid)  # compile
        count_labels(lut, codes[:10], resid[:10], eps_mid)
        t0 = time.perf_counter()
        reps = 5
        for q in range(reps):
            lut = build_lut(xq[q], centroids)
            filter_codes(lut, codes, resid, eps_mid)
        dt = (time.perf_counter() - t0) / reps
        print(f"    M={m:>2}: {dt * 1e3:6.1f} ms for {n_codes:,} codes  ({dt * 1e9 / n_codes:.2f} ns per code)")

    print()
    if failures:
        raise AssertionError(f"{failures} wrong Yes/No labels: the kernel breaks the bound")
    print("PASS: every Yes is a true match and every No is a true non-match, for all M and levels.")


if __name__ == "__main__":
    run(config)