"""Step 5: feasibility gate.

On the dev subset (X_dev 1K x Y_dev 1M), label every (x, y) pair with the
residual bound for each M and each eps level, check correctness against exact
distances, and report how many pairs end up Maybe.

    Yes   : d(x, y_hat) + r_y <= eps
    No    : d(x, y_hat) - r_y >  eps
    Maybe : otherwise
    (eps = sqrt(eps2), eps2 = integer from eps.json["dev"]; a true match is d(x,y)^2 <= eps2)

Loop order: Y_dev is read once, in chunks. Per chunk the exact distances are
computed once and reused for every M and level. Distances use
||x||^2 + ||y||^2 - 2 x.y in float64 (exact for integer data).

Run: python -m scripts.step5_feasibility
"""
import json
import math
import time
from pathlib import Path

import numpy as np

HEADER_BYTES = 8
LEVELS = ("low", "mid", "high")


def decode(codes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """codes (n, M) uint8, centroids (M, 256, dsub) -> (n, M*dsub) float64."""
    m = centroids.shape[0]
    return np.concatenate([centroids[j][codes[:, j]] for j in range(m)], axis=1).astype(np.float64)


def sq_dists(xq: np.ndarray, x_norm: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared L2 distances (n_x, n_y) in float64, clamped at 0."""
    y_norm = np.einsum("ij,ij->i", y, y)
    d2 = x_norm[:, None] + y_norm[None, :] - 2.0 * (xq @ y.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def run(cfg):
    from src.data.io import read_u8bin

    t_all = time.time()
    data_dir = Path(cfg.DATA_DIR)
    pq_dir = data_dir / "pq"
    gt_dir = data_dir / "ground_truth" / "dev"
    results_dir = Path(getattr(cfg, "RESULTS_DIR", data_dir.parent / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    chunk = getattr(cfg, "FEAS_CHUNK_SIZE", 20_000)
    threshold = getattr(cfg, "MAYBE_MULTIPLIER_THRESHOLD", 50)
    n_y = cfg.N_Y_DEV

    with open(cfg.EPS_JSON) as f:
        eps_dev = json.load(f)["dev"]
    eps2 = {lvl: int(eps_dev[lvl]["eps2"]) for lvl in LEVELS}
    eps = {lvl: math.sqrt(eps2[lvl]) for lvl in LEVELS}

    _, _, X = read_u8bin(cfg.X_DEV_BIN)
    xq = X.astype(np.float64)
    n_x = xq.shape[0]
    x_norm = np.einsum("ij,ij->i", xq, xq)

    gt = {lvl: np.load(gt_dir / lvl / "counts.npy").astype(np.int64) for lvl in LEVELS}

    pq = {}
    for m in cfg.PQ_M_VALUES:
        d = pq_dir / f"M{m}"
        pq[m] = {
            "centroids": np.load(d / "centroids.npy"),
            "codes": np.load(d / "codes.npy", mmap_mode="r"),
            "resid": np.load(d / "residuals.npy", mmap_mode="r"),
        }

    # per-x accumulators
    true_x = {lvl: np.zeros(n_x, np.int64) for lvl in LEVELS}
    acc = {m: {lvl: {k: np.zeros(n_x, np.int64) for k in ("yes", "maybe", "no")}
               for lvl in LEVELS} for m in cfg.PQ_M_VALUES}

    y_all = np.memmap(cfg.Y_DEV_BIN, dtype=np.uint8, mode="r", offset=HEADER_BYTES, shape=(n_y, cfg.D))

    print(f"Step 5: X_dev {n_x:,} x Y_dev {n_y:,}, chunk {chunk:,}, M = {list(cfg.PQ_M_VALUES)}")
    print("  dev eps: " + "  ".join(f"{l}={eps[l]:.2f} (eps2={eps2[l]})" for l in LEVELS), flush=True)

    for start in range(0, n_y, chunk):
        end = min(start + chunk, n_y)
        y = np.asarray(y_all[start:end], dtype=np.float64)
        d2_true = np.rint(sq_dists(xq, x_norm, y))  # exact integers
        for lvl in LEVELS:
            true_x[lvl] += (d2_true <= eps2[lvl]).sum(axis=1)

        for m in cfg.PQ_M_VALUES:
            p = pq[m]
            y_hat = decode(np.asarray(p["codes"][start:end]), p["centroids"])
            d_hat = np.sqrt(sq_dists(xq, x_norm, y_hat))
            r = np.asarray(p["resid"][start:end], dtype=np.float64)[None, :]
            upper = d_hat + r
            lower = d_hat - r
            del d_hat
            for lvl in LEVELS:
                yes = upper <= eps[lvl]
                no = lower > eps[lvl]
                bad_yes = int((yes & (d2_true > eps2[lvl])).sum())
                bad_no = int((no & (d2_true <= eps2[lvl])).sum())
                if bad_yes or bad_no:
                    raise AssertionError(
                        f"M={m} {lvl} rows {start}-{end}: {bad_yes} wrong Yes, {bad_no} wrong No")
                a = acc[m][lvl]
                a["yes"] += yes.sum(axis=1)
                a["no"] += no.sum(axis=1)
                a["maybe"] += (end - start) - yes.sum(axis=1) - no.sum(axis=1)
        print(f"    rows {end:,} / {n_y:,}  ({time.time() - t_all:.0f} s)", flush=True)

    for lvl in LEVELS:
        if not np.array_equal(true_x[lvl], gt[lvl]):
            raise AssertionError(f"{lvl}: exact match counts differ from dev ground truth")
    print("  exact match counts equal dev ground truth for all levels: OK")

    total_pairs = n_x * n_y
    out = {"n_x": n_x, "n_y": n_y, "eps_dev": {l: {"eps": eps[l], "eps2": eps2[l]} for l in LEVELS},
           "per_M": {}, "gate_warning": None}

    print()
    print(f"  {'M':>3} {'level':>5} {'true':>11} {'yes':>11} {'maybe':>13} {'maybe/true':>10} "
          f"{'maybe frac':>10} {'yes/true':>8}")
    for m in cfg.PQ_M_VALUES:
        out["per_M"][str(m)] = {}
        for lvl in LEVELS:
            a = acc[m][lvl]
            t = int(true_x[lvl].sum())
            yes, maybe, no = int(a["yes"].sum()), int(a["maybe"].sum()), int(a["no"].sum())
            assert yes + maybe + no == total_pairs
            has = true_x[lvl] > 0
            ratio = maybe / t if t else float("inf")
            row = {
                "true_matches": t,
                "yes": yes,
                "maybe": maybe,
                "no": no,
                "maybe_per_true": ratio,
                "maybe_frac_all": maybe / total_pairs,
                "yes_per_true": yes / t if t else 0.0,
                "maybe_per_x_mean": float(a["maybe"].mean()),
                "maybe_per_x_max": int(a["maybe"].max()),
                "maybe_x_with_match": int(a["maybe"][has].sum()),
                "maybe_x_without_match": int(a["maybe"][~has].sum()),
            }
            out["per_M"][str(m)][lvl] = row
            print(f"  {m:>3} {lvl:>5} {t:>11,} {yes:>11,} {maybe:>13,} {ratio:>10.1f} "
                  f"{row['maybe_frac_all']:>10.5f} {row['yes_per_true']:>8.3f}")

    if 16 in cfg.PQ_M_VALUES:
        r16 = out["per_M"]["16"]["mid"]["maybe_per_true"]
        if r16 > threshold:
            out["gate_warning"] = f"M=16 mid: maybe/true = {r16:.1f} > {threshold}; try OPQ before Step 6"
            print(f"\n  *** GATE WARNING: {out['gate_warning']} ***")
        else:
            print(f"\n  Gate passed: M=16 mid maybe/true = {r16:.1f} <= {threshold}")

    with open(results_dir / "feasibility.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {results_dir / 'feasibility.json'}  (total {time.time() - t_all:.0f} s)")
    return out