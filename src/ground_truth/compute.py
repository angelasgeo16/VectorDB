"""Step 3: compute exact range-search ground truth.

One pass over Y per dataset at the HIGH level's faiss_radius; mid/low are
obtained by filtering the returned squared distances with d^2 <= eps2.
Matches are defined as d(x, y)^2 <= eps2 (integer), same rule as Step 2.

Outputs, per dataset (dev / full) and level (low / mid / high):
    data/ground_truth/<dataset>/<level>/pairs.npy   (n_pairs, 2) int32, (x_id, y_id), sorted
    data/ground_truth/<dataset>/<level>/counts.npy  (n_x,) int32, matches per x
    data/ground_truth/summary.json

Validation:
    dev                 -> counts must exactly reproduce eps.json["dev"] stats
    full, first 1K of X -> counts must exactly reproduce eps.json["full"] stats
                           (X_dev is the first 1K rows of X)
    full, all 10K       -> printed only

Run: python -m scripts.step3_ground_truth
"""
import json
import math
from pathlib import Path

import faiss
import numpy as np

HEADER_BYTES = 8
LEVELS = ("low", "mid", "high")


def iter_y_chunks(y_path: Path, total_n: int, chunk_size: int, d: int):
    """Yield (offset, float32 chunk) for the first total_n rows of a .u8bin file."""
    with open(y_path, "rb") as f:
        f.seek(HEADER_BYTES)
        processed = 0
        while processed < total_n:
            this_chunk = min(chunk_size, total_n - processed)
            raw = f.read(this_chunk * d)
            if len(raw) < this_chunk * d:
                raise EOFError(f"{y_path}: unexpected EOF at row {processed}")
            chunk = np.frombuffer(raw, dtype=np.uint8).reshape(this_chunk, d).astype(np.float32)
            yield processed, chunk
            processed += this_chunk


def range_ground_truth(xq: np.ndarray, y_path: Path, total_n: int, eps_section: dict,
                       chunk_size: int, d: int):
    """Single pass over Y. Returns {level: (pairs int32 (n,2) sorted, counts int32 (n_x,))}."""
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    n_x = xq.shape[0]
    high_radius = float(eps_section["high"]["faiss_radius"])
    eps2 = {lvl: int(eps_section[lvl]["eps2"]) for lvl in LEVELS}

    # Sanity: high must be the widest level, and its radius must be eps2 + 0.5.
    assert eps2["low"] <= eps2["mid"] <= eps2["high"], eps2
    assert abs(high_radius - (eps2["high"] + 0.5)) < 1e-9, (high_radius, eps2["high"])

    parts = {lvl: [] for lvl in LEVELS}
    counts = {lvl: np.zeros(n_x, dtype=np.int64) for lvl in LEVELS}

    for offset, y_chunk in iter_y_chunks(y_path, total_n, chunk_size, d):
        index = faiss.IndexFlatL2(d)
        index.add(y_chunk)
        lims, dist2, ids = index.range_search(xq, high_radius)

        x_ids = np.repeat(np.arange(n_x, dtype=np.int64), np.diff(lims.astype(np.int64)))
        y_ids = ids.astype(np.int64) + offset
        # uint8 data -> squared distances are integers (exact in float32 below 2^24)
        dist2_int = np.rint(dist2).astype(np.int64)

        for lvl in LEVELS:
            keep = dist2_int <= eps2[lvl]
            xk, yk = x_ids[keep], y_ids[keep]
            counts[lvl] += np.bincount(xk, minlength=n_x)
            parts[lvl].append(np.stack([xk, yk], axis=1).astype(np.int32))

        done = offset + y_chunk.shape[0]
        print(f"    chunk done: {done:,} / {total_n:,} rows "
              f"(high-level pairs so far: {int(counts['high'].sum()):,})", flush=True)

    result = {}
    for lvl in LEVELS:
        pairs = np.concatenate(parts[lvl], axis=0) if parts[lvl] else np.zeros((0, 2), np.int32)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[order]
        c = counts[lvl].astype(np.int32)
        assert len(pairs) == int(c.sum()), (lvl, len(pairs), int(c.sum()))
        result[lvl] = (pairs, c)
    return result


def stats(counts: np.ndarray):
    return float(counts.mean()), float((counts > 0).mean())


def check_exact(label: str, counts: np.ndarray, expected: dict):
    """Assert mean matches and share>=1 reproduce eps.json exactly."""
    mean_m, share = stats(counts)
    ok = math.isclose(mean_m, expected["mean_matches"], rel_tol=0, abs_tol=1e-9) and \
         math.isclose(share, expected["share_ge1"], rel_tol=0, abs_tol=1e-9)
    status = "OK" if ok else "MISMATCH"
    print(f"    [{status}] {label}: mean={mean_m:.3f} (eps.json {expected['mean_matches']:.3f}), "
          f"share>=1={share:.4f} (eps.json {expected['share_ge1']:.4f})", flush=True)
    if not ok:
        raise AssertionError(f"{label}: ground truth does not reproduce eps.json")


def save_level(out_dir: Path, pairs: np.ndarray, counts: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pairs.npy", pairs)
    np.save(out_dir / "counts.npy", counts)


def run(cfg):
    """cfg: the config module (or any object with the same attributes)."""
    from src.data.io import read_u8bin

    with open(cfg.EPS_JSON) as f:
        eps = json.load(f)

    n_check = int(eps["params"]["n_x"])  # queries used in Step 2 (first n of X_dev)
    gt_dir = Path(cfg.DATA_DIR) / "ground_truth"
    summary = {"match_rule": "d(x,y)^2 <= eps2", "dev": {}, "full": {}}

    datasets = [
        ("dev", cfg.X_DEV_BIN, cfg.Y_DEV_BIN, cfg.N_Y_DEV),
        ("full", cfg.X_BIN, cfg.Y_BIN, cfg.N_Y_FULL),
    ]

    for name, x_path, y_path, n_y in datasets:
        _, _, X = read_u8bin(x_path)
        xq = X.astype(np.float32)
        print(f"\n== {name}: X = {xq.shape[0]:,} queries, Y = {n_y:,} rows ==", flush=True)

        res = range_ground_truth(xq, y_path, n_y, eps[name], cfg.GT_CHUNK_SIZE, cfg.D)

        for lvl in LEVELS:
            pairs, counts = res[lvl]
            save_level(gt_dir / name / lvl, pairs, counts)

            # exact reproduction of Step 2 on the same queries
            check_exact(f"{name}/{lvl} (first {n_check} x)", counts[:n_check], eps[name][lvl])

            mean_m, share = stats(counts)
            print(f"    {name}/{lvl} all {len(counts):,} x: pairs={len(pairs):,}, "
                  f"mean={mean_m:.2f}, share>=1={share:.4f}, max per x={int(counts.max())}", flush=True)
            summary[name][lvl] = {
                "eps2": int(eps[name][lvl]["eps2"]),
                "n_x": int(len(counts)),
                "n_pairs": int(len(pairs)),
                "mean_matches": mean_m,
                "share_ge1": share,
                "max_matches_per_x": int(counts.max()),
            }

    with open(gt_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved ground truth under {gt_dir}")
    return summary