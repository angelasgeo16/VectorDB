"""Step 2: pick ε thresholds via k-NN pooling + range-search validation + bisection.

Algorithm:
1. Exact k-NN (k = EPS_K_NN) of X_dev against Y, read in chunks (uint8 -> float32).
2. Pool all n_x * k squared distances, take the (target * n_x)-th smallest.
   That ε gives mean matches per x = target (as long as no x has > k matches).
3. Validate each ε with chunked range_search; report mean matches and share of x with >= 1 match.
4. If mean is off by more than EPS_TOLERANCE, bracket and bisect ε until within tolerance.
5. Do this for dev Y (1M) and full Y (10M); save eps.json.

Distance convention (important for later steps):
  BigANN vectors are uint8, so every squared L2 distance is an integer.
  We store eps2 as an INTEGER and define a match as  d(x, y)^2 <= eps2.
  FAISS range_search keeps d^2 < radius (strict), so always call it with
  radius = eps2 + 0.5. Use faiss_radius() for this everywhere.

Run: python -m scripts.step2_pick_eps
"""
import json
import math
from pathlib import Path

import faiss
import numpy as np

HEADER_BYTES = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def faiss_radius(eps2: int) -> float:
    """Radius to pass to FAISS range_search so that matches are d^2 <= eps2."""
    return float(eps2) + 0.5


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


# ---------------------------------------------------------------------------
# Chunked exact k-NN
# ---------------------------------------------------------------------------
def chunked_knn(xq: np.ndarray, y_path: Path, k: int, chunk_size: int, total_n: int, d: int):
    """Exact k-NN of xq against the first total_n rows of y_path.

    Returns (dists2, ids), each (n_x, k), sorted ascending by squared distance.
    """
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    n_x = xq.shape[0]
    best_d = np.full((n_x, k), np.inf, dtype=np.float32)
    best_i = np.full((n_x, k), -1, dtype=np.int64)
    rows = np.arange(n_x)[:, None]

    for offset, y_chunk in iter_y_chunks(y_path, total_n, chunk_size, d):
        index = faiss.IndexFlatL2(d)
        index.add(y_chunk)
        kk = min(k, y_chunk.shape[0])
        d_chunk, i_chunk = index.search(xq, kk)

        merged_d = np.hstack([best_d, d_chunk])
        merged_i = np.hstack([best_i, i_chunk + offset])
        part = np.argpartition(merged_d, k - 1, axis=1)[:, :k]
        best_d = merged_d[rows, part]
        best_i = merged_i[rows, part]

        done = offset + y_chunk.shape[0]
        print(f"    k-NN: {done:,} / {total_n:,} rows", flush=True)

    order = np.argsort(best_d, axis=1)
    return np.take_along_axis(best_d, order, axis=1), np.take_along_axis(best_i, order, axis=1)


# ---------------------------------------------------------------------------
# ε from pooled distances
# ---------------------------------------------------------------------------
def pooled_eps2(knn_dists2: np.ndarray, target: int) -> int:
    """Integer eps2 such that the pooled count of d^2 <= eps2 is about target * n_x."""
    n_x = knn_dists2.shape[0]
    flat = knn_dists2.ravel()
    kth = target * n_x - 1
    if kth >= flat.size:
        raise ValueError(f"target {target} too large for k={knn_dists2.shape[1]}")
    val = np.partition(flat, kth)[kth]
    return int(round(float(val)))  # squared distances are integers for uint8 data


def check_truncation(knn_dists2: np.ndarray, eps2: int) -> float:
    """Share of x whose k-th neighbor is still within eps2 (their count got cut at k)."""
    return float((knn_dists2[:, -1] <= eps2).mean())


# ---------------------------------------------------------------------------
# Chunked range_search validation
# ---------------------------------------------------------------------------
def chunked_range_counts(xq: np.ndarray, y_path: Path, eps2: int, chunk_size: int, total_n: int, d: int):
    """Number of y with d(x, y)^2 <= eps2, per query."""
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    counts = np.zeros(xq.shape[0], dtype=np.int64)
    radius = faiss_radius(eps2)
    for _, y_chunk in iter_y_chunks(y_path, total_n, chunk_size, d):
        index = faiss.IndexFlatL2(d)
        index.add(y_chunk)
        lims, _, _ = index.range_search(xq, radius)
        counts += np.diff(lims).astype(np.int64)
    return counts


def evaluate(xq, y_path, eps2, chunk_size, total_n, d):
    counts = chunked_range_counts(xq, y_path, eps2, chunk_size, total_n, d)
    return float(counts.mean()), float((counts > 0).mean())


# ---------------------------------------------------------------------------
# Refinement: bracket + bisection on integer eps2
# ---------------------------------------------------------------------------
def refine_eps2(xq, y_path, target, eps2_start, chunk_size, total_n, d, tolerance, max_iter):
    """Adjust eps2 until mean matches is within tolerance of target.

    Returns (eps2, mean_matches, share_ge1, n_evaluations).
    """
    def close(m):
        return abs(m - target) / target <= tolerance

    evals = 0
    mean_m, share = evaluate(xq, y_path, eps2_start, chunk_size, total_n, d)
    evals += 1
    print(f"    eps2={eps2_start}: mean={mean_m:.1f}, share>=1={share:.3f}", flush=True)
    if close(mean_m):
        return eps2_start, mean_m, share, evals

    best = (eps2_start, mean_m, share)

    # Bracket: lo has mean < target, hi has mean > target.
    if mean_m > target:
        hi, lo = eps2_start, 0
    else:
        lo, hi = eps2_start, None
        step = max(1, eps2_start)
        while hi is None and evals < max_iter:
            cand = lo + step
            m, s = evaluate(xq, y_path, cand, chunk_size, total_n, d)
            evals += 1
            print(f"    bracket eps2={cand}: mean={m:.1f}", flush=True)
            if abs(m - target) < abs(best[1] - target):
                best = (cand, m, s)
            if close(m):
                return cand, m, s, evals
            if m > target:
                hi = cand
            else:
                lo, step = cand, step * 2
        if hi is None:
            print("    WARNING: could not bracket target; returning closest", flush=True)
            return (*best, evals)

    while evals < max_iter and hi - lo > 1:
        mid = (lo + hi) // 2
        m, s = evaluate(xq, y_path, mid, chunk_size, total_n, d)
        evals += 1
        print(f"    bisect eps2={mid}: mean={m:.1f}", flush=True)
        if abs(m - target) < abs(best[1] - target):
            best = (mid, m, s)
        if close(m):
            return mid, m, s, evals
        if m > target:
            hi = mid
        else:
            lo = mid

    print("    WARNING: tolerance not reached; returning closest", flush=True)
    return (*best, evals)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
LEVELS = ("low", "mid", "high")


def pick_for_dataset(name, xq, y_path, total_n, *, k, chunk_size, d, targets, tolerance, max_iter):
    print(f"\n== {name}: Y = {total_n:,} rows, X = {xq.shape[0]:,} queries ==", flush=True)
    knn_d2, _ = chunked_knn(xq, y_path, k, chunk_size, total_n, d)

    out = {}
    for level, target in zip(LEVELS, targets):
        print(f"  [{level}] target mean = {target}", flush=True)
        eps2_0 = pooled_eps2(knn_d2, target)
        trunc = check_truncation(knn_d2, eps2_0)
        if trunc > 0:
            print(f"    note: {trunc:.1%} of x have >= k matches at the pooled eps; refining with range_search", flush=True)
        eps2, mean_m, share, evals = refine_eps2(
            xq, y_path, target, eps2_0, chunk_size, total_n, d, tolerance, max_iter
        )
        out[level] = {
            "target": target,
            "eps2": int(eps2),
            "eps": math.sqrt(eps2),
            "faiss_radius": faiss_radius(eps2),
            "mean_matches": mean_m,
            "share_ge1": share,
            "pooled_eps2_initial": int(eps2_0),
            "share_truncated_at_k": trunc,
            "range_search_evals": evals,
        }
        print(f"    -> eps={math.sqrt(eps2):.2f} (eps2={eps2}), mean={mean_m:.1f}, share>=1={share:.3f}", flush=True)
    return out


def run(cfg):
    """cfg: the config module (or any object with the same attributes)."""
    from src.data.io import read_u8bin

    _, _, X_dev = read_u8bin(cfg.X_DEV_BIN)
    xq = X_dev[: cfg.EPS_SAMPLE_X].astype(np.float32)

    common = dict(
        k=cfg.EPS_K_NN,
        chunk_size=cfg.EPS_CHUNK_SIZE,
        d=cfg.D,
        targets=cfg.EPS_TARGET_MEAN,
        tolerance=cfg.EPS_TOLERANCE,
        max_iter=cfg.EPS_BISECT_MAX_ITER,
    )

    result = {
        "params": {
            "k_nn": cfg.EPS_K_NN,
            "n_x": int(xq.shape[0]),
            "query_sample": "first EPS_SAMPLE_X rows of X_dev",
            "targets": list(cfg.EPS_TARGET_MEAN),
            "tolerance": cfg.EPS_TOLERANCE,
            "y_dev_size": cfg.N_Y_DEV,
            "y_full_size": cfg.N_Y_FULL,
            "match_rule": "d(x,y)^2 <= eps2 (integer); FAISS radius = eps2 + 0.5",
        },
        "dev": pick_for_dataset("dev", xq, cfg.Y_DEV_BIN, cfg.N_Y_DEV, **common),
        "full": pick_for_dataset("full", xq, cfg.Y_BIN, cfg.N_Y_FULL, **common),
    }

    Path(cfg.EPS_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.EPS_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {cfg.EPS_JSON}")
    return result