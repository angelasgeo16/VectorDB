"""Step 4: train PQ (M = 8, 16, 32), encode all of Y, compute residuals.

For each M:
  1. Train a FAISS ProductQuantizer (8 bits -> 256 centroids per subspace)
     on a random sample of Y (PQ_TRAIN_SAMPLE rows, seeded).
  2. Encode all N_Y_FULL vectors in chunks -> codes (N, M) uint8.
  3. Residual r_y = ||y - decode(code_y)||, computed in float64, then stored as
     float16 ROUNDED UP (r16 >= r for every y), so the bound
         d(x, y_hat) - r_y <= d(x, y) <= d(x, y_hat) + r_y
     stays valid after compression.
  4. Checks: round-up holds for every y; the bound holds on random (x, y)
     pairs; residual stats are printed next to the eps values.

Outputs (original Y order; Step 7 reorders by bucket later):
    data/pq/M{M}/pq.faiss         FAISS ProductQuantizer
    data/pq/M{M}/centroids.npy    (M, 256, D/M) float32, for the Numba kernel (Step 6)
    data/pq/M{M}/codes.npy        (N_Y_FULL, M) uint8
    data/pq/M{M}/residuals.npy    (N_Y_FULL,) float16, rounded up
    data/pq/summary.json

The dev subset is Y's first N_Y_DEV rows, so dev codes/residuals are just
codes[:N_Y_DEV] and residuals[:N_Y_DEV].

Run: python -m scripts.step4_pq
"""
import json
import time
from pathlib import Path

import faiss
import numpy as np

HEADER_BYTES = 8


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def open_y(y_path: Path, n: int, d: int) -> np.memmap:
    """Memory-map the first n rows of a .u8bin file as (n, d) uint8."""
    return np.memmap(y_path, dtype=np.uint8, mode="r", offset=HEADER_BYTES, shape=(n, d))


def sample_rows(y: np.memmap, n_sample: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(y.shape[0], size=min(n_sample, y.shape[0]), replace=False))
    return np.ascontiguousarray(y[idx], dtype=np.float32)


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------
def float16_round_up(r: np.ndarray) -> np.ndarray:
    """Cast float64 values to float16 so that every result is >= the input."""
    r16 = r.astype(np.float16)
    low = r16.astype(np.float64) < r
    r16[low] = np.nextafter(r16[low], np.float16(np.inf))
    if not np.all(r16.astype(np.float64) >= r):
        raise AssertionError("float16 round-up failed")
    if np.isinf(r16).any():
        raise AssertionError("residual overflowed float16")
    return r16


def residuals_exact(y_chunk_u8: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    """||y - y_hat|| in float64 (y is integer-valued, y_hat is the float32 decode)."""
    diff = y_chunk_u8.astype(np.float64) - decoded.astype(np.float64)
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


# ---------------------------------------------------------------------------
# Train + encode one M
# ---------------------------------------------------------------------------
def train_pq(train_x: np.ndarray, d: int, m: int, n_iter: int, seed: int):
    if d % m != 0:
        raise ValueError(f"D={d} not divisible by M={m}")
    pq = faiss.ProductQuantizer(d, m, 8)
    pq.cp.niter = n_iter
    pq.cp.seed = seed
    pq.train(train_x)
    return pq


def centroids_array(pq) -> np.ndarray:
    return faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub).astype(np.float32)


def encode_all(pq, y: np.memmap, chunk_size: int):
    n = y.shape[0]
    codes = np.empty((n, pq.M), dtype=np.uint8)
    resid = np.empty(n, dtype=np.float16)
    resid_max_err = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_u8 = np.asarray(y[start:end])
        chunk = chunk_u8.astype(np.float32)
        c = pq.compute_codes(chunk)
        dec = pq.decode(c)
        r = residuals_exact(chunk_u8, dec)
        r16 = float16_round_up(r)
        codes[start:end] = c
        resid[start:end] = r16
        resid_max_err = max(resid_max_err, float((r16.astype(np.float64) - r).max()))
        print(f"      encoded {end:,} / {n:,}", flush=True)
    return codes, resid, resid_max_err


# ---------------------------------------------------------------------------
# Check: the residual bound holds on random (x, y) pairs
# ---------------------------------------------------------------------------
def check_bound(pq, x: np.ndarray, y: np.memmap, codes: np.ndarray, resid: np.ndarray,
                n_pairs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    xi = rng.integers(0, x.shape[0], n_pairs)
    yi = rng.integers(0, y.shape[0], n_pairs)
    xv = x[xi].astype(np.float64)
    yv = np.asarray(y[np.sort(yi)])  # memmap fancy-index is faster when sorted
    order = np.argsort(np.argsort(yi))
    yv = yv[order].astype(np.float64)
    yhat = pq.decode(codes[yi]).astype(np.float64)
    r = resid[yi].astype(np.float64)

    d_true = np.linalg.norm(xv - yv, axis=1)
    d_hat = np.linalg.norm(xv - yhat, axis=1)
    viol_hi = int((d_true > d_hat + r + 1e-9).sum())
    viol_lo = int((d_true < d_hat - r - 1e-9).sum())
    if viol_hi or viol_lo:
        raise AssertionError(f"bound violated: {viol_hi} above, {viol_lo} below")
    return {"pairs": n_pairs, "violations": 0,
            "mean_abs_error_d_hat": float(np.abs(d_true - d_hat).mean())}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(cfg):
    from src.data.io import read_u8bin

    t_all = time.time()
    y = open_y(cfg.Y_BIN, cfg.N_Y_FULL, cfg.D)
    _, _, X = read_u8bin(cfg.X_BIN)
    eps = None
    if Path(cfg.EPS_JSON).exists():
        with open(cfg.EPS_JSON) as f:
            eps = json.load(f)["full"]

    chunk_size = getattr(cfg, "PQ_ENCODE_CHUNK", 1_000_000)
    n_check = getattr(cfg, "PQ_BOUND_CHECK_PAIRS", 100_000)
    out_root = Path(cfg.DATA_DIR) / "pq"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {cfg.PQ_TRAIN_SAMPLE:,} training vectors (seed {cfg.RANDOM_SEED})", flush=True)
    train_x = sample_rows(y, cfg.PQ_TRAIN_SAMPLE, cfg.RANDOM_SEED)

    summary = {"n_y": cfg.N_Y_FULL, "train_sample": int(train_x.shape[0]),
               "seed": cfg.RANDOM_SEED, "per_M": {}}

    for m in cfg.PQ_M_VALUES:
        print(f"\n== M = {m} ({cfg.D // m} dims per sub-vector) ==", flush=True)
        out = out_root / f"M{m}"
        out.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        pq = train_pq(train_x, cfg.D, m, cfg.PQ_N_ITER, cfg.RANDOM_SEED)
        t_train = time.time() - t0
        print(f"    trained in {t_train:.1f} s", flush=True)

        t0 = time.time()
        codes, resid, max_round = encode_all(pq, y, chunk_size)
        t_encode = time.time() - t0
        print(f"    encoded in {t_encode:.1f} s", flush=True)

        faiss.write_ProductQuantizer(pq, str(out / "pq.faiss"))
        np.save(out / "centroids.npy", centroids_array(pq))
        np.save(out / "codes.npy", codes)
        np.save(out / "residuals.npy", resid)

        bound = check_bound(pq, X, y, codes, resid, n_check, cfg.RANDOM_SEED)
        r = resid.astype(np.float64)
        stats = {
            "train_s": t_train,
            "encode_s": t_encode,
            "ram_codes_MB": codes.nbytes / 1e6,
            "ram_residuals_MB": resid.nbytes / 1e6,
            "residual_mean": float(r.mean()),
            "residual_median": float(np.median(r)),
            "residual_p95": float(np.percentile(r, 95)),
            "residual_max": float(r.max()),
            "dev_residual_mean": float(r[: cfg.N_Y_DEV].mean()),
            "float16_roundup_max_added": max_round,
            "bound_check": bound,
        }
        summary["per_M"][str(m)] = stats

        print(f"    residual mean={stats['residual_mean']:.2f}  median={stats['residual_median']:.2f}  "
              f"p95={stats['residual_p95']:.2f}  max={stats['residual_max']:.2f}", flush=True)
        if eps:
            print("    eps: " + "  ".join(f"{k}={v['eps']:.1f}" for k, v in eps.items())
                  + f"   (Maybe band width is about 2 x residual = {2 * stats['residual_mean']:.1f})",
                  flush=True)
        print(f"    bound check: {bound['pairs']:,} random pairs, 0 violations", flush=True)
        print(f"    RAM: codes {stats['ram_codes_MB']:.0f} MB + residuals {stats['ram_residuals_MB']:.0f} MB",
              flush=True)

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved under {out_root}  (total {time.time() - t_all:.1f} s)")
    return summary