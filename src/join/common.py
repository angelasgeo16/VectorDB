"""Shared pieces for the join runs (B1 now; B3 / B2 later).

- peak_rss_mb(): peak resident memory of this process
- exact_matches(): Numba kernel, exact d^2 <= eps2 check of one query against many vectors
- PairWriter: streams result pairs to disk so results never sit in RAM
- load_order(): query processing order (id / random / saved file)
- append_run(): one row per run to results/runs.csv
- recall_check(): compare written pairs with the ground truth (after the timed run)
"""
import csv
import resource
import sys
from pathlib import Path

import numpy as np
from numba import njit


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # macOS: bytes, Linux: KB


@njit(cache=True, nogil=True)
def exact_matches(vecs, x, eps2):
    """Row indices i with ||vecs[i] - x||^2 <= eps2 (uint8 vecs, int64 x, integer eps2).
    Stops summing a row as soon as it exceeds eps2."""
    n, d = vecs.shape
    out = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):
        s = 0
        for j in range(d):
            t = np.int64(vecs[i, j]) - x[j]
            s += t * t
            if s > eps2:
                break
        if s <= eps2:
            out[k] = i
            k += 1
    return out[:k].copy()


class PairWriter:
    """Buffers (x_id, sorted_position) pairs and appends them to a binary int32 file."""

    def __init__(self, path, flush_rows=1_000_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "wb")
        self.flush_rows = flush_rows
        self.buf_x, self.buf_p = [], []
        self.n_buf = 0
        self.n_pairs = 0

    def add(self, x_id, positions):
        if len(positions) == 0:
            return
        self.buf_x.append(np.full(len(positions), x_id, dtype=np.int32))
        self.buf_p.append(positions.astype(np.int32))
        self.n_buf += len(positions)
        if self.n_buf >= self.flush_rows:
            self.flush()

    def flush(self):
        if self.n_buf:
            np.stack([np.concatenate(self.buf_x), np.concatenate(self.buf_p)], axis=1).tofile(self.f)
            self.n_pairs += self.n_buf
            self.buf_x, self.buf_p, self.n_buf = [], [], 0

    def close(self):
        self.flush()
        self.f.close()
        return self.n_pairs


def load_order(kind, n_x, seed=0, order_dir=None):
    if kind == "id":
        return np.arange(n_x, dtype=np.int64)
    if kind == "random":
        return np.random.default_rng(seed + 11).permutation(n_x).astype(np.int64)
    path = Path(kind)
    if order_dir is not None and not kind.endswith(".npy"):
        path = Path(order_dir) / f"{kind}_n{n_x}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run python3 -m scripts.step11_order first")
    order = np.load(path).astype(np.int64)
    assert np.array_equal(np.sort(order), np.arange(n_x)), "order file is not a permutation"
    return order


RUN_FIELDS = [
    "system", "dataset", "level", "eps2", "M", "pruning", "q", "order", "read_policy",
    "cache_MB", "cache_policy", "n_x", "n_candidates", "n_pairs", "recall", "precision", "proven_recall",
    "maybe_frac", "bytes_read_MB", "device_MB", "n_requests", "cache_hit_rate",
    "plan_s", "filter_s", "io_s", "verify_s", "total_s", "peak_rss_MB", "over_cap",
]


def append_run(results_dir, row):
    path = Path(results_dir) / "runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUN_FIELDS})
    return path


def recall_check(pairs_path, ids_path, gt_pairs_path, n_y, n_x=None):
    """Map sorted positions to original ids and compare with ground truth pairs
    (only queries x < n_x if given). Returns (recall, precision, n_found, n_true)."""
    found = np.fromfile(pairs_path, dtype=np.int32).reshape(-1, 2)
    ids = np.load(ids_path)
    fx = found[:, 0].astype(np.int64)
    fy = ids[found[:, 1]].astype(np.int64)
    fkey = np.unique(fx * n_y + fy)
    gt = np.load(gt_pairs_path).astype(np.int64)
    if n_x is not None:
        gt = gt[gt[:, 0] < n_x]
    gkey = np.unique(gt[:, 0] * n_y + gt[:, 1])
    inter = np.intersect1d(fkey, gkey, assume_unique=True).size
    recall = inter / len(gkey) if len(gkey) else 1.0
    precision = inter / len(fkey) if len(fkey) else 1.0
    return recall, precision, int(len(found)), int(len(gkey))


@njit(cache=True, nogil=True)
def exact_matches_multi(vecs, xs, eps2):
    """All (query row, vector row) pairs with ||vecs[i] - xs[q]||^2 <= eps2.
    vecs: (n, d) uint8; xs: (k, d) int64. Returns (q_idx, v_idx) int64 arrays."""
    n, d = vecs.shape
    k = xs.shape[0]
    cap = 1024
    qo = np.empty(cap, dtype=np.int64)
    vo = np.empty(cap, dtype=np.int64)
    m = 0
    for q in range(k):
        for i in range(n):
            s = 0
            for j in range(d):
                t = np.int64(vecs[i, j]) - xs[q, j]
                s += t * t
                if s > eps2:
                    break
            if s <= eps2:
                if m == cap:
                    cap *= 2
                    qn = np.empty(cap, dtype=np.int64)
                    vn = np.empty(cap, dtype=np.int64)
                    qn[:m] = qo[:m]
                    vn[:m] = vo[:m]
                    qo, vo = qn, vn
                qo[m] = q
                vo[m] = i
                m += 1
    return qo[:m].copy(), vo[:m].copy()


class PairWriterMulti(PairWriter):
    def add_many(self, x_ids, positions):
        if len(positions) == 0:
            return
        self.buf_x.append(np.asarray(x_ids, dtype=np.int32))
        self.buf_p.append(np.asarray(positions, dtype=np.int32))
        self.n_buf += len(positions)
        if self.n_buf >= self.flush_rows:
            self.flush()


@njit(cache=True, nogil=True)
def exact_pairs_idx(vecs, rows, X, xids, eps2):
    """ok[i] = ||vecs[rows[i]] - X[xids[i]]||^2 <= eps2 (all uint8 inputs, integer eps2)."""
    k = rows.shape[0]
    d = vecs.shape[1]
    ok = np.zeros(k, dtype=np.bool_)
    for i in range(k):
        r = rows[i]
        q = xids[i]
        s = 0
        for j in range(d):
            t = np.int64(vecs[r, j]) - np.int64(X[q, j])
            s += t * t
            if s > eps2:
                break
        ok[i] = s <= eps2
    return ok


class PartitionWriter:
    """Maybe pairs (x_id, position) split into files by bucket range, so phase 2 can
    read every bucket / page at most once while holding only one partition in RAM."""

    def __init__(self, dir_path, n_parts, n_buckets, starts, flush_rows=1_000_000):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n_parts = n_parts
        self.n_buckets = n_buckets
        self.starts = starts
        self.flush_rows = flush_rows
        self.buf = {}
        self.n_buf = 0
        self.n_pairs = 0
        for p in range(n_parts):
            open(self.path(p), "wb").close()

    def path(self, p):
        return self.dir / f"part_{p:04d}.bin"

    def add(self, x_id, positions):
        if len(positions) == 0:
            return
        b = np.searchsorted(self.starts, positions, side="right") - 1
        parts = b * self.n_parts // self.n_buckets
        for p in np.unique(parts):
            sel = positions[parts == p]
            self.buf.setdefault(int(p), []).append(
                np.stack([np.full(len(sel), x_id, dtype=np.int32), sel.astype(np.int32)], axis=1))
            self.n_buf += len(sel)
        if self.n_buf >= self.flush_rows:
            self.flush()

    def flush(self):
        for p, chunks in self.buf.items():
            with open(self.path(p), "ab") as f:
                np.concatenate(chunks).tofile(f)
        self.n_pairs += self.n_buf
        self.buf, self.n_buf = {}, 0

    def close(self):
        self.flush()
        return self.n_pairs


class RecordReader:
    """Sequentially serves records (int32 pairs) from a file in fixed-size chunks."""

    def __init__(self, path, chunk_rows=1_000_000):
        self.f = open(path, "rb")
        self.chunk_rows = chunk_rows
        self.buf = np.zeros((0, 2), dtype=np.int32)
        self.pos = 0

    def take(self, n):
        out = []
        while n > 0:
            if self.pos >= len(self.buf):
                self.buf = np.fromfile(self.f, dtype=np.int32, count=2 * self.chunk_rows).reshape(-1, 2)
                self.pos = 0
                if len(self.buf) == 0:
                    raise EOFError("record stream ended early")
            k = min(n, len(self.buf) - self.pos)
            out.append(self.buf[self.pos:self.pos + k])
            self.pos += k
            n -= k
        return out[0] if len(out) == 1 else np.concatenate(out)

    def close(self):
        self.f.close()
