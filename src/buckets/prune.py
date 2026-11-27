"""Bucket pruning: which buckets can contain a match for each query.

Exact pruning:          bucket b is a candidate for x  iff  d(x, c_b) - R_b      <= eps
Probabilistic pruning:  same test with the quantile radius R_b(q), q = 0.95 / 0.99.
    A pair (x, b) kept by exact pruning but dropped by the quantile test can only
    lose members farther than R_b(q) from c_b, and there are outside_q[b] of them.
    Summing those over dropped (x, b) gives U, and  recall >= F / (F + U)
    where F = pairs found (the proven recall bound).

All distances between uint8 queries and uint8 centers are exact (integer d^2).
`slack` only ever adds candidates, never removes them.

Memory: query norms / float copies are computed once (Pruner), and per-block
work reuses a single matrix, so pruning adds only tens of MB of RSS.
"""
import numpy as np


class Pruner:
    def __init__(self, X_u8, centers_u8, radius, eps, radius_q=None, outside_q=None, slack=1e-9):
        self.x = X_u8.astype(np.float64)
        self.x2 = np.einsum("ij,ij->i", self.x, self.x)
        self.c = centers_u8.astype(np.float64)
        self.c2 = np.einsum("ij,ij->i", self.c, self.c)
        self.lim = eps + np.asarray(radius, dtype=np.float64) + slack
        self.prob = radius_q is not None
        if self.prob:
            self.lim_q = eps + np.asarray(radius_q, dtype=np.float64) + slack
            self.outside_q = np.asarray(outside_q, dtype=np.int64)

    def dist_query(self, k):
        """Exact distances from query k to all centers, (n_buckets,)."""
        d2 = self.c @ self.x[k]
        d2 *= -2.0
        d2 += self.x2[k]
        d2 += self.c2
        return np.sqrt(np.maximum(np.rint(d2), 0.0))

    def candidates(self, k):
        """(sorted candidate bucket ids, U) for query k."""
        d = self.dist_query(k)
        if not self.prob:
            return np.flatnonzero(d <= self.lim), 0
        keep = d <= self.lim_q
        dropped = (d <= self.lim) & ~keep
        return np.flatnonzero(keep), int(self.outside_q[dropped].sum())

    def dist_bucket_block(self, s, e):
        """Exact distances from all queries to centers s..e-1, (n_x, e - s)."""
        d2 = self.x @ self.c[s:e].T
        d2 *= -2.0
        d2 += self.x2[:, None]
        d2 += self.c2[None, s:e]
        np.rint(d2, out=d2)
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2, out=d2)


def center_distances(xq_u8, centers_u8):
    """Exact distances (n_x, B) float64 between uint8 queries and uint8 centers."""
    x = xq_u8.astype(np.float64)
    c = centers_u8.astype(np.float64)
    d2 = (x * x).sum(1)[:, None] + (c * c).sum(1)[None, :] - 2.0 * (x @ c.T)
    return np.sqrt(np.maximum(np.rint(d2), 0.0))


def candidates_exact(xq_u8, centers_u8, radius, eps, batch=256, slack=1e-9):
    """CSR candidate lists under exact pruning (query-at-a-time B1 ablation).
    ptr (n_x + 1,), buckets (total,) int32; query k's candidates = buckets[ptr[k]:ptr[k+1]]."""
    pr = Pruner(xq_u8, centers_u8, radius, eps, slack=slack)
    n_x = xq_u8.shape[0]
    ptr = np.zeros(n_x + 1, dtype=np.int64)
    parts = []
    for k in range(n_x):
        cb, _ = pr.candidates(k)
        parts.append(cb.astype(np.int32))
        ptr[k + 1] = ptr[k] + len(cb)
    buckets = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    return ptr, buckets


def queries_per_bucket(xq_u8, centers_u8, radius, eps, block=128, slack=1e-9):
    """Bucket-at-a-time pruning. Yields (b, query_ids) for every bucket that at least
    one query can match, in bucket id (= file) order."""
    pr = Pruner(xq_u8, centers_u8, radius, eps, slack=slack)
    n_b = centers_u8.shape[0]
    for s in range(0, n_b, block):
        e = min(s + block, n_b)
        keep = pr.dist_bucket_block(s, e) <= pr.lim[None, s:e]
        for j in range(e - s):
            q = np.flatnonzero(keep[:, j])
            if len(q):
                yield s + j, q
