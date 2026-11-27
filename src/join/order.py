"""Step 11: query processing order (the SimJoin idea, used for the streaming setting).

MST order:
  1. exact k-NN graph over the queries X (k = ORDER_KNN, FAISS), made symmetric
  2. a virtual root joined to every query with a weight larger than any edge, so the
     graph is connected even if the k-NN graph is not
  3. minimum spanning tree, then depth-first order from the root
  Consecutive queries are then close, so their candidate buckets overlap and the
  cache gets more hits. The order is known before the join starts, so Belady can
  use it.

Also writes id and random orders, and reports for each order:
  - mean distance between consecutive queries
  - mean overlap (Jaccard) of consecutive queries' candidate buckets at the mid eps

Outputs: data/orders/<dataset>/{mst,random,id}_n{N}.npy (permutations of 0..N-1)
Run: python -m scripts.step11_order --dataset dev
"""
import json
import math
import time
from pathlib import Path

import faiss
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import depth_first_order, minimum_spanning_tree


def mst_order(X_u8, k=20):
    n = X_u8.shape[0]
    xf = X_u8.astype(np.float32)
    index = faiss.IndexFlatL2(xf.shape[1])
    index.add(xf)
    kk = min(k + 1, n)
    _, nn = index.search(xf, kk)
    rows = np.repeat(np.arange(n), kk)
    cols = nn.ravel()
    keep = (cols >= 0) & (cols != rows)
    rows, cols = rows[keep], cols[keep]
    # exact distances in float64 (integer d^2 for uint8 data)
    diff = X_u8[rows].astype(np.int64) - X_u8[cols].astype(np.int64)
    w = np.sqrt(np.einsum("ij,ij->i", diff, diff).astype(np.float64))
    w = np.maximum(w, 1e-9)  # zero weights would be dropped by scipy (duplicate vectors)
    big = float(w.max()) * 10 + 1.0
    root = n
    r_rows = np.full(n, root)
    r_cols = np.arange(n)
    all_r = np.concatenate([rows, cols, r_rows, r_cols])
    all_c = np.concatenate([cols, rows, r_cols, r_rows])
    all_w = np.concatenate([w, w, np.full(n, big), np.full(n, big)])
    g = coo_matrix((all_w, (all_r, all_c)), shape=(n + 1, n + 1)).tocsr()
    g.sum_duplicates()
    t = minimum_spanning_tree(g)
    t = t + t.T
    order, _ = depth_first_order(t, root, directed=False, return_predecessors=True)
    order = order[order != root]
    assert len(order) == n and np.array_equal(np.sort(order), np.arange(n))
    n_components = int((t[root].toarray().ravel() >= big * 0.999).sum())
    return order.astype(np.int64), n_components


def consecutive_stats(X_u8, order, cand_fn=None):
    xs = X_u8[order].astype(np.int64)
    dist = np.sqrt(((xs[1:] - xs[:-1]) ** 2).sum(1))
    out = {"mean_consecutive_dist": float(dist.mean())}
    if cand_fn is not None:
        prev = cand_fn(int(order[0]))
        jac = []
        for q in order[1:]:
            cur = cand_fn(int(q))
            inter = np.intersect1d(prev, cur, assume_unique=True).size
            union = len(prev) + len(cur) - inter
            jac.append(inter / union if union else 1.0)
            prev = cur
        out["mean_consecutive_bucket_jaccard"] = float(np.mean(jac))
    return out


def run(cfg, dataset="dev", n_queries=None, k=None):
    from src.buckets.prune import Pruner
    from src.data.io import read_u8bin

    k = k or getattr(cfg, "ORDER_KNN", 20)
    data_dir = Path(cfg.DATA_DIR)
    out_dir = data_dir / "orders" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, X = read_u8bin(cfg.X_DEV_BIN if dataset == "dev" else cfg.X_BIN)
    if n_queries:
        X = np.ascontiguousarray(X[:n_queries])
    n = X.shape[0]

    t0 = time.time()
    mst, n_comp = mst_order(X, k)
    t_mst = time.time() - t0
    orders = {"id": np.arange(n, dtype=np.int64),
              "random": np.random.default_rng(cfg.RANDOM_SEED + 11).permutation(n).astype(np.int64),
              "mst": mst}
    for name, o in orders.items():
        np.save(out_dir / f"{name}_n{n}.npy", o)

    # quality: consecutive distance and candidate-bucket overlap at mid eps
    bdir = data_dir / "buckets" / dataset
    cand_fn = None
    if (bdir / "centers.npy").exists():
        with open(cfg.EPS_JSON) as f:
            eps = math.sqrt(int(json.load(f)[dataset]["mid"]["eps2"]))
        pr = Pruner(X, np.load(bdir / "centers.npy"), np.load(bdir / "radius.npy"), eps)
        cand_fn = lambda q: pr.candidates(q)[0]

    print(f"Query orders for {dataset}: {n:,} queries, k-NN graph k={k}, MST built in {t_mst:.1f} s "
          f"({n_comp} k-NN graph component(s) joined through the virtual root)")
    report = {"dataset": dataset, "n": n, "k": k, "mst_seconds": t_mst, "components": n_comp, "orders": {}}
    print(f"  {'order':>7} {'mean dist to previous query':>28} {'bucket overlap with previous (mid)':>36}")
    for name, o in orders.items():
        s = consecutive_stats(X, o, cand_fn)
        report["orders"][name] = s
        jac = s.get("mean_consecutive_bucket_jaccard")
        print(f"  {name:>7} {s['mean_consecutive_dist']:>28.1f} {('%.3f' % jac) if jac is not None else 'n/a':>36}")
    with open(out_dir / f"orders_n{n}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  saved to {out_dir}  (use --order mst / random / id in step10_b1 and step13_b3)")
    return report
