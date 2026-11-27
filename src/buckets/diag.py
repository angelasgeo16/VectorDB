"""Pruning diagnostic (dev only, no disk reads of the join itself).

Question: can better buckets cut the share of (x, y) pairs that survive bucket
pruning (92% with 1,000 random-sample centers on dev)?

Grid:
    centers   : "random" (sampled Y rows, as in Step 7) or "kmeans" (FAISS k-means)
    n_buckets : e.g. 1,000 and 4,000
    radius    : max (exact pruning, recall 1.0), q99, q95 (probabilistic pruning)
For every dev eps level it reports
    pairs_kept      share of all (x, y) pairs whose bucket survives pruning
                    (= exact checks for B1, codes scanned for B3)
    buckets_per_x   average candidate buckets per query
    recall          share of ground-truth pairs whose bucket survives
                    (1.0 for max radius by construction; measured for q99 / q95)

Run: python -m scripts.diag_pruning
"""
import json
import math
import time
from pathlib import Path

import faiss
import numpy as np

HEADER_BYTES = 8
LEVELS = ("low", "mid", "high")


def dists_to(points_f64, centers_f64):
    p2 = (points_f64 * points_f64).sum(1)[:, None]
    c2 = (centers_f64 * centers_f64).sum(1)[None, :]
    return np.sqrt(np.maximum(p2 + c2 - 2.0 * (points_f64 @ centers_f64.T), 0.0))


def build(y_u8, kind, n_buckets, seed, train_n=200_000, niter=20):
    n, d = y_u8.shape
    rng = np.random.default_rng(seed + 7)
    if kind == "random":
        centers = y_u8[np.sort(rng.choice(n, n_buckets, replace=False))].astype(np.float32)
    else:
        sample = y_u8[np.sort(rng.choice(n, min(train_n, n), replace=False))].astype(np.float32)
        km = faiss.Kmeans(d, n_buckets, niter=niter, seed=seed, verbose=False)
        km.train(sample)
        centers = km.centroids.astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(centers)
    assign = np.empty(n, dtype=np.int64)
    dist = np.empty(n, dtype=np.float64)
    c64 = centers.astype(np.float64)
    for s in range(0, n, 200_000):
        e = min(s + 200_000, n)
        rows = y_u8[s:e]
        _, nn = index.search(rows.astype(np.float32), 1)
        a = nn[:, 0]
        assign[s:e] = a
        diff = rows.astype(np.float64) - c64[a]
        dist[s:e] = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    counts = np.bincount(assign, minlength=n_buckets)
    order = np.lexsort((dist, assign))
    ds = dist[order]
    starts = np.zeros(n_buckets, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    ne = counts > 0
    radii = {}
    for name, q in (("max", 1.0), ("q99", 0.99), ("q95", 0.95)):
        r = np.zeros(n_buckets)
        k = np.ceil(q * counts - 1e-9).astype(np.int64)
        r[ne] = ds[starts[ne] + k[ne] - 1]
        radii[name] = r
    return c64, assign, counts, radii


def run(cfg, n_buckets_list=(1_000, 4_000), kinds=("random", "kmeans")):
    from src.data.io import read_u8bin

    data_dir = Path(cfg.DATA_DIR)
    results_dir = Path(getattr(cfg, "RESULTS_DIR", data_dir.parent / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    n_y = cfg.N_Y_DEV
    y = np.asarray(np.memmap(cfg.Y_DEV_BIN, dtype=np.uint8, mode="r", offset=HEADER_BYTES, shape=(n_y, cfg.D)))
    _, _, X = read_u8bin(cfg.X_DEV_BIN)
    x64 = X.astype(np.float64)
    n_x = X.shape[0]
    with open(cfg.EPS_JSON) as f:
        eps = {l: math.sqrt(int(v["eps2"])) for l, v in json.load(f)["dev"].items()}
    gt = {l: np.load(data_dir / "ground_truth" / "dev" / l / "pairs.npy").astype(np.int64) for l in LEVELS}

    print(f"Pruning diagnostic on dev: {n_x:,} x vs {n_y:,} y; dev eps "
          + ", ".join(f"{l}={eps[l]:.1f}" for l in LEVELS))
    print(f"\n  {'centers':>7} {'buckets':>7} {'radius':>6} {'med R':>6} | "
          + " | ".join(f"{l:^27}" for l in LEVELS))
    print(f"  {'':>7} {'':>7} {'':>6} {'':>6} | "
          + " | ".join(f"{'pairs kept':>10} {'bkt/x':>6} {'recall':>8}" for _ in LEVELS))

    out = []
    for kind in kinds:
        for nb in n_buckets_list:
            t0 = time.time()
            c64, assign, counts, radii = build(y, kind, nb, cfg.RANDOM_SEED)
            dxc = dists_to(x64, c64)  # (n_x, nb)
            for rname, r in radii.items():
                cells = []
                row = {"centers": kind, "n_buckets": nb, "radius": rname,
                       "median_radius": float(np.median(r[counts > 0])),
                       "bucket_size_max": int(counts.max()), "levels": {}}
                for l in LEVELS:
                    keep = dxc <= (eps[l] + r + 1e-6)[None, :]
                    pairs_kept = float((keep * counts[None, :]).sum()) / (n_x * n_y)
                    bpx = float(keep.sum()) / n_x
                    g = gt[l]
                    rec = float(keep[g[:, 0], assign[g[:, 1]]].mean()) if len(g) else 1.0
                    row["levels"][l] = {"pairs_kept": pairs_kept, "buckets_per_x": bpx, "recall": rec}
                    cells.append(f"{pairs_kept:>10.1%} {bpx:>6.0f} {rec:>8.4f}")
                out.append(row)
                print(f"  {kind:>7} {nb:>7,} {rname:>6} {row['median_radius']:>6.0f} | " + " | ".join(cells),
                      flush=True)
            print(f"  {'':>7} ({kind} {nb:,} built and evaluated in {time.time() - t0:.0f} s)", flush=True)

    with open(results_dir / "pruning_diag.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {results_dir / 'pruning_diag.json'}")
    return out
