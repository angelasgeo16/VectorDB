"""Step 7: bucketize Y and write it to disk in bucket order.

For each dataset (dev: Y_dev 1M, full: Y 10M):
  1. Sample n_centers rows of Y as bucket centers (seeded).
  2. Assign every y to its nearest center (FAISS), then recompute the EXACT
     squared distance to that center (uint8 -> integers).
  3. Radii per bucket (all distances are exact):
        radius   = max member distance                 (exact pruning)
        radius_q = q-quantile member distance, q in BUCKET_QUANTILES
        outside_q = number of members farther than radius_q
                    (<= (1 - q) * |b|; used for the recall bound in Step 12)
  4. Sort Y by (bucket, distance to center) and write it as a RAW file with
     no header: sorted position p is at byte offset p * D, so 32 vectors fill
     each 4 KB page exactly and no vector crosses a page boundary.
  5. Reorder the PQ codes and residuals of every M into the same order, so a
     bucket's codes are one contiguous slice in RAM.

Outputs in data/buckets/<dataset>/:
    Y_sorted.bin                  raw uint8, (n, D), bucket order
    ids.npy                       (n,) int32   original Y row id of each sorted position
    centers.npy                   (B, D) uint8 center vectors
    center_ids.npy                (B,) int64   original row ids of the centers
    starts.npy, counts.npy        (B,) int64   bucket b = sorted positions [start, start+count)
    radius.npy, radius2.npy       (B,) float64 / int64
    radius_q95.npy, outside_q95.npy, radius_q99.npy, outside_q99.npy
    codes_M{M}.npy, residuals_M{M}.npy   PQ data in bucket order
    meta.json

Run: python -m scripts.step7_bucketize            (dev, then full)
"""
import json
import math
import time
from pathlib import Path

import faiss
import numpy as np

HEADER_BYTES = 8
KEY_SHIFT = 1 << 24  # > max squared distance for 128-d uint8 (128 * 255^2 = 8,323,200)


def open_y(path, n, d):
    return np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER_BYTES, shape=(n, d))


def exact_d2(rows_u8, centers_u8, assign):
    diff = rows_u8.astype(np.int32) - centers_u8[assign].astype(np.int32)
    return np.einsum("ij,ij->i", diff, diff).astype(np.int64)


def q_tag(q):
    return f"q{int(round(q * 100))}"


def bucketize(name, y_path, n, d, n_centers, quantiles, seed, chunk, pq_dir, m_values, out_dir):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = open_y(y_path, n, d)
    print(f"\n== {name}: {n:,} vectors -> {n_centers:,} buckets ==", flush=True)

    # 1. centers
    rng = np.random.default_rng(seed + 7)
    center_ids = np.sort(rng.choice(n, size=n_centers, replace=False)).astype(np.int64)
    centers = np.ascontiguousarray(y[center_ids])
    index = faiss.IndexFlatL2(d)
    index.add(centers.astype(np.float32))

    # 2. assignment + exact distances
    assign = np.empty(n, dtype=np.int32)
    d2 = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rows = np.asarray(y[s:e])
        _, nn = index.search(rows.astype(np.float32), 1)
        a = nn[:, 0].astype(np.int32)
        assign[s:e] = a
        d2[s:e] = exact_d2(rows, centers, a)
        print(f"    assigned {e:,} / {n:,}  ({time.time() - t0:.0f} s)", flush=True)
    t_assign = time.time() - t0

    # 3. order by (bucket, distance), bucket bounds and radii
    key = assign.astype(np.int64) * KEY_SHIFT + d2
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    d2_sorted = d2[order]
    del key

    counts = np.bincount(assign, minlength=n_centers).astype(np.int64)
    starts = np.zeros(n_centers, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    nonempty = counts > 0

    radius2 = np.zeros(n_centers, dtype=np.int64)
    radius2[nonempty] = d2_sorted[starts[nonempty] + counts[nonempty] - 1]

    quant = {}
    for q in quantiles:
        k = np.ceil(q * counts - 1e-9).astype(np.int64)
        rq2 = np.zeros(n_centers, dtype=np.int64)
        rq2[nonempty] = d2_sorted[starts[nonempty] + k[nonempty] - 1]
        b = np.arange(n_centers, dtype=np.int64)
        within = np.searchsorted(key_sorted, b * KEY_SHIFT + rq2, side="right") - starts
        outside = np.where(nonempty, counts - within, 0)
        assert np.all(outside <= np.floor((1 - q) * counts + 1e-9)), q
        quant[q] = (rq2, outside)

    ids = order.astype(np.int32)

    # 4. write Y in bucket order (raw, no header)
    t1 = time.time()
    with open(out_dir / "Y_sorted.bin", "wb") as f:
        for s in range(0, n, chunk):
            idx = order[s:s + chunk]
            srt = np.argsort(idx)
            rows = np.empty((len(idx), d), dtype=np.uint8)
            rows[srt] = np.asarray(y[idx[srt]])
            f.write(rows.tobytes())
    t_write = time.time() - t1
    print(f"    wrote Y_sorted.bin ({n * d / 1e6:.0f} MB) in {t_write:.0f} s", flush=True)

    # 5. reorder PQ codes and residuals
    for m in m_values:
        src = Path(pq_dir) / f"M{m}"
        codes = np.asarray(np.load(src / "codes.npy", mmap_mode="r")[:n])
        resid = np.asarray(np.load(src / "residuals.npy", mmap_mode="r")[:n])
        np.save(out_dir / f"codes_M{m}.npy", np.ascontiguousarray(codes[order]))
        np.save(out_dir / f"residuals_M{m}.npy", np.ascontiguousarray(resid[order]))
        del codes, resid
    print(f"    reordered PQ codes for M = {list(m_values)}", flush=True)

    # save arrays
    np.save(out_dir / "ids.npy", ids)
    np.save(out_dir / "centers.npy", centers)
    np.save(out_dir / "center_ids.npy", center_ids)
    np.save(out_dir / "starts.npy", starts)
    np.save(out_dir / "counts.npy", counts)
    np.save(out_dir / "radius2.npy", radius2)
    np.save(out_dir / "radius.npy", np.sqrt(radius2.astype(np.float64)))
    for q, (rq2, outside) in quant.items():
        np.save(out_dir / f"radius_{q_tag(q)}.npy", np.sqrt(rq2.astype(np.float64)))
        np.save(out_dir / f"radius2_{q_tag(q)}.npy", rq2)
        np.save(out_dir / f"outside_{q_tag(q)}.npy", outside.astype(np.int64))

    # checks
    check_buckets(out_dir, y, d, m_values, pq_dir, n, seed)

    r = np.sqrt(radius2[nonempty].astype(np.float64))
    c = counts[nonempty]
    meta = {
        "dataset": name,
        "n": int(n), "d": int(d), "n_buckets": int(n_centers),
        "n_empty_buckets": int((~nonempty).sum()),
        "layout": "Y_sorted.bin is raw uint8 with no header; sorted position p at byte p*D",
        "bucket_size": {"min": int(c.min()), "median": float(np.median(c)), "mean": float(c.mean()),
                        "p99": float(np.percentile(c, 99)), "max": int(c.max())},
        "radius": {"median": float(np.median(r)), "p95": float(np.percentile(r, 95)), "max": float(r.max())},
        "quantiles": list(quantiles),
        "outside": {q_tag(q): int(quant[q][1].sum()) for q in quantiles},
        "time_s": {"assign": t_assign, "write": t_write, "total": time.time() - t0},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    bucket size: min {meta['bucket_size']['min']}, median {meta['bucket_size']['median']:.0f}, "
          f"max {meta['bucket_size']['max']:,}, empty {meta['n_empty_buckets']}", flush=True)
    print(f"    radius: median {meta['radius']['median']:.1f}, p95 {meta['radius']['p95']:.1f}, "
          f"max {meta['radius']['max']:.1f}", flush=True)
    print(f"    done in {meta['time_s']['total']:.0f} s -> {out_dir}", flush=True)
    return meta


def check_buckets(out_dir, y, d, m_values, pq_dir, n, seed, n_check=50):
    out_dir = Path(out_dir)
    ids = np.load(out_dir / "ids.npy")
    starts = np.load(out_dir / "starts.npy")
    counts = np.load(out_dir / "counts.npy")
    centers = np.load(out_dir / "centers.npy")
    radius2 = np.load(out_dir / "radius2.npy")
    assert counts.sum() == n
    assert np.array_equal(np.sort(ids), np.arange(n, dtype=np.int32)), "ids is not a permutation"
    ys = np.memmap(out_dir / "Y_sorted.bin", dtype=np.uint8, mode="r", shape=(n, d))
    codes = {m: np.load(out_dir / f"codes_M{m}.npy", mmap_mode="r") for m in m_values}
    codes_src = {m: np.load(Path(pq_dir) / f"M{m}" / "codes.npy", mmap_mode="r") for m in m_values}

    rng = np.random.default_rng(seed + 77)
    nonempty = np.flatnonzero(counts)
    for b in rng.choice(nonempty, size=min(n_check, len(nonempty)), replace=False):
        s, c = starts[b], counts[b]
        orig = ids[s:s + c]
        rows = np.asarray(ys[s:s + c])
        assert np.array_equal(rows, np.asarray(y[np.sort(orig)])[np.argsort(np.argsort(orig))]), b
        dd = exact_d2(rows, centers, np.full(c, b))
        assert dd.max() == radius2[b] and np.all(np.diff(dd) >= 0), b
        for m in m_values:
            assert np.array_equal(np.asarray(codes[m][s:s + c]), np.asarray(codes_src[m][np.sort(orig)])[
                np.argsort(np.argsort(orig))]), (b, m)
    print(f"    check: {min(n_check, len(nonempty))} random buckets match Y, radii and PQ codes: OK",
          flush=True)


def run(cfg, datasets=("dev", "full")):
    n_centers = getattr(cfg, "BUCKET_N_CENTERS", {"dev": 1_000, "full": 10_000})
    quantiles = tuple(getattr(cfg, "BUCKET_QUANTILES", (0.95, 0.99)))
    chunk = getattr(cfg, "BUCKET_CHUNK", 1_000_000)
    data_dir = Path(cfg.DATA_DIR)
    spec = {"dev": (cfg.Y_DEV_BIN, cfg.N_Y_DEV), "full": (cfg.Y_BIN, cfg.N_Y_FULL)}
    out = {}
    for name in datasets:
        y_path, n = spec[name]
        out[name] = bucketize(name, y_path, n, cfg.D, n_centers[name], quantiles, cfg.RANDOM_SEED, chunk,
                              data_dir / "pq", cfg.PQ_M_VALUES, data_dir / "buckets" / name)
    return out
