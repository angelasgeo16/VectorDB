"""Step 10: baseline B1 (DiskJoin-style buckets, no PQ codes).

Two loop orders (team decision, Nov 25: bucket-at-a-time is the main design):

loop="bucket"  (default, main design)
    for each bucket b (in file order):
        Q_b = queries with d(x, c_b) - R_b <= eps     (exact pruning, all X in RAM)
        if Q_b is empty: skip, no read
        read b once from SSD, check every x in Q_b exactly, write matches
    Every bucket is read at most once (<= 1x Y). No cache, query order irrelevant.

loop="query"   (ablation only: shows why we switched)
    for each query x (in the given order): for each candidate bucket: get it from a
    Belady/LRU cache (SSD on miss) and check x. Buckets needed by many queries get
    re-read whenever the cache has dropped them.

Peak RSS is compared with MEMORY_CAP_MB. Recall is checked against the ground
truth after the timed part; the pairs file is then deleted unless keep_pairs=True.

Run: python -m scripts.step10_b1 --dataset dev --level mid
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from src.buckets.prune import candidates_exact, queries_per_bucket
from src.cache.belady import BeladyCache, LRUCache
from src.diskio.reader import BucketReader
from src.join.common import (PairWriterMulti, append_run, exact_matches, exact_matches_multi,
                             load_order, peak_rss_mb, recall_check)


def _setup(cfg, dataset, level, n_queries=None):
    from src.data.io import read_u8bin

    data_dir = Path(cfg.DATA_DIR)
    bdir = data_dir / "buckets" / dataset
    with open(cfg.EPS_JSON) as f:
        eps2 = int(json.load(f)[dataset][level]["eps2"])
    meta = {
        "data_dir": data_dir, "bdir": bdir, "dataset": dataset,
        "results_dir": Path(getattr(cfg, "RESULTS_DIR", data_dir.parent / "results")),
        "cap_mb": getattr(cfg, "MEMORY_CAP_MB", 512),
        "eps2": eps2, "eps": math.sqrt(eps2),
        "centers": np.load(bdir / "centers.npy"),
        "radius": np.load(bdir / "radius.npy"),
        "starts": np.load(bdir / "starts.npy"),
        "counts": np.load(bdir / "counts.npy"),
    }
    meta["n_y"] = int(meta["counts"].sum())
    _, _, X = read_u8bin(cfg.X_DEV_BIN if dataset == "dev" else cfg.X_BIN)
    if n_queries:
        X = np.ascontiguousarray(X[:n_queries])
    meta["X"] = X
    return meta


def _loop_bucket(cfg, m, reader, writer):
    """Main design: read each needed bucket once and check all its candidate queries."""
    X = m["X"]
    starts, counts = m["starts"], m["counts"]
    t = {"plan": 0.0, "io": 0.0, "verify": 0.0}
    n_cand = 0          # (x, bucket) candidate pairs
    n_vec_pairs = 0     # (x, y) exact distance checks
    n_buckets_read = 0
    n_b = len(counts)
    report = max(1, n_b // 10)

    gen = queries_per_bucket(X, m["centers"], m["radius"], m["eps"])
    t_start = time.perf_counter()
    last = 0
    while True:
        t0 = time.perf_counter()
        try:
            b, qs = next(gen)
        except StopIteration:
            t["plan"] += time.perf_counter() - t0
            break
        t["plan"] += time.perf_counter() - t0
        if counts[b] == 0:
            continue
        t0 = time.perf_counter()
        vecs = reader.read_bucket(b)
        t["io"] += time.perf_counter() - t0
        n_buckets_read += 1
        n_cand += len(qs)
        n_vec_pairs += len(qs) * int(counts[b])

        t0 = time.perf_counter()
        qi, vi = exact_matches_multi(vecs, X[qs].astype(np.int64), m["eps2"])
        t["verify"] += time.perf_counter() - t0
        writer.add_many(qs[qi], vi + starts[b])

        if b // report > last:
            last = b // report
            print(f"    bucket {b + 1:,} / {n_b:,}  ({time.perf_counter() - t_start:.0f} s, "
                  f"{reader.bytes_read / 1e6:,.0f} MB read)", flush=True)
    return t, {"n_candidates": n_cand, "n_vec_pairs": n_vec_pairs, "n_buckets_read": n_buckets_read,
               "cache_hit_rate": ""}


def _loop_query(cfg, m, reader, writer, order, cache_mb, cache_policy):
    """Ablation: query at a time through a byte-limited cache."""
    X = m["X"]
    starts, counts = m["starts"], m["counts"]
    n_x = X.shape[0]
    q_order = load_order(order, n_x, cfg.RANDOM_SEED, m["data_dir"] / "orders" / m["dataset"])
    t = {"plan": 0.0, "io": 0.0, "verify": 0.0}

    t0 = time.perf_counter()
    ptr, seq = candidates_exact(X[q_order], m["centers"], m["radius"], m["eps"])
    t["plan"] = time.perf_counter() - t0

    def loader(b):
        t1 = time.perf_counter()
        v = reader.read_bucket(b)
        t["io"] += time.perf_counter() - t1
        return v

    cache = (BeladyCache if cache_policy == "belady" else LRUCache)(int(cache_mb * 1e6), loader, seq)
    report = max(1, n_x // 10)
    t_start = time.perf_counter()
    for k in range(n_x):
        x_id = int(q_order[k])
        x = X[x_id].astype(np.int64)
        for i in range(ptr[k], ptr[k + 1]):
            b = int(seq[i])
            vecs = cache.get(i, b)
            t1 = time.perf_counter()
            hit = exact_matches(vecs, x, m["eps2"])
            t["verify"] += time.perf_counter() - t1
            if len(hit):
                writer.add_many(np.full(len(hit), x_id), hit + starts[b])
        if (k + 1) % report == 0:
            print(f"    {k + 1:,} / {n_x:,} queries  ({time.perf_counter() - t_start:.0f} s, "
                  f"{reader.bytes_read / 1e6:,.0f} MB read)", flush=True)
    return t, {"n_candidates": len(seq), "n_vec_pairs": int(counts[seq].sum()),
               "n_buckets_read": cache.misses, "cache_hit_rate": cache.stats()["cache_hit_rate"]}


def run_b1(cfg, dataset="dev", level="mid", loop="bucket", order="id", cache_mb=None,
           cache_policy="belady", keep_pairs=False, n_queries=None):
    m = _setup(cfg, dataset, level, n_queries)
    X, counts, n_y = m["X"], m["counts"], m["n_y"]
    n_x = X.shape[0]
    if loop == "query" and cache_mb is None:
        cache_mb = getattr(cfg, "B1_CACHE_MB", {"dev": 30, "full": 300})[dataset]
    order_name = "bucket-major" if loop == "bucket" else (
        order if order in ("id", "random") else Path(order).stem)
    run_name = f"B1_{dataset}_{level}_{order_name}" + (f"_{cache_policy}_{cache_mb}MB" if loop == "query" else "") \
        + (f"_n{n_x}" if n_queries else "")

    desc = "bucket at a time" if loop == "bucket" else f"query at a time, {order_name}, {cache_mb} MB {cache_policy}"
    print(f"B1 | {dataset} {level} (eps={m['eps']:.2f}) | {n_x:,} x vs {n_y:,} y | {desc}", flush=True)

    reader = BucketReader(m["bdir"] / "Y_sorted.bin", m["starts"], counts, d=cfg.D)
    writer = PairWriterMulti(m["results_dir"] / "pairs" / f"{run_name}.bin")
    t_start = time.perf_counter()
    if loop == "bucket":
        t, s = _loop_bucket(cfg, m, reader, writer)
    else:
        t, s = _loop_query(cfg, m, reader, writer, order, cache_mb, cache_policy)
    n_pairs = writer.close()
    t_total = time.perf_counter() - t_start
    rss = peak_rss_mb()
    io = reader.counters()
    reader.close()

    gt = m["data_dir"] / "ground_truth" / dataset / level / "pairs.npy"
    recall, precision, _, n_true = recall_check(writer.path, m["bdir"] / "ids.npy", gt, n_y,
                                                n_x if n_queries else None)
    if not keep_pairs:
        writer.path.unlink()

    row = {
        "system": "B1", "dataset": dataset, "level": level, "eps2": m["eps2"], "M": "", "pruning": "exact",
        "q": "", "order": order_name, "read_policy": "bucket",
        "cache_MB": cache_mb if loop == "query" else "", "cache_policy": cache_policy if loop == "query" else "",
        "n_x": n_x, "n_candidates": s["n_candidates"], "n_pairs": n_pairs, "recall": recall,
        "precision": precision, "proven_recall": 1.0, "maybe_frac": "",
        "bytes_read_MB": io["bytes_read"] / 1e6, "device_MB": io["device_bytes"] / 1e6,
        "n_requests": io["n_requests"], "cache_hit_rate": s["cache_hit_rate"],
        "plan_s": t["plan"], "filter_s": 0.0, "io_s": t["io"], "verify_s": t["verify"], "total_s": t_total,
        "peak_rss_MB": rss, "over_cap": rss > m["cap_mb"],
    }
    path = append_run(m["results_dir"], row)

    n_b = len(counts)
    print(f"    pruning: {s['n_candidates'] / n_x:,.1f} candidate buckets per x (of {n_b:,}), "
          f"{s['n_vec_pairs'] / n_x:,.0f} vectors checked per x ({s['n_vec_pairs'] / (n_x * n_y):.1%} of all pairs)")
    print(f"    pairs: {n_pairs:,} (ground truth {n_true:,})  recall {recall:.6f}  precision {precision:.6f}")
    print(f"    disk: {io['bytes_read'] / 1e6:,.1f} MB read in {io['n_requests']:,} requests "
          f"({io['bytes_read'] / (n_y * cfg.D):.2f}x the size of Y)"
          + (f"; cache hit rate {s['cache_hit_rate']:.3f}" if loop == "query" else
             f"; {s['n_buckets_read']:,} of {n_b:,} buckets read"))
    print(f"    time: plan {t['plan']:.1f} s, io {t['io']:.1f} s, verify {t['verify']:.1f} s, total {t_total:.1f} s")
    print(f"    peak RSS {rss:.0f} MB (cap {m['cap_mb']} MB){'  *** OVER CAP ***' if rss > m['cap_mb'] else ''}")
    print(f"    appended to {path}")
    if recall < 1.0 or precision < 1.0:
        raise AssertionError("B1 must find exactly the ground-truth pairs")
    return row
