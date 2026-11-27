"""Step 13: B3 (ours) and B2 (PQ only), both loop settings.

Phase 1, filter (in RAM, same for both loops):
    for each query x (in order):
        candidate buckets  = exact or quantile-radius pruning (Pruner)
        scan their PQ codes with x's lookup table (Numba kernel)
        Yes   -> written as matches (no disk read)
        Maybe -> kept for phase 2
        (B2: residuals treated as 0, so there is no Maybe: accept iff d(x, y_hat) <= eps)

Phase 2, verify (reads from SSD, checks exactly):
    loop="bucket"  (batch: all queries known, X in RAM)
        Maybe pairs are written to partition files by bucket range; each partition
        is loaded, sorted by position, and every needed bucket / page is read ONCE.
    loop="query"   (streaming: queries handled one at a time, in a given order)
        Maybe pairs are kept in query order; for each (x, bucket) group the bucket
        comes from a Belady / LRU cache (read_policy="bucket") or only its pages are
        read (read_policy="page", no cache). Mirrors B1's query loop.

Memory: PQ codes + residuals of the chosen M are in RAM. Cache default (query loop,
bucket reads) = B1's cache minus the codes + residuals, so B1 and B3 get the same
total memory budget.

Run: python -m scripts.step13_b3 --dataset dev --level mid --M 16
"""
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from src.buckets.prune import Pruner
from src.cache.belady import BeladyCache, LRUCache
from src.diskio.reader import BucketReader
from src.filter.kernel import build_lut, filter_query
from src.join.common import (PairWriterMulti, PartitionWriter, RecordReader, append_run, exact_pairs_idx,
                             load_order, peak_rss_mb, recall_check)

LEVELS = ("low", "mid", "high")


def run_b3(cfg, dataset="dev", level="mid", M=16, pruning="exact", loop="bucket", read_policy="bucket",
           order="id", cache_mb=None, cache_policy="belady", system="B3", n_queries=None,
           n_parts=256, keep_pairs=False):
    from src.data.io import read_u8bin

    assert system in ("B3", "B2") and pruning in ("exact", "q95", "q99")
    assert loop in ("bucket", "query") and read_policy in ("bucket", "page")
    data_dir = Path(cfg.DATA_DIR)
    results_dir = Path(getattr(cfg, "RESULTS_DIR", data_dir.parent / "results"))
    bdir = data_dir / "buckets" / dataset
    cap_mb = getattr(cfg, "MEMORY_CAP_MB", 512)

    with open(cfg.EPS_JSON) as f:
        eps2 = int(json.load(f)[dataset][level]["eps2"])
    eps = math.sqrt(eps2)

    centers = np.load(bdir / "centers.npy")
    radius = np.load(bdir / "radius.npy")
    starts = np.load(bdir / "starts.npy")
    counts = np.load(bdir / "counts.npy")
    n_b, n_y = len(counts), int(counts.sum())
    radius_q = outside_q = None
    if pruning != "exact":
        radius_q = np.load(bdir / f"radius_{pruning}.npy")
        outside_q = np.load(bdir / f"outside_{pruning}.npy")

    _, _, X = read_u8bin(cfg.X_DEV_BIN if dataset == "dev" else cfg.X_BIN)
    if n_queries:
        X = np.ascontiguousarray(X[:n_queries])
    n_x = X.shape[0]

    codes = np.ascontiguousarray(np.load(bdir / f"codes_M{M}.npy"))
    if system == "B3":
        rbits = np.ascontiguousarray(np.load(bdir / f"residuals_M{M}.npy")).view(np.uint16)
    else:
        rbits = np.zeros(n_y, dtype=np.uint16)  # B2: no residual -> no Maybe
    centroids = np.load(data_dir / "pq" / f"M{M}" / "centroids.npy")
    codes_mb = (codes.nbytes + rbits.nbytes) / 1e6

    q_order = load_order(order, n_x, cfg.RANDOM_SEED, data_dir / "orders" / dataset) if loop == "query" \
        else np.arange(n_x)
    order_name = "bucket-major" if loop == "bucket" else (
        order if order in ("id", "random") else Path(order).stem)
    use_cache = system == "B3" and loop == "query" and read_policy == "bucket"
    if use_cache and cache_mb is None:
        b1_cache = getattr(cfg, "B1_CACHE_MB", {"dev": 30, "full": 300})[dataset]
        cache_mb = max(round(b1_cache - codes_mb, 1), b1_cache * 0.05)

    run_name = f"{system}_{dataset}_{level}_M{M}_{pruning}_{order_name}_{read_policy}" + (
        f"_{cache_policy}_{cache_mb}MB" if use_cache else "") + (f"_n{n_x}" if n_queries else "")
    tmp_dir = results_dir / "tmp" / run_name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print(f"{system} | {dataset} {level} (eps={eps:.2f}) | {n_x:,} x vs {n_y:,} y | M={M} "
          f"({codes_mb:.0f} MB codes+residuals) | pruning {pruning} | "
          + ("batch (bucket at a time)" if loop == "bucket" else f"query at a time, order {order_name}")
          + (f" | reads: {read_policy}" if system == "B3" else "")
          + (f" | cache {cache_mb} MB {cache_policy}" if use_cache else ""), flush=True)

    t = {"plan": 0.0, "filter": 0.0, "io": 0.0, "verify": 0.0}
    writer = PairWriterMulti(tmp_dir / "pairs.bin", flush_rows=500_000)
    pruner = Pruner(X, centers, radius, eps, radius_q, outside_q)
    t_start = time.perf_counter()

    # ---------------- phase 1: filter in RAM ----------------
    if loop == "bucket":
        maybe_out = PartitionWriter(tmp_dir / "maybe", min(n_parts, n_b), n_b, starts, flush_rows=1_000_000)
    else:
        maybe_f = open(tmp_dir / "maybe_stream.bin", "wb")
        g_bucket, g_size, g_x = [], [], []
    n_cand = n_scanned = n_yes = n_maybe = 0
    U = 0
    report = max(1, n_x // 10)
    for k in range(n_x):
        x_id = int(q_order[k])
        t0 = time.perf_counter()
        cands, u = pruner.candidates(x_id)
        U += u
        t1 = time.perf_counter()
        lut = build_lut(X[x_id], centroids)
        yes, maybe = filter_query(lut, codes, rbits, starts, counts, cands, eps)
        t2 = time.perf_counter()
        t["plan"] += t1 - t0
        t["filter"] += t2 - t1
        n_cand += len(cands)
        n_scanned += int(counts[cands].sum())
        n_yes += len(yes)
        n_maybe += len(maybe)
        writer.add_many(np.full(len(yes), x_id), yes)
        if len(maybe):
            if loop == "bucket":
                maybe_out.add(x_id, maybe)
            else:
                np.stack([np.full(len(maybe), x_id, dtype=np.int32), maybe.astype(np.int32)], 1).tofile(maybe_f)
                b = np.searchsorted(starts, maybe, side="right") - 1
                ub, sizes = np.unique(b, return_counts=True)  # maybe is ascending, so b is too
                g_bucket.append(ub.astype(np.int32))
                g_size.append(sizes.astype(np.int32))
                g_x.append(np.full(len(ub), x_id, dtype=np.int32))
        if (k + 1) % report == 0:
            print(f"    filter {k + 1:,} / {n_x:,} queries  ({time.perf_counter() - t_start:.0f} s, "
                  f"{n_maybe:,} Maybe so far)", flush=True)
    if system == "B2":
        assert n_maybe == 0

    # ---------------- phase 2: verify from SSD ----------------
    reader = BucketReader(bdir / "Y_sorted.bin", starts, counts, d=cfg.D)
    n_verified = 0
    cache = None
    if loop == "bucket":
        maybe_out.close()
        for p in range(maybe_out.n_parts):
            arr = np.fromfile(maybe_out.path(p), dtype=np.int32).reshape(-1, 2)
            maybe_out.path(p).unlink()
            if len(arr) == 0:
                continue
            srt = np.argsort(arr[:, 1], kind="stable")
            xs, ps = arr[srt, 0], arr[srt, 1].astype(np.int64)
            del arr
            if read_policy == "bucket":
                bk = np.searchsorted(starts, ps, side="right") - 1
                cuts = np.flatnonzero(np.diff(bk)) + 1
                for g0, g1 in zip(np.r_[0, cuts], np.r_[cuts, len(ps)]):
                    b = int(bk[g0])
                    t0 = time.perf_counter()
                    vecs = reader.read_bucket(b)
                    t1 = time.perf_counter()
                    ok = exact_pairs_idx(vecs, ps[g0:g1] - starts[b], X, xs[g0:g1], eps2)
                    t["io"] += t1 - t0
                    t["verify"] += time.perf_counter() - t1
                    writer.add_many(xs[g0:g1][ok], ps[g0:g1][ok])
            else:
                up = np.unique(ps)
                t0 = time.perf_counter()
                vecs = reader.read_positions(up)
                t1 = time.perf_counter()
                ok = exact_pairs_idx(vecs, np.searchsorted(up, ps), X, xs, eps2)
                t["io"] += t1 - t0
                t["verify"] += time.perf_counter() - t1
                writer.add_many(xs[ok], ps[ok])
            n_verified += len(ps)
    else:
        maybe_f.close()
        seq = np.concatenate(g_bucket) if g_bucket else np.zeros(0, np.int32)
        gsz = np.concatenate(g_size) if g_size else np.zeros(0, np.int32)
        gx = np.concatenate(g_x) if g_x else np.zeros(0, np.int32)
        del g_bucket, g_size, g_x
        if read_policy == "bucket":
            def loader(b):
                t0 = time.perf_counter()
                v = reader.read_bucket(b)
                t["io"] += time.perf_counter() - t0
                return v
            cache = (BeladyCache if cache_policy == "belady" else LRUCache)(int(cache_mb * 1e6), loader, seq)
        stream = RecordReader(tmp_dir / "maybe_stream.bin")
        for i in range(len(seq)):
            b, x_id = int(seq[i]), int(gx[i])
            rec = stream.take(int(gsz[i]))
            ps = rec[:, 1].astype(np.int64)
            xs = rec[:, 0]
            if read_policy == "bucket":
                vecs = cache.get(i, b)
                rows = ps - starts[b]
            else:
                t0 = time.perf_counter()
                vecs = reader.read_positions(ps)
                t["io"] += time.perf_counter() - t0
                rows = np.arange(len(ps))
            t0 = time.perf_counter()
            ok = exact_pairs_idx(vecs, rows, X, xs, eps2)
            t["verify"] += time.perf_counter() - t0
            writer.add_many(xs[ok], ps[ok])
            n_verified += len(ps)
        stream.close()

    n_pairs = writer.close()
    t_total = time.perf_counter() - t_start
    rss = peak_rss_mb()
    io = reader.counters()
    reader.close()

    # ---------------- correctness (after the timed part) ----------------
    gt = data_dir / "ground_truth" / dataset / level / "pairs.npy"
    recall, precision, _, n_true = recall_check(writer.path, bdir / "ids.npy", gt, n_y,
                                                n_x if n_queries else None)
    proven = n_pairs / (n_pairs + U) if (n_pairs + U) else 1.0
    if pruning == "exact" and system == "B3":
        proven = 1.0
    if keep_pairs:
        dst = results_dir / "pairs" / f"{run_name}.bin"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(writer.path), dst)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    cs = cache.stats() if cache is not None else {}
    row = {
        "system": system, "dataset": dataset, "level": level, "eps2": eps2, "M": M, "pruning": pruning,
        "q": "" if pruning == "exact" else pruning[1:], "order": order_name,
        "read_policy": read_policy if system == "B3" else "none",
        "cache_MB": cache_mb if use_cache else "", "cache_policy": cache_policy if use_cache else "",
        "n_x": n_x, "n_candidates": n_cand, "n_pairs": n_pairs, "recall": recall, "precision": precision,
        "proven_recall": proven if system == "B3" else "", "maybe_frac": n_maybe / (n_x * n_y),
        "bytes_read_MB": io["bytes_read"] / 1e6, "device_MB": io["device_bytes"] / 1e6,
        "n_requests": io["n_requests"], "cache_hit_rate": cs.get("cache_hit_rate", ""),
        "plan_s": t["plan"], "filter_s": t["filter"], "io_s": t["io"], "verify_s": t["verify"],
        "total_s": t_total, "peak_rss_MB": rss, "over_cap": rss > cap_mb,
    }
    path = append_run(results_dir, row)

    print(f"    pruning: {n_cand / n_x:,.1f} candidate buckets per x (of {n_b:,}), "
          f"{n_scanned / n_x:,.0f} codes scanned per x ({n_scanned / (n_x * n_y):.1%} of all pairs)")
    print(f"    labels: {n_yes:,} Yes, {n_maybe:,} Maybe ({n_maybe / (n_x * n_y):.5%} of all pairs), "
          f"{n_verified:,} verified on disk")
    print(f"    pairs: {n_pairs:,} (ground truth {n_true:,})  recall {recall:.6f}  precision {precision:.6f}"
          + (f"  proven recall >= {proven:.6f}" if system == "B3" and pruning != "exact" else ""))
    print(f"    disk: {io['bytes_read'] / 1e6:,.1f} MB read in {io['n_requests']:,} requests "
          f"({io['bytes_read'] / (n_y * cfg.D):.3f}x the size of Y)"
          + (f"; cache hit rate {cs['cache_hit_rate']:.3f}" if cs else ""))
    print(f"    time: plan {t['plan']:.1f} s, filter {t['filter']:.1f} s, io {t['io']:.1f} s, "
          f"verify {t['verify']:.1f} s, total {t_total:.1f} s")
    print(f"    peak RSS {rss:.0f} MB (cap {cap_mb} MB){'  *** OVER CAP ***' if rss > cap_mb else ''}")
    print(f"    appended to {path}")
    if system == "B3" and pruning == "exact" and (recall < 1.0 or precision < 1.0):
        raise AssertionError("B3 with exact pruning must find exactly the ground-truth pairs")
    if system == "B3" and precision < 1.0:
        raise AssertionError("B3 must never output a non-match")
    if system == "B3" and recall + 1e-12 < proven:
        raise AssertionError(f"measured recall {recall} below the proven bound {proven}")
    return row
