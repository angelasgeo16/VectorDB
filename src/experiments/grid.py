"""Step 15: the experiment grid.

Each run is one call of an existing script (step10_b1 / step13_b3) in its own
process, so peak RSS is measured per run. Groups:

  stream         full, first 1,000 queries, mid: the main streaming comparison
                 B1 vs B3 x order (id / random / mst) x cache (Belady / LRU), B3 page reads,
                 B3 with M = 8 / 32, B3 with q95 pruning
  stream_levels  same setting at low and high (B1 and B3, MST order, Belady)
  memory         streaming mid, total memory budget 400 / 512 / 768 MB
                 (cache = budget - ~212 MB process overhead, minus the codes for B3)
  batch          full, all 10,000 queries, low / mid / high: B1, B3 (bucket reads),
                 B3 (page reads), B2
  batch_M        batch B3 with M = 8 and 32
  batch_prune    batch B3 with q95 and q99 pruning (proven vs measured recall)
  dev            quick versions on the dev subset (sanity check, a few minutes)

Rough times on a laptop (they scale with how much pruning keeps):
  full batch run 5-15 min, full streaming B1 ~10 min, full streaming B3 1-3 min.
"""

OVERHEAD_MB = 212     # Python + NumPy + Numba + metadata, measured from the B1 runs
CODES_MB = {8: 100, 16: 180, 32: 340}  # codes + residuals for 10M vectors
N_STREAM = 1000

GROUP_ORDER = ["stream", "stream_levels", "memory", "batch", "batch_M", "batch_prune"]


def _run(script, group, est_min, **kw):
    args = []
    for k, v in kw.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        args += [flag] if v is True else [flag, str(v)]
    key = script + " " + " ".join(args)
    return {"script": script, "args": args, "key": key, "group": group, "est_min": est_min}


def b1(group, est, **kw):
    return _run("scripts.step10_b1", group, est, **kw)


def b3(group, est, **kw):
    return _run("scripts.step13_b3", group, est, **kw)


def build_grid():
    runs = []
    # ---------------- streaming (main result) ----------------
    S = dict(dataset="full", n_queries=N_STREAM)
    g = "stream"
    for order in ("id", "mst"):
        runs.append(b1(g, 10, level="mid", loop="query", order=order, **S))
    runs.append(b1(g, 10, level="mid", loop="query", order="mst", cache_policy="lru", **S))
    for order in ("id", "random", "mst"):
        runs.append(b3(g, 2, level="mid", M=16, loop="query", order=order, **S))
    runs.append(b3(g, 2, level="mid", M=16, loop="query", order="mst", cache_policy="lru", **S))
    runs.append(b3(g, 2, level="mid", M=16, loop="query", order="mst", read_policy="page", **S))
    for m in (8, 32):
        runs.append(b3(g, 2, level="mid", M=m, loop="query", order="mst", **S))
    runs.append(b3(g, 2, level="mid", M=16, loop="query", order="mst", pruning="q95", **S))

    g = "stream_levels"
    for lvl in ("low", "high"):
        runs.append(b1(g, 10, level=lvl, loop="query", order="mst", **S))
        runs.append(b3(g, 3, level=lvl, M=16, loop="query", order="mst", **S))

    g = "memory"
    for budget in (400, 512, 768):
        runs.append(b1(g, 10, level="mid", loop="query", order="mst", cache_mb=budget - OVERHEAD_MB,
                       cap_mb=budget, **S))
        runs.append(b3(g, 2, level="mid", M=16, loop="query", order="mst",
                       cache_mb=max(budget - OVERHEAD_MB - CODES_MB[16], 5), cap_mb=budget, **S))

    # ---------------- batch ----------------
    g = "batch"
    for lvl in ("low", "mid", "high"):
        runs.append(b1(g, 10, dataset="full", level=lvl))
        runs.append(b3(g, 10, dataset="full", level=lvl, M=16))
        runs.append(b3(g, 10, dataset="full", level=lvl, M=16, read_policy="page"))
        runs.append(b3(g, 6, dataset="full", level=lvl, M=16, system="B2"))
    g = "batch_M"
    for lvl in ("low", "mid", "high"):
        for m in (8, 32):
            runs.append(b3(g, 10, dataset="full", level=lvl, M=m))
    g = "batch_prune"
    for lvl in ("low", "mid", "high"):
        for q in ("q95", "q99"):
            runs.append(b3(g, 8, dataset="full", level=lvl, M=16, pruning=q))

    # ---------------- dev (quick) ----------------
    g = "dev"
    for lvl in ("low", "mid", "high"):
        runs.append(b1(g, 0.5, dataset="dev", level=lvl))
        runs.append(b3(g, 0.5, dataset="dev", level=lvl, M=16))
        runs.append(b3(g, 0.3, dataset="dev", level=lvl, M=16, system="B2"))
    for order in ("id", "mst"):
        runs.append(b1(g, 2.5, dataset="dev", level="mid", loop="query", order=order))
        runs.append(b3(g, 0.5, dataset="dev", level="mid", M=16, loop="query", order=order))
    return runs
