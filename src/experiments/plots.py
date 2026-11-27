"""Step 16: figures and summary tables from results/runs.csv (+ feasibility.json).

Every figure is skipped with a note if its runs are not in runs.csv yet, so this can
be re-run at any point. When a configuration was run more than once, the latest row
is used. Outputs go to results/plots/ (PNG) plus results/plots/summary.md (tables of
the numbers behind every figure).

Style: one y-axis per chart, recessive grid, thin bars with a gap, value labels on
bars, a legend whenever there are 2+ series, fixed colors per entity:
    B1 blue, B3 orange, B3 page reads aqua, B2 yellow;  M = 8 / 16 / 32 the same order.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
INK, INK2, GRID = "#191919", "#666666", "#e8e8e8"
SYSTEM_COLOR = {"B1": BLUE, "B3": ORANGE, "B3 page": AQUA, "B2": YELLOW}
M_COLOR = {8: BLUE, 16: ORANGE, 32: AQUA}
LEVELS = ("low", "mid", "high")
STREAM_NX = 1000   # set from the data in run()
BATCH_NX = 10000

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlepad": 26, "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 8,
})


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
CONFIG_FIELDS = ["system", "dataset", "level", "M", "pruning", "order", "read_policy", "cache_MB",
                 "cache_policy", "n_x"]


def load_runs(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    latest = {}
    for r in rows:
        latest[tuple(r.get(k, "") for k in CONFIG_FIELDS)] = r
    out = []
    for r in latest.values():
        for k in ("bytes_read_MB", "total_s", "plan_s", "filter_s", "io_s", "verify_s", "recall", "precision",
                  "proven_recall", "maybe_frac", "cache_hit_rate", "peak_rss_MB"):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError, KeyError):
                r[k] = None
        for k in ("n_x", "M"):
            r[k] = int(r[k]) if r.get(k) not in (None, "") else None
        r["cache_MB"] = float(r["cache_MB"]) if r.get("cache_MB") not in (None, "") else None
        out.append(r)
    return out


def pick(runs, **cond):
    """Rows matching every condition (value None in cond = field must be empty/None)."""
    res = []
    for r in runs:
        ok = True
        for k, v in cond.items():
            rv = r.get(k)
            if isinstance(v, (list, tuple, set)):
                ok = rv in v
            elif v is None:
                ok = rv in (None, "")
            else:
                ok = rv == v
            if not ok:
                break
        if ok:
            res.append(r)
    return res


def one(runs, **cond):
    rs = pick(runs, **cond)
    return rs[0] if rs else None


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def fmt(v, unit=""):
    if v is None:
        return ""
    if v >= 100:
        return f"{v:,.0f}{unit}"
    if v >= 10:
        return f"{v:.1f}{unit}"
    return f"{v:.2f}{unit}"


def grouped_bars(ax, groups, series, values, colors, unit="", ylabel=""):
    """groups: x labels; series: names; values[s][g] (None = missing)."""
    n = len(series)
    width = 0.8 / max(n, 1)
    for i, s in enumerate(series):
        xs, hs = [], []
        for j, _ in enumerate(groups):
            v = values[s][j]
            if v is not None:
                xs.append(j - 0.4 + width * (i + 0.5))
                hs.append(v)
        bars = ax.bar(xs, hs, width=width * 0.86, color=colors[i], label=s, zorder=2)
        for b, h in zip(bars, hs):
            ax.annotate(fmt(h, unit), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.12)
    legend_above(ax)


def legend_above(ax, ncols=4):
    """Legend in one row between the title and the plot, so it never covers data."""
    h, l = ax.get_legend_handles_labels()
    if len(h) >= 2:
        ax.legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=min(ncols, len(h)),
                  frameon=False, borderaxespad=0.3, handlelength=1.4)


def save(fig, out_dir, name, notes):
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    notes.append(f"saved {path.name}")
    return path


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_stream_orders(runs, out, notes, md):
    groups = ["id", "random", "mst"]
    vals = {"B1": [], "B3": []}
    rows_md = []
    for s, extra in (("B1", {}), ("B3", {"M": 16, "pruning": "exact", "read_policy": "bucket"})):
        for o in groups:
            r = one(runs, system=s, dataset="full", level="mid", order=o, cache_policy="belady",
                    n_x=STREAM_NX, **extra)
            if s == "B1":
                r = r if r and r["cache_MB"] in (None, 300.0) else (r if r else None)
            vals[s].append(r["bytes_read_MB"] / 1e3 if r else None)
            if r:
                rows_md.append(f"| {s} | {o} | {r['bytes_read_MB'] / 1e3:,.1f} | {fmt(r['cache_hit_rate'])} | "
                               f"{r['total_s']:,.0f} |")
    if all(v is None for s in vals for v in vals[s]):
        notes.append("skip stream_orders: no streaming runs on full (n=1000) yet")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bars(ax, groups, ["B1", "B3"], vals, [BLUE, ORANGE], ylabel="GB read from SSD")
    ax.set_title(f"Streaming: disk reads by query order (full, {STREAM_NX:,} queries, mid ε)")
    ax.set_xlabel("query order (Belady cache, same total memory)")
    save(fig, out, "1_stream_bytes_by_order", notes)
    md += [f"## Streaming: query order (full, {STREAM_NX:,} queries, mid)", "",
           "| system | order | GB read | cache hit rate | total s |", "|---|---|---|---|---|", *rows_md, ""]


def fig_stream_breakdown(runs, out, notes, md):
    cfgs = [("B1 id", dict(system="B1", order="id")),
            ("B1 mst", dict(system="B1", order="mst")),
            ("B3 id", dict(system="B3", order="id", M=16, pruning="exact", read_policy="bucket")),
            ("B3 mst", dict(system="B3", order="mst", M=16, pruning="exact", read_policy="bucket"))]
    rows = []
    for name, c in cfgs:
        r = one(runs, dataset="full", level="mid", n_x=STREAM_NX, cache_policy="belady", **c)
        if r and (r["system"] != "B1" or r["cache_MB"] in (None, 300.0)):
            rows.append((name, r))
    if not rows:
        notes.append("skip stream_time: no streaming runs yet")
        return
    parts = [("filter (codes)", "filter_s", ORANGE), ("disk I/O", "io_s", BLUE),
             ("exact checks", "verify_s", AQUA), ("pruning", "plan_s", YELLOW)]
    fig, ax = plt.subplots(figsize=(7, 4))
    bottoms = [0.0] * len(rows)
    for label, key, color in parts:
        hs = [r[key] or 0.0 for _, r in rows]
        ax.bar(range(len(rows)), hs, bottom=bottoms, width=0.55, color=color, label=label,
               edgecolor="white", linewidth=1.5, zorder=2)
        bottoms = [b + h for b, h in zip(bottoms, hs)]
    for i, (_, r) in enumerate(rows):
        ax.annotate(f"{r['total_s']:,.0f} s", (i, bottoms[i]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([n for n, _ in rows])
    ax.set_ylabel("seconds")
    ax.margins(y=0.12)
    ax.set_title(f"Streaming: where the time goes (full, {STREAM_NX:,} queries, mid ε)")
    legend_above(ax)
    save(fig, out, "2_stream_time_breakdown", notes)


def fig_cache_policy(runs, out, notes, md):
    vals = {"Belady": [], "LRU": []}
    groups = ["B1", "B3"]
    for pol, key in (("belady", "Belady"), ("lru", "LRU")):
        for s in groups:
            extra = {"M": 16, "pruning": "exact", "read_policy": "bucket"} if s == "B3" else {}
            r = one(runs, system=s, dataset="full", level="mid", order="mst", cache_policy=pol, n_x=STREAM_NX, **extra)
            if r and s == "B1" and r["cache_MB"] not in (None, 300.0):
                r = None
            vals[key].append(r["bytes_read_MB"] / 1e3 if r else None)
    if all(v is None for k in vals for v in vals[k]):
        notes.append("skip cache_policy: no LRU / Belady pair yet")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped_bars(ax, groups, ["Belady", "LRU"], vals, [BLUE, ORANGE], ylabel="GB read from SSD")
    ax.set_title("Cache policy (streaming, MST order, mid ε)")
    save(fig, out, "3_cache_policy", notes)


def fig_memory(runs, out, notes, md):
    pts = {"B1": [], "B3": []}
    for s in pts:
        extra = {"M": 16, "pruning": "exact", "read_policy": "bucket"} if s == "B3" else {}
        for r in pick(runs, system=s, dataset="full", level="mid", order="mst", cache_policy="belady",
                      n_x=STREAM_NX, **extra):
            if r["cache_MB"] is not None:
                budget = r["cache_MB"] + 212 + (180 if s == "B3" else 0)
                pts[s].append((round(budget), r["bytes_read_MB"] / 1e3))
    if sum(len(v) for v in pts.values()) < 3:
        notes.append("skip memory_budget: need the memory group")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for s, color in (("B1", BLUE), ("B3", ORANGE)):
        p = sorted(set(pts[s]))
        if p:
            ax.plot([a for a, _ in p], [b for _, b in p], marker="o", color=color, label=s, zorder=3)
    ax.set_xlabel("total memory budget (MB): cache + codes + ~212 MB process overhead")
    ax.set_ylabel("GB read from SSD")
    ax.set_ylim(bottom=0)
    ax.set_title("Streaming: disk reads vs memory budget (mid ε)")
    legend_above(ax)
    save(fig, out, "4_memory_budget", notes)


def fig_batch_time(runs, out, notes, md):
    series = [("B1", dict(system="B1")),
              ("B3", dict(system="B3", M=16, pruning="exact", read_policy="bucket")),
              ("B3 page", dict(system="B3", M=16, pruning="exact", read_policy="page")),
              ("B2", dict(system="B2", M=16, pruning="exact"))]
    vals = {s: [] for s, _ in series}
    rows_md = []
    for s, c in series:
        for lvl in LEVELS:
            r = one(runs, dataset="full", level=lvl, order="bucket-major", n_x=BATCH_NX, **c)
            vals[s].append(r["total_s"] / 60 if r else None)
            if r:
                rows_md.append(f"| {s} | {lvl} | {r['total_s'] / 60:.1f} | {r['bytes_read_MB'] / 1e3:.2f} | "
                               f"{r['recall']:.4f} | {r['precision']:.4f} | {fmt(r['peak_rss_MB'])} |")
    if not rows_md:
        notes.append("skip batch_time: no batch runs on full yet")
        return
    names = [s for s, _ in series]
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped_bars(ax, list(LEVELS), names, vals, [SYSTEM_COLOR[s] for s in names], ylabel="minutes")
    ax.set_title("Batch (bucket at a time): total time, full 10K × 10M")
    ax.set_xlabel("ε level")
    save(fig, out, "5_batch_time", notes)
    md += ["## Batch: all systems (full, 10,000 queries)", "",
           "| system | level | minutes | GB read | recall | precision | peak RSS MB |",
           "|---|---|---|---|---|---|---|", *rows_md, ""]


def fig_maybe_vs_m(results, out, notes, md):
    path = results / "feasibility.json"
    if not path.exists():
        notes.append("skip maybe_vs_M: results/feasibility.json missing (Step 5)")
        return
    feas = json.load(open(path))["per_M"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rows_md = []
    for m in (8, 16, 32):
        if str(m) not in feas:
            continue
        ys = [feas[str(m)][l]["maybe_per_true"] for l in LEVELS]
        ax.plot(range(3), ys, marker="o", color=M_COLOR[m], label=f"M = {m}", zorder=3)
        rows_md += [f"| {m} | {l} | {feas[str(m)][l]['maybe_per_true']:.1f} | "
                    f"{feas[str(m)][l]['yes_per_true']:.3f} |" for l in LEVELS]
    ax.axhline(50, color=INK2, linewidth=1, linestyle="--", zorder=1)
    ax.annotate("feasibility gate (50×)", (0, 50), xytext=(0, 4), textcoords="offset points", color=INK2,
                fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(LEVELS)
    ax.set_xlim(-0.2, 2.2)
    ax.set_xlabel("ε level (dev)")
    ax.set_ylabel("Maybe pairs per true match (log scale)")
    ax.set_title("Residual bound: Maybe pairs per true match (dev, 1B pairs)")
    legend_above(ax)
    save(fig, out, "6_maybe_vs_M", notes)
    md += ["## Feasibility gate (dev, Step 5)", "", "| M | level | Maybe per true match | share answered Yes |",
           "|---|---|---|---|", *rows_md, ""]


def fig_recall(runs, out, notes, md):
    rows = []
    for q in ("q99", "q95"):
        for lvl in LEVELS:
            r = one(runs, system="B3", dataset="full", level=lvl, M=16, pruning=q, order="bucket-major",
                    read_policy="bucket", n_x=BATCH_NX)
            if r:
                rows.append((f"{q[1:]}% · {lvl}", r))
    if not rows:
        notes.append("skip recall: no batch q95/q99 runs yet")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = range(len(rows))
    ax.plot(xs, [r["recall"] for _, r in rows], "o", color=ORANGE, label="measured recall", zorder=3)
    ax.plot(xs, [r["proven_recall"] for _, r in rows], "D", color=BLUE, label="proven lower bound", zorder=3)
    for i, (_, r) in enumerate(rows):
        ax.vlines(i, r["proven_recall"], r["recall"], color=GRID, linewidth=2, zorder=2)
        ax.annotate(f"{r['recall']:.4f}", (i, r["recall"]), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8)
        ax.annotate(f"{r['proven_recall']:.3f}", (i, r["proven_recall"]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([n for n, _ in rows])
    lo = min(r["proven_recall"] for _, r in rows)
    ax.set_ylim(max(0.0, lo - 0.06), 1.07)
    ax.set_ylabel("recall")
    ax.set_title("Probabilistic pruning: measured recall vs proven bound (B3, full)")
    legend_above(ax)
    save(fig, out, "7_recall_proven_vs_measured", notes)
    md += ["## Probabilistic pruning (B3, full batch)", "",
           "| quantile · level | measured recall | proven bound | minutes |", "|---|---|---|---|",
           *[f"| {n} | {r['recall']:.5f} | {r['proven_recall']:.4f} | {r['total_s'] / 60:.1f} |" for n, r in rows], ""]


def fig_b2(runs, out, notes, md):
    rows = []
    for lvl in LEVELS:
        r = one(runs, system="B2", dataset="full", level=lvl, M=16, order="bucket-major", n_x=BATCH_NX)
        if r:
            rows.append((lvl, r))
    if not rows:
        notes.append("skip b2: no B2 runs on full yet")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = range(len(rows))
    ax.plot(xs, [r["recall"] for _, r in rows], marker="o", color=BLUE, label="B2 recall", zorder=3)
    ax.plot(xs, [r["precision"] for _, r in rows], marker="D", color=ORANGE, label="B2 precision", zorder=3)
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
    ax.annotate("B3: recall = precision = 1.0", (len(rows) - 0.45, 1.0), xytext=(0, 4),
                textcoords="offset points", ha="right", color=INK2, fontsize=8)
    for i, (_, r) in enumerate(rows):
        ax.annotate(f"{r['recall']:.3f}", (i, r["recall"]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8)
        ax.annotate(f"{r['precision']:.3f}", (i, r["precision"]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l for l, _ in rows])
    ax.set_xlim(-0.3, len(rows) - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("ε level")
    ax.set_ylabel("recall / precision")
    ax.set_title("B2 (codes only, no bound): accuracy, M = 16, full")
    legend_above(ax)
    save(fig, out, "8_b2_accuracy", notes)


def fig_m_sweep(runs, out, notes, md):
    rows_md = []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    any_line = False
    for m in (8, 16, 32):
        ys = []
        for lvl in LEVELS:
            r = one(runs, system="B3", dataset="full", level=lvl, M=m, pruning="exact", order="bucket-major",
                    read_policy="bucket", n_x=BATCH_NX)
            ys.append(r["total_s"] / 60 if r else None)
            if r:
                rows_md.append(f"| {m} | {lvl} | {r['maybe_frac'] * 100:.4f}% | {r['total_s'] / 60:.1f} | "
                               f"{fmt(r['peak_rss_MB'])} |")
        pts = [(i, y) for i, y in enumerate(ys) if y is not None]
        if len(pts) >= 2:
            any_line = True
            ax.plot([a for a, _ in pts], [b for _, b in pts], marker="o", color=M_COLOR[m], label=f"M = {m}",
                    zorder=3)
    if not any_line:
        plt.close(fig)
        notes.append("skip M_sweep: need batch_M runs")
        return
    ax.set_xticks(range(3))
    ax.set_xticklabels(LEVELS)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("ε level")
    ax.set_ylabel("minutes")
    ax.set_title("B3 batch time by PQ size M (full)")
    legend_above(ax)
    save(fig, out, "9_batch_M_sweep", notes)
    md += ["## PQ size M (B3, full batch)", "", "| M | level | Maybe share of all pairs | minutes | peak RSS MB |",
           "|---|---|---|---|---|", *rows_md, ""]


def run(cfg):
    data_dir = Path(cfg.DATA_DIR)
    results = Path(getattr(cfg, "RESULTS_DIR", data_dir.parent / "results"))
    out = results / "plots"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results / "runs.csv")
    global STREAM_NX, BATCH_NX
    s_nx = sorted({r["n_x"] for r in runs if r["dataset"] == "full" and r["order"] != "bucket-major"})
    b_nx = sorted({r["n_x"] for r in runs if r["dataset"] == "full" and r["order"] == "bucket-major"})
    STREAM_NX = s_nx[0] if s_nx else 1000
    BATCH_NX = b_nx[-1] if b_nx else 10000
    notes, md = [], ["# Results summary", "", f"{len(runs)} distinct configurations in runs.csv.", ""]
    fig_stream_orders(runs, out, notes, md)
    fig_stream_breakdown(runs, out, notes, md)
    fig_cache_policy(runs, out, notes, md)
    fig_memory(runs, out, notes, md)
    fig_batch_time(runs, out, notes, md)
    fig_maybe_vs_m(results, out, notes, md)
    fig_recall(runs, out, notes, md)
    fig_b2(runs, out, notes, md)
    fig_m_sweep(runs, out, notes, md)
    (out / "summary.md").write_text("\n".join(md))
    for n in notes:
        print("  " + n)
    print(f"Plots and summary.md in {out}")
