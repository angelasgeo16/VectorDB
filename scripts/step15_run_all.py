#!/usr/bin/env python3
"""Step 15: run the experiment grid. From the project root:

    python3 -m scripts.step15_run_all --dry-run                 # list runs + rough time
    python3 -m scripts.step15_run_all                           # all groups except dev
    python3 -m scripts.step15_run_all --groups stream,memory    # only some groups
    python3 -m scripts.step15_run_all --groups dev              # quick sanity pass

Each run is a separate process (honest peak RSS per run). Finished runs are listed
in results/done.txt and skipped next time, so the runner can be stopped (Ctrl+C)
and restarted. Full output of every run goes to results/logs/. A failed run is
logged and the runner moves on.

To keep the Mac awake overnight:   caffeinate -i python3 -m scripts.step15_run_all
"""
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import config
from src.experiments.grid import GROUP_ORDER, N_STREAM, build_grid

ROOT = Path(__file__).resolve().parent.parent


def ensure_orders():
    """Query orders needed by the streaming groups (fast)."""
    data_dir = Path(config.DATA_DIR)
    need = [("full", N_STREAM), ("full", None), ("dev", None)]
    for ds, n in need:
        n_eff = n or (10_000 if ds == "full" else 1_000)
        if not (data_dir / "orders" / ds / f"mst_n{n_eff}.npy").exists() and \
                (data_dir / "buckets" / ds / "centers.npy").exists():
            cmd = [sys.executable, "-m", "scripts.step11_order", "--dataset", ds]
            if n:
                cmd += ["--n-queries", str(n)]
            print("building query orders:", " ".join(cmd[2:]), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)


def summary_lines(log_text):
    keep = ("pairs:", "disk:", "time:", "peak RSS", "labels:", "Error", "error", "Traceback")
    return [ln.strip() for ln in log_text.splitlines() if any(k in ln for k in keep)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", default=",".join(GROUP_ORDER),
                   help="comma-separated: " + ", ".join(GROUP_ORDER + ["dev"]))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rerun", action="store_true", help="ignore results/done.txt")
    a = p.parse_args()

    results = Path(getattr(config, "RESULTS_DIR", ROOT / "results"))
    logs = results / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    done_path = results / "done.txt"
    done = set() if a.rerun or not done_path.exists() else set(done_path.read_text().splitlines())

    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    runs = [r for g in groups for r in build_grid() if r["group"] == g]
    todo = [r for r in runs if r["key"] not in done]
    est = sum(r["est_min"] for r in todo)
    print(f"{len(runs)} runs in groups {groups}; {len(runs) - len(todo)} already done; "
          f"{len(todo)} to run, roughly {est / 60:.1f} h")
    if a.dry_run:
        for r in todo:
            print(f"  [{r['group']:>13}] ~{r['est_min']:>4} min  python3 -m {r['key']}")
        return

    ensure_orders()
    t_all = time.time()
    failed = []
    for i, r in enumerate(todo, 1):
        name = hashlib.md5(r["key"].encode()).hexdigest()[:10]
        log = logs / f"{r['group']}_{name}.log"
        print(f"\n[{i}/{len(todo)}] ({r['group']}) python3 -m {r['key']}", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            f.write(f"$ python3 -m {r['key']}\n")
            f.flush()
            proc = subprocess.run([sys.executable, "-m", r["script"], *r["args"]], cwd=ROOT,
                                  stdout=f, stderr=subprocess.STDOUT)
        text = log.read_text()
        for ln in summary_lines(text)[-6:]:
            print("    " + ln)
        mins = (time.time() - t0) / 60
        if proc.returncode == 0:
            with open(done_path, "a") as f:
                f.write(r["key"] + "\n")
            print(f"    done in {mins:.1f} min (log: {log.name})", flush=True)
        else:
            failed.append((r["key"], log.name))
            print(f"    FAILED after {mins:.1f} min, see {log}", flush=True)
    print(f"\nFinished {len(todo) - len(failed)} of {len(todo)} runs in {(time.time() - t_all) / 3600:.1f} h")
    if failed:
        print("Failed runs:")
        for k, ln in failed:
            print(f"  {k}   ({ln})")
    print("Next: python3 -m scripts.step16_plots")


if __name__ == "__main__":
    main()
