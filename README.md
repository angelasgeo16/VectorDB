# PQ Vector Join

Exact similarity join under a memory cap. CS 594 Vector Databases (Fall 2026, UIC), Angela George & Yash Yeole.

## What we did

We find every pair (x, y) with ‖x − y‖ ≤ ε between 10K queries and 10M base vectors (BigANN, SIFT, 128-d), using at most **512 MB of RAM**, with the base vectors on SSD.

Our method (B3) keeps a compressed **PQ code** of every base vector in RAM, plus its residual r = ‖y − ŷ‖. The triangle inequality bounds the true distance by d(x, ŷ) ± r, so each pair is labeled in RAM:

- **Yes**: definitely a match, no disk read
- **No**: definitely not a match, no disk read
- **Maybe**: read the vector from SSD and check it exactly

The result stays exact. The base vectors are stored in buckets on SSD (DiskJoin-style), and queries can be processed in MST order (SimJoin-style) so nearby queries reuse cached buckets.

Systems compared:
- **B1 (baseline):** the same buckets without codes; every candidate bucket is read from SSD and checked exactly
- **B2 (codes only):** PQ codes alone, with no bound and no verification; fast but not exact
- **B3 (ours):** buckets + PQ codes + residual bound; Yes / No decided in RAM, only Maybe pairs read from SSD and verified; exact

## Results

Full dataset, 10K × 10M, M = 16.

**Streaming (queries handled one at a time, first 1,000 queries, mid ε)**

| | B1 | B3 (ours) |
|---|---|---|
| Disk read, ID order | 210 GB | 2.7 GB |
| Disk read, MST order | 168 GB | 2.2 GB |
| Speed-up | | 5–6× |

B3 reads about 77× less from disk, with identical results.

**Batch (all queries known in advance)**

- B1 reads the base set once in order, which is hard to beat: B3 is 20–80% slower in total time.
- With page-level reads, B3 reads 4.6× less at low ε and 1.7× less at mid ε.
- B2 (codes only) misses 9–36% of matches and 16–27% of its output is wrong. The residual bound is what makes the codes usable.
- Exact B1 and B3 runs always return exactly the ground-truth pairs.

**Other findings**

- Larger M gives tighter codes: Maybe pairs per true match drop from 159× (M = 8) to 14× (M = 16) to 3× (M = 32).
- Pruning with the 95th-percentile bucket radius cuts B3 batch time by 29–41% with no measured recall loss.
- Limitation: B3 batch exceeds the 512 MB cap at high ε.

## Setup

```
python3 -m pip install numpy scipy faiss-cpu numba matplotlib
```

All settings (paths, ε targets, M values, bucket counts, memory cap) are in `config.py`. Run every command from the project root.

## How to run

```
python3 -m scripts.prepare_data            # download BigANN10M
python3 -m scripts.step1_load              # split into Y, X and dev subsets
python3 -m scripts.step2_pick_eps          # choose the three ε levels
python3 -m scripts.step3_ground_truth      # exact ground truth
python3 -m scripts.step4_pq                # PQ codes + residuals (M = 8, 16, 32)
python3 -m scripts.step5_feasibility       # how many pairs end up Maybe
python3 -m scripts.step7_bucketize         # buckets on SSD
python3 -m scripts.step11_order --dataset full --n-queries 1000   # query orders

python3 -m scripts.step10_b1 --dataset full --level mid           # baseline B1
python3 -m scripts.step13_b3 --dataset full --level mid --M 16    # our method B3

python3 -m scripts.step15_run_all          # full experiment grid
python3 -m scripts.step16_plots            # figures + summary tables
```

Useful options for B1 / B3:
- `--loop query --order mst` for the streaming setting
- `--read-policy page` for page-level reads
- `--pruning q95` for probabilistic pruning
- `--system B2` for the codes-only baseline

Tests: `python3 -m tests.test_kernel`, `python3 -m tests.test_diskio_cache`

## Project layout

```
config.py       settings
scripts/        one script per step
src/            data, ground truth, PQ, filter kernel, buckets, disk I/O, cache, join (B1, B3), experiments
tests/          kernel, disk reader and cache tests
results/        runs.csv, plots/, summary.md
```
