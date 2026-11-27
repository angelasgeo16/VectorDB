"""Central configuration for the PQ vector-join project.

All paths, dataset sizes, and hyperparameters live here so every script and
module agrees on them. Import this module rather than hard-coding values.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent

# Plain clone of big-ann-benchmarks (gitignored, not a submodule).
EXTERNAL_DIR = ROOT / "external" / "big-ann-benchmarks"

# create_dataset.py --dataset bigann-10M writes here. We point straight at the
# produced files: no symlinks, no copies (the base crop is 1.28 GB).
BIGANN_DIR = EXTERNAL_DIR / "data" / "bigann"
BASE_BIN = BIGANN_DIR / "base.1B.u8bin.crop_nb_10000000"
QUERY_BIN = BIGANN_DIR / "query.public.10K.u8bin"

# Output directory for everything we generate.
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR

# Step 1 outputs
Y_BIN = PROCESSED_DIR / "Y.u8bin"          # 10M x 128 uint8
X_BIN = PROCESSED_DIR / "X.u8bin"          # 10K x 128 uint8
Y_DEV_BIN = PROCESSED_DIR / "Y_dev.u8bin"  # 1M x 128 uint8
X_DEV_BIN = PROCESSED_DIR / "X_dev.u8bin"  # 1K x 128 uint8

# Step 2 output (dev and full-radius epsilon values)
EPS_JSON = PROCESSED_DIR / "eps.json"

# Later-step output directories
GT_DIR = PROCESSED_DIR / "ground_truth"
PQ_DIR = PROCESSED_DIR / "pq"
FILTER_DIR = PROCESSED_DIR / "filter"

# Experiment log (Step 15)
RESULTS_DIR = ROOT / "results"
RUNS_CSV = RESULTS_DIR / "runs.csv"

# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------
D = 128                    # SIFT dimension
N_Y_FULL = 10_000_000      # first 10M base vectors
N_X_FULL = 10_000          # official query set
N_Y_DEV = 1_000_000        # dev subset of Y (prefix)
N_X_DEV = 1_000            # dev subset of X (prefix)

# Expected .u8bin sizes: 8-byte header (n, d as uint32) + n * d bytes.
EXPECTED_Y_BYTES = 8 + N_Y_FULL * D        # 1,280,000,008
EXPECTED_X_BYTES = 8 + N_X_FULL * D        # 1,280,008
EXPECTED_Y_DEV_BYTES = 8 + N_Y_DEV * D     # 128,000,008
EXPECTED_X_DEV_BYTES = 8 + N_X_DEV * D     # 128,008

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42           # 500K PQ training sample, any shuffling

# ---------------------------------------------------------------------------
# PQ parameters (Step 4)
# ---------------------------------------------------------------------------
PQ_M_VALUES = [8, 16, 32]  # subquantizer sweep
PQ_K = 256                 # centroids per subquantizer (uint8 codes)
PQ_TRAIN_SAMPLE = 500_000  # training sample size drawn from Y
PQ_N_ITER = 20             # k-means iterations
PQ_SEED = RANDOM_SEED      # reproducible training sample / k-means seed

# ---------------------------------------------------------------------------
# Epsilon selection (Step 2)
# ---------------------------------------------------------------------------
EPS_K_NN = 2000                 # k for k-NN (pool 1K x 2000 distances)
EPS_TARGET_MEAN = [10, 100, 1000]  # target mean matches per x (low/mid/high)
EPS_SAMPLE_X = 1000             # number of x sampled to pick epsilon (X_dev prefix)
EPS_CHUNK_SIZE = 1_000_000      # Y chunk size for full 10M (same as GT)
EPS_TOLERANCE = 0.10            # ±10% tolerance on mean matches
EPS_BISECT_MAX_ITER = 20        # max bisection iterations

# ---------------------------------------------------------------------------
# Ground truth (Step 3)
# ---------------------------------------------------------------------------
GT_CHUNK_SIZE = 1_000_000  # Y is scanned in 1M-vector chunks

# ---------------------------------------------------------------------------
# Memory / runtime
# ---------------------------------------------------------------------------
MEMORY_CAP_GB = 0.5            # 512 MB capped runs
CACHE_BYTES_BUDGET = 300_000_000  # ~300 MB for the bucket cache

# ---------------------------------------------------------------------------
# Feasibility gate (Step 5)
# ---------------------------------------------------------------------------
MAYBE_MULTIPLIER_THRESHOLD = 50  # Maybe > 50x true matches -> try OPQ
FEAS_CHUNK_SIZE = 20_000          # Y rows per chunk (20K x 1K pairs = 20M per chunk)

# ---------------------------------------------------------------------------
# Unit test (Step 6)
# ---------------------------------------------------------------------------
TEST_PAIRS = 100_000  # random pairs for the Yes/No correctness check


BUCKET_N_CENTERS = {"dev": 1_000, "full": 10_000}
BUCKET_QUANTILES = (0.95, 0.99)
BUCKET_CHUNK = 1_000_000
MEMORY_CAP_MB = 512
B1_CACHE_MB = {"dev": 30, "full": 300}
RESULTS_DIR = ROOT / "results"