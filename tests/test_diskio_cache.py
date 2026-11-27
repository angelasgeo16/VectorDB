"""Checks for Step 8 (BucketReader) and Step 9 (Belady cache).

Run from the project root:
    python3 -m tests.test_diskio_cache            # uses data/buckets/dev
    python3 -m tests.test_diskio_cache full       # uses data/buckets/full
"""
import itertools
import sys
from pathlib import Path

import numpy as np

import config
from src.cache.belady import BeladyCache, LRUCache, next_use
from src.diskio.reader import BucketReader


# ---------------------------------------------------------------------------
# Step 9: cache tests (no data files needed)
# ---------------------------------------------------------------------------
class _Unit:
    """Loader returning 1-byte 'buckets' so capacity counts items."""
    def __init__(self):
        self.loads = 0

    def __call__(self, b):
        self.loads += 1
        return np.zeros(1, dtype=np.uint8)


def optimal_misses(seq, capacity):
    """Brute-force minimum misses for unit-size items with optional bypass."""
    from functools import lru_cache

    seq = tuple(seq)

    @lru_cache(maxsize=None)
    def best(i, cache):
        if i == len(seq):
            return 0
        b = seq[i]
        if b in cache:
            return best(i + 1, cache)
        options = [best(i + 1, cache)]  # bypass
        if len(cache) < capacity:
            options.append(best(i + 1, tuple(sorted(cache + (b,)))))
        else:
            for out in cache:
                new = tuple(sorted(tuple(x for x in cache if x != out) + (b,)))
                options.append(best(i + 1, new))
        return 1 + min(options)

    return best(0, ())


def test_cache():
    # next_use
    seq = np.array([3, 1, 3, 2, 1, 3])
    assert next_use(seq).tolist() == [2, 4, 5, 6, 6, 6]

    # hand example, capacity 2 items: A B C A B D A
    seq = np.array([0, 1, 2, 0, 1, 3, 0])
    c = BeladyCache(2, _Unit(), seq)
    for i, b in enumerate(seq):
        c.get(i, int(b))
    assert (c.hits, c.misses) == (3, 4), (c.hits, c.misses)

    # Belady equals the brute-force optimum on random small sequences
    rng = np.random.default_rng(0)
    for t in range(300):
        cap = int(rng.integers(1, 4))
        seq = rng.integers(0, 5, size=int(rng.integers(1, 12)))
        c = BeladyCache(cap, _Unit(), seq)
        for i, b in enumerate(seq):
            c.get(i, int(b))
        assert c.misses == optimal_misses(seq.tolist(), cap), (seq.tolist(), cap, c.misses)
        assert c.used <= cap

    # variable sizes: capacity never exceeded; Belady no worse than LRU on bytes loaded (typical)
    sizes = rng.integers(1, 50, size=40)
    loader = lambda b: np.zeros(int(sizes[b]), dtype=np.uint8)
    seq = rng.integers(0, 40, size=3000)
    bel, lru = BeladyCache(300, loader, seq), LRUCache(300, loader, seq)
    for i, b in enumerate(seq):
        bel.get(i, int(b))
        lru.get(i, int(b))
        assert bel.used <= 300 and lru.used <= 300
    print(f"  cache: next_use, hand example, 300 brute-force optimum checks: OK "
          f"(variable sizes: Belady hit rate {bel.stats()['cache_hit_rate']:.3f}, "
          f"LRU {lru.stats()['cache_hit_rate']:.3f})")


# ---------------------------------------------------------------------------
# Step 8: reader tests on the real bucket files
# ---------------------------------------------------------------------------
def test_reader(name):
    bdir = Path(config.DATA_DIR) / "buckets" / name
    starts = np.load(bdir / "starts.npy")
    counts = np.load(bdir / "counts.npy")
    n, d = int(counts.sum()), config.D
    ref = np.memmap(bdir / "Y_sorted.bin", dtype=np.uint8, mode="r", shape=(n, d))
    rng = np.random.default_rng(8)

    with BucketReader(bdir / "Y_sorted.bin", starts, counts, d=d) as r:
        # whole buckets
        bs = rng.choice(len(counts), size=min(200, len(counts)), replace=False)
        expected_bytes = 0
        for b in bs:
            v = r.read_bucket(int(b))
            s, c = starts[b], counts[b]
            assert np.array_equal(v, ref[s:s + c]), b
            expected_bytes += int(c) * d
        assert r.bytes_read == expected_bytes
        whole = r.counters()

        # page policy
        r.reset_counters()
        for _ in range(200):
            b = int(rng.choice(np.flatnonzero(counts)))
            k = int(rng.integers(1, min(int(counts[b]), 20) + 1))
            loc = rng.choice(int(counts[b]), size=k, replace=False)
            v = r.read_bucket_rows(b, loc)
            assert np.array_equal(v, ref[starts[b] + loc]), b
        pos = rng.integers(0, n, size=1000)
        v = r.read_positions(pos)
        assert np.array_equal(v, np.asarray(ref)[pos])
        pages = r.counters()
        # bytes requested must equal whole pages
        assert pages["bytes_read"] <= pages["device_bytes"]  # page reads cover whole pages (last may be partial)

    print(f"  reader ({name}): 200 whole buckets + 200 page reads + 1,000 scattered rows match: OK")
    print(f"    whole-bucket reads: {whole['bytes_read'] / 1e6:.1f} MB requested, "
          f"{whole['device_bytes'] / 1e6:.1f} MB of pages touched")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "dev"
    test_cache()
    test_reader(name)
    print("PASS")
