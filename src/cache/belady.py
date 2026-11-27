"""Step 9: bucket caches with a byte capacity.

BeladyCache: the access sequence is known in advance (the join plan), so on a
miss we evict the cached bucket whose next use is furthest in the future.
If the incoming bucket's own next use is further than every cached bucket's
(or it is never used again), it is not cached at all ("bypass").

LRUCache: same interface, least-recently-used eviction (for comparison).

Interface (both):
    cache = BeladyCache(capacity_bytes, loader, sequence)
    data  = cache.get(i, b)   # i = position in the sequence, b = sequence[i]
    cache.stats()
"""
import heapq
from collections import OrderedDict

import numpy as np


def next_use(seq):
    """next_use[i] = smallest j > i with seq[j] == seq[i], or len(seq) if none."""
    seq = np.asarray(seq)
    n = len(seq)
    nxt = np.full(n, n, dtype=np.int64)
    if n == 0:
        return nxt
    order = np.argsort(seq, kind="stable")
    same = seq[order[:-1]] == seq[order[1:]]
    nxt[order[:-1][same]] = order[1:][same]
    return nxt


class _Stats:
    def _init_stats(self, capacity):
        self.capacity = int(capacity)
        self.used = 0
        self.peak_used = 0
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.evictions = 0
        self.bytes_loaded = 0

    def stats(self):
        total = self.hits + self.misses
        return {"cache_capacity_bytes": self.capacity, "cache_hits": self.hits, "cache_misses": self.misses,
                "cache_hit_rate": self.hits / total if total else 0.0, "cache_bypasses": self.bypasses,
                "cache_evictions": self.evictions, "cache_peak_bytes": self.peak_used,
                "bytes_loaded": self.bytes_loaded}


class BeladyCache(_Stats):
    def __init__(self, capacity_bytes, loader, sequence):
        self._init_stats(capacity_bytes)
        self.loader = loader
        self.seq = np.asarray(sequence)
        self.nxt = next_use(self.seq)
        self.never = len(self.seq)
        self.store = {}
        self.size = {}
        self.cur_next = {}
        self.heap = []  # (-next_use, bucket), lazily invalidated

    def _furthest(self):
        while self.heap:
            neg, b = self.heap[0]
            if b in self.store and self.cur_next[b] == -neg:
                return -neg, b
            heapq.heappop(self.heap)
        return None, None

    def get(self, i, b):
        nu = int(self.nxt[i])
        if b in self.store:
            self.hits += 1
            self.cur_next[b] = nu
            heapq.heappush(self.heap, (-nu, b))
            return self.store[b]

        self.misses += 1
        data = self.loader(b)
        size = data.nbytes
        self.bytes_loaded += size
        if nu >= self.never or size > self.capacity:
            self.bypasses += 1
            return data
        while self.used + size > self.capacity:
            far_nu, far_b = self._furthest()
            if far_b is None or nu >= far_nu:
                self.bypasses += 1
                return data
            heapq.heappop(self.heap)
            self.used -= self.size.pop(far_b)
            del self.store[far_b]
            del self.cur_next[far_b]
            self.evictions += 1
        self.store[b] = data
        self.size[b] = size
        self.cur_next[b] = nu
        heapq.heappush(self.heap, (-nu, b))
        self.used += size
        self.peak_used = max(self.peak_used, self.used)
        return data


class LRUCache(_Stats):
    def __init__(self, capacity_bytes, loader, sequence=None):
        self._init_stats(capacity_bytes)
        self.loader = loader
        self.store = OrderedDict()

    def get(self, i, b):
        if b in self.store:
            self.hits += 1
            self.store.move_to_end(b)
            return self.store[b]
        self.misses += 1
        data = self.loader(b)
        size = data.nbytes
        self.bytes_loaded += size
        if size > self.capacity:
            self.bypasses += 1
            return data
        while self.used + size > self.capacity:
            _, old = self.store.popitem(last=False)
            self.used -= old.nbytes
            self.evictions += 1
        self.store[b] = data
        self.used += size
        self.peak_used = max(self.peak_used, self.used)
        return data
