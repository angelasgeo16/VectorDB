"""Step 8: SSD read layer for the bucket-ordered Y file.

Reads go through os.pread on a raw file (Y_sorted.bin, no header), and every
read is counted. OS caching is switched off where possible so repeated reads
really hit the disk:
    macOS: fcntl F_NOCACHE on the file descriptor
    Linux: posix_fadvise(DONTNEED) after each read
(The byte counters are the primary metric either way.)

Two read policies:
    read_bucket(b)        whole bucket: one contiguous read           (DiskJoin)
    read_positions(pos)   only the 4 KB pages that hold those vectors  (page policy)

Counters:
    bytes_read     bytes actually requested from the file
    pages_touched  4 KB pages those requests cover (what the device must fetch)
    n_requests     number of pread calls
"""
import os
import sys

import numpy as np

PAGE_SIZE = 4096


class BucketReader:
    def __init__(self, path, starts, counts, d=128, page_size=PAGE_SIZE, nocache=True):
        self.path = str(path)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.d = d
        self.page_size = page_size
        if page_size % d != 0:
            raise ValueError("page size must be a multiple of the vector size")
        self.vec_per_page = page_size // d
        self.fd = os.open(self.path, os.O_RDONLY)
        self.file_size = os.fstat(self.fd).st_size
        self.nocache = nocache
        self._fadvise = False
        if nocache:
            if sys.platform == "darwin":
                import fcntl
                fcntl.fcntl(self.fd, getattr(fcntl, "F_NOCACHE", 48), 1)
            elif hasattr(os, "posix_fadvise"):
                self._fadvise = True
        self.reset_counters()

    # -- counters -----------------------------------------------------------
    def reset_counters(self):
        self.bytes_read = 0
        self.pages_touched = 0
        self.n_requests = 0

    def counters(self):
        return {"bytes_read": self.bytes_read, "pages_touched": self.pages_touched,
                "device_bytes": self.pages_touched * self.page_size, "n_requests": self.n_requests}

    # -- low level ------------------------------------------------------------
    def _pread(self, nbytes, offset):
        buf = os.pread(self.fd, nbytes, offset)
        if len(buf) != nbytes:
            raise IOError(f"short read at offset {offset}: {len(buf)} of {nbytes} bytes")
        if self._fadvise:
            os.posix_fadvise(self.fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
        self.bytes_read += nbytes
        first = offset // self.page_size
        last = (offset + nbytes - 1) // self.page_size
        self.pages_touched += last - first + 1
        self.n_requests += 1
        return buf

    # -- policies -------------------------------------------------------------
    def read_bucket(self, b):
        """All vectors of bucket b as (count, d) uint8, in one read."""
        c = int(self.counts[b])
        if c == 0:
            return np.empty((0, self.d), dtype=np.uint8)
        buf = self._pread(c * self.d, int(self.starts[b]) * self.d)
        return np.frombuffer(buf, dtype=np.uint8).reshape(c, self.d)

    def read_positions(self, positions):
        """Vectors at the given sorted positions, reading only the pages that hold them.

        positions: 1-D int array of global sorted positions (any order, may repeat).
        Returns (len(positions), d) uint8 in the same order as `positions`.
        Consecutive pages are merged into one request.
        """
        pos = np.asarray(positions, dtype=np.int64)
        out = np.empty((len(pos), self.d), dtype=np.uint8)
        if len(pos) == 0:
            return out
        pages = np.unique(pos // self.vec_per_page)
        breaks = np.flatnonzero(np.diff(pages) != 1) + 1
        runs = np.split(pages, breaks)
        page_buf = {}
        for run in runs:
            p0, p1 = int(run[0]), int(run[-1])
            off = p0 * self.page_size
            nbytes = min((p1 - p0 + 1) * self.page_size, self.file_size - off)  # last page may be partial
            buf = self._pread(nbytes, off)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(-1, self.d)
            page_buf[p0] = arr
        # map each position to its run
        run_first = np.array([int(r[0]) for r in runs], dtype=np.int64)
        page_of = pos // self.vec_per_page
        which = np.searchsorted(run_first, page_of, side="right") - 1
        for k, p0 in enumerate(run_first):
            sel = which == k
            if sel.any():
                out[sel] = page_buf[int(p0)][pos[sel] - p0 * self.vec_per_page]
        return out

    def read_bucket_rows(self, b, local_idx):
        """Rows `local_idx` (positions within bucket b) via the page policy."""
        return self.read_positions(int(self.starts[b]) + np.asarray(local_idx, dtype=np.int64))

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
