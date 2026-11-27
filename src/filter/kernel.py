"""Step 6: code-filter kernel (ADC + residual bound), compiled with Numba.

For one query x and a slice of PQ codes (e.g. one bucket), label every y:

    d_hat = sqrt( sum_m T[m, code_m] )          (ADC distance to y's reconstruction)
    Yes   : d_hat + r_y <= eps                 -> match, no disk read
    No    : d_hat - r_y >  eps                 -> discard, no disk read
    Maybe : otherwise                           -> read y from SSD and check exactly

Early exit: the partial ADC sum only grows, so once it exceeds (eps + r_y)^2 the
pair is certainly No and the remaining lookups are skipped.

Residuals stay float16 in RAM (20 MB for 10M). Numba cannot read float16 arrays,
so the kernel reads their raw bits (uint16 view) and decodes them exactly.

Usage:
    lut = build_lut(x, centroids)                       # once per query x
    yes_idx, maybe_idx = filter_codes(lut, codes, resid, eps)
    (indices are positions within the given codes slice)

Everything is float64 so the bound stays valid (see PLAN.md section 2).
"""
import numpy as np
from numba import njit


def build_lut(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """ADC lookup table T[m, j] = ||x_m - centroid[m, j]||^2, shape (M, 256), float64.

    x: (D,) any numeric dtype; centroids: (M, 256, D/M) float32 from data/pq/M{M}/centroids.npy.
    """
    m, k, dsub = centroids.shape
    xs = np.asarray(x, dtype=np.float64).reshape(m, 1, dsub)
    diff = xs - centroids.astype(np.float64)
    return np.ascontiguousarray(np.einsum("mkd,mkd->mk", diff, diff))


def _resid_bits(resid: np.ndarray) -> np.ndarray:
    """float16 residuals -> uint16 bit view (no copy if contiguous)."""
    r = np.asarray(resid)
    if r.dtype == np.float16:
        return np.ascontiguousarray(r).view(np.uint16)
    if r.dtype == np.uint16:
        return r
    # float32/float64 input: convert to float16 rounding UP so the bound stays valid
    r64 = r.astype(np.float64)
    r16 = r64.astype(np.float16)
    low = r16.astype(np.float64) < r64
    r16[low] = np.nextafter(r16[low], np.float16(np.inf))
    return r16.view(np.uint16)


@njit(cache=True, inline="always")
def _half_to_f64(h):
    """Exact IEEE half-precision -> float64."""
    sign = -1.0 if (h >> 15) & 1 else 1.0
    exp = (h >> 10) & 0x1F
    frac = h & 0x3FF
    if exp == 0:
        return sign * frac * 5.9604644775390625e-08  # 2**-24
    if exp == 31:
        return sign * np.inf if frac == 0 else np.nan
    return sign * (1024 + frac) * 2.0 ** (exp - 25)


@njit(cache=True, nogil=True)
def _filter(lut, codes, rbits, eps):
    n, m = codes.shape
    yes = np.empty(n, dtype=np.int64)
    maybe = np.empty(n, dtype=np.int64)
    n_yes = 0
    n_maybe = 0
    for i in range(n):
        r = _half_to_f64(rbits[i])
        cut = (eps + r) * (eps + r)  # partial sum above this -> certainly No
        s = 0.0
        pruned = False
        for j in range(m):
            s += lut[j, codes[i, j]]
            if s > cut:
                pruned = True
                break
        if pruned:
            continue
        d_hat = np.sqrt(s)
        if d_hat - r > eps:
            continue
        if d_hat + r <= eps:
            yes[n_yes] = i
            n_yes += 1
        else:
            maybe[n_maybe] = i
            n_maybe += 1
    return yes[:n_yes].copy(), maybe[:n_maybe].copy()


@njit(cache=True, nogil=True)
def _count(lut, codes, rbits, eps):
    n, m = codes.shape
    n_yes = 0
    n_maybe = 0
    for i in range(n):
        r = _half_to_f64(rbits[i])
        cut = (eps + r) * (eps + r)
        s = 0.0
        pruned = False
        for j in range(m):
            s += lut[j, codes[i, j]]
            if s > cut:
                pruned = True
                break
        if pruned:
            continue
        d_hat = np.sqrt(s)
        if d_hat - r > eps:
            continue
        if d_hat + r <= eps:
            n_yes += 1
        else:
            n_maybe += 1
    return n_yes, n_maybe, n - n_yes - n_maybe


def filter_codes(lut, codes, resid, eps):
    """Label every row of `codes` against one query.

    lut   : (M, 256) float64 from build_lut
    codes : (n, M) uint8 (contiguous)
    resid : (n,) float16 residuals (rounded up), or their uint16 view
    eps   : float, the distance threshold (sqrt of the integer eps2)

    Returns (yes_idx, maybe_idx): int64 positions within `codes`.
    For hot loops, pass resid already as a uint16 view to avoid a check per call.
    """
    return _filter(lut, np.ascontiguousarray(codes), _resid_bits(resid), float(eps))


def count_labels(lut, codes, resid, eps):
    """Same rule as filter_codes; returns (n_yes, n_maybe, n_no)."""
    return _count(lut, np.ascontiguousarray(codes), _resid_bits(resid), float(eps))


@njit(cache=True, nogil=True)
def _filter_query(lut, codes, rbits, starts, counts, cands, eps):
    """One query against all of its candidate buckets in one call.
    Returns (yes_pos, maybe_pos): GLOBAL sorted positions, ascending."""
    m = codes.shape[1]
    cap_y = 256
    cap_m = 256
    yes = np.empty(cap_y, dtype=np.int64)
    maybe = np.empty(cap_m, dtype=np.int64)
    ny = 0
    nm = 0
    for t in range(cands.shape[0]):
        b = cands[t]
        s0 = starts[b]
        for i in range(s0, s0 + counts[b]):
            r = _half_to_f64(rbits[i])
            cut = (eps + r) * (eps + r)
            s = 0.0
            pruned = False
            for j in range(m):
                s += lut[j, codes[i, j]]
                if s > cut:
                    pruned = True
                    break
            if pruned:
                continue
            d_hat = np.sqrt(s)
            if d_hat - r > eps:
                continue
            if d_hat + r <= eps:
                if ny == cap_y:
                    cap_y *= 2
                    tmp = np.empty(cap_y, dtype=np.int64)
                    tmp[:ny] = yes[:ny]
                    yes = tmp
                yes[ny] = i
                ny += 1
            else:
                if nm == cap_m:
                    cap_m *= 2
                    tmp = np.empty(cap_m, dtype=np.int64)
                    tmp[:nm] = maybe[:nm]
                    maybe = tmp
                maybe[nm] = i
                nm += 1
    return yes[:ny].copy(), maybe[:nm].copy()


def filter_query(lut, codes, rbits_u16, starts, counts, cands, eps):
    """Label one query against all codes of its candidate buckets.

    codes: (n, M) uint8 in bucket order (whole dataset, in RAM)
    rbits_u16: residuals as a uint16 view (np.float16 array .view(np.uint16))
    starts/counts: bucket bounds; cands: candidate bucket ids (int64)
    Returns (yes_pos, maybe_pos) as global sorted positions.
    """
    return _filter_query(lut, codes, rbits_u16, starts, counts, cands, float(eps))
