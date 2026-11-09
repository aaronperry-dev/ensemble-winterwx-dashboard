"""Consensus snowfall products that keep realistic amounts.

The plain ensemble mean smooths snowfall: where members put their snow in different places, the mean is a wide, weak swath and the
heaviest amounts any member produced disappear.  The products here repair that, and all of them take the arrays of one forecast time,
``x`` of shape ``(members, ny, nx)`` with NaN for missing data, plus optional member weights.

PMM    probability-matched mean (Ebert 2001).  The *pattern* is the ensemble mean; the *amounts* are the pooled member values over the whole
       map.  The cell with the n-th largest mean receives the value of the same rank in the pool of every member's every cell.
LPMM   local PMM (Clark 2017; Snook et al. 2020).  The same, patch by patch: each patch takes its amounts from the pool of the members'
       values in the 3 x 3 patches around it, so snow from one storm is not given to another, far-away one.
SAM    spatially aligned mean (Lee et al. 2024, Weather and Forecasting 39, 1545-1558).  Each member is first moved toward the PMM by a
       smooth displacement field found in two passes (big patches for synoptic offsets, then small ones), its own distribution of values is
       restored (moving a field by interpolation smooths it), and the moved members are averaged.
SAM-LPM  the aligned members combined by LPM instead of averaged.

Everything is vectorised over members; the display grid is only ~41 x 33 cells, so a whole ensemble is processed in milliseconds.
"""
from __future__ import annotations

import numpy as np

from . import config


# --------------------------------------------------------------------------- helpers
def weighted_mean(x: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """Mean over axis 0, NaN members ignored per cell (NaN where there is none)."""
    valid = ~np.isnan(x)
    if w is None:
        n = valid.sum(axis=0)
        s = np.where(valid, x, 0.0).sum(axis=0, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
    ww = np.asarray(w, np.float64).reshape((-1,) + (1,) * (x.ndim - 1)) * valid
    tot = ww.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (np.where(valid, x, 0.0) * ww).sum(axis=0) / tot, np.nan).astype(np.float32)


def _tie_rank(sorted_ref: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rank of each ``v`` among ``sorted_ref`` (0-based; the middle of a run of equal values)."""
    return (np.searchsorted(sorted_ref, v, "left") + np.searchsorted(sorted_ref, v, "right") - 1) / 2.0


def _matched(query: np.ndarray, ref: np.ndarray, pool: np.ndarray, pool_w: np.ndarray | None) -> np.ndarray:
    """Values for ``query`` (finite mean values): the pooled value at the quantile that each one has among ``ref`` (the means of the domain)."""
    if query.size == 0:
        return query
    if pool.size == 0:
        return np.full(query.shape, np.nan, np.float32)
    lo, hi = float(pool.min()), float(pool.max())
    if lo == hi:                                                   # flat neighbourhood (usually dry): nothing to rank
        return np.full(query.shape, lo, np.float32)
    q = (_tie_rank(np.sort(ref), query) + 0.5) / ref.size
    if pool_w is None:
        sp = np.sort(pool)
        return sp[np.minimum(pool.size - 1, np.floor(q * pool.size).astype(np.intp))]
    order = np.argsort(pool, kind="stable")
    cw = np.cumsum(pool_w[order])
    cw = cw / cw[-1]
    return pool[order][np.minimum(pool.size - 1, np.searchsorted(cw, q, "left"))]


def _pool(x: np.ndarray, w: np.ndarray | None):
    """Finite values of ``x`` (M, ...) as a flat pool and, with member weights, the weight of each."""
    ok = ~np.isnan(x)
    vals = x[ok]
    if w is None:
        return vals, None
    return vals, np.broadcast_to(np.asarray(w, np.float64).reshape((-1,) + (1,) * (x.ndim - 1)), x.shape)[ok]


# --------------------------------------------------------------------------- PMM / LPMM
def pmm(x: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Probability-matched mean over the whole map."""
    mean = weighted_mean(x, w)
    ok = ~np.isnan(mean)
    out = np.full(mean.shape, np.nan, np.float32)
    if ok.any():
        pool, pw = _pool(x, w)
        out[ok] = _matched(mean[ok], mean[ok], pool, pw)
    return out


def lpmm(x: np.ndarray, w: np.ndarray | None = None, patch: int = config.LPMM_PATCH_CELLS) -> np.ndarray:
    """Local probability-matched mean: ``patch`` x ``patch`` cells receive values from the pool of the 3 x 3 patches around them."""
    _, ny, nx = x.shape
    mean = weighted_mean(x, w)
    out = np.full((ny, nx), np.nan, np.float32)
    for py in range(0, ny, patch):
        for px in range(0, nx, patch):
            y0, y1, x0, x1 = max(0, py - patch), min(ny, py + 2 * patch), max(0, px - patch), min(nx, px + 2 * patch)
            dom = mean[y0:y1, x0:x1]
            dok = ~np.isnan(dom)
            if not dok.any():
                continue
            cell = (slice(py, min(ny, py + patch)), slice(px, min(nx, px + patch)))
            mine = mean[cell]
            mok = ~np.isnan(mine)
            if not mok.any():
                continue
            xd = x[:, y0:y1, x0:x1]
            if not (xd > 0).any():                                 # a dry neighbourhood: every rank is zero
                out[cell] = np.where(mok, 0.0, np.nan)
                continue
            pool, pw = _pool(xd, w)
            block = np.full(mine.shape, np.nan, np.float32)
            block[mok] = _matched(mine[mok], dom[dok], pool, pw)
            out[cell] = block
    return out


# --------------------------------------------------------------------------- SAM
def _transform(a: np.ndarray, floor: float) -> np.ndarray:
    """Square root of the amount above the floor (NaN counts as 0): even weight for light and heavy snow in the matching."""
    return np.sqrt(np.clip(np.nan_to_num(a, nan=0.0) - floor, 0.0, None)).astype(np.float32)


def restore(moved: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """Give each member ``moved[m]`` the distribution of values of ``orig`` (or ``orig[m]``): the value of rank r becomes the original
    value of rank r.  Undoes the smoothing that moving a field by interpolation causes.  NaN stays NaN."""
    out = np.full(moved.shape, np.nan, np.float32)
    shared = orig.ndim == moved.ndim - 1
    for m in range(moved.shape[0]):
        a = moved[m]
        o = orig if shared else orig[m]
        ao, oo = ~np.isnan(a), ~np.isnan(o)
        if not ao.any() or not oo.any():
            continue
        sm, so = np.sort(a[ao]), np.sort(o[oo])
        idx = np.minimum(so.size - 1, np.floor((_tie_rank(sm, a[ao]) + 0.5) * so.size / sm.size).astype(np.intp))
        out[m][ao] = so[idx]
    return out


def _bilinear(a: np.ndarray, xs: np.ndarray, ys: np.ndarray, nan_aware: bool = False) -> np.ndarray:
    """Sample ``a`` (M, ny, nx) at fractional cell coordinates ``xs``, ``ys`` (M, ny, nx); the edge value continues outside the grid.
    With ``nan_aware`` missing neighbours are left out of the weighting (NaN only where all four are missing)."""
    m, ny, nx = a.shape
    xs = np.clip(xs, 0, nx - 1)
    ys = np.clip(ys, 0, ny - 1)
    x0, y0 = np.floor(xs).astype(np.intp), np.floor(ys).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
    fx, fy = (xs - x0).astype(np.float32), (ys - y0).astype(np.float32)
    flat = a.reshape(m, -1)

    def take(yy, xx):
        return np.take_along_axis(flat, (yy * nx + xx).reshape(m, -1), 1).reshape(a.shape)

    corners = ((take(y0, x0), (1 - fx) * (1 - fy)), (take(y0, x1), fx * (1 - fy)), (take(y1, x0), (1 - fx) * fy), (take(y1, x1), fx * fy))
    if not nan_aware:
        return sum(v * k for v, k in corners).astype(np.float32)
    num = sum(np.where(np.isnan(v), 0.0, v) * k for v, k in corners)
    den = sum(np.where(np.isnan(v), 0.0, k) for v, k in corners)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan).astype(np.float32)


def _soar(r: np.ndarray) -> np.ndarray:
    """Second-order autoregressive penalty on the length of a shift (Thiebaux et al. 1990), as in Lee et al. (2024) eq. 2."""
    return np.exp(r) / (1.0 + r)


def _centres(n: int, step: int) -> np.ndarray:
    return np.arange(0, n + step, step)


def _boxes(n: int, centres: np.ndarray, half: float):
    lo = np.maximum(0, np.floor(centres - half + 0.5).astype(int))
    hi = np.minimum(n, np.floor(centres + half + 0.5).astype(int))
    keep = hi > lo
    return lo[keep], hi[keep], centres[keep]


def _pass(src: np.ndarray, tgt: np.ndarray, patch: int, rng_cells: int, alpha: float, eps: float):
    """One alignment pass: the displacement field (cells) that moves each member of ``src`` onto ``tgt`` (Lee et al. 2024, eqs. 1-3).

    In every overlapping patch the shift in the search window that minimises the penalised squared difference is found (and refined to a
    fraction of a cell by a parabola through the neighbouring costs); patch shifts are blended into a smooth field by tent weights.
    Dry patches of the target are left where they are."""
    m, ny, nx = src.shape
    step = max(1, patch // 2)
    half = patch / 2.0
    ylo, yhi, yc = _boxes(ny, _centres(ny, step), half)
    xlo, xhi, xc = _boxes(nx, _centres(nx, step), half)
    npy, npx, r = len(yc), len(xc), rng_cells
    w = 2 * r + 1
    lam = alpha * np.sqrt(2.0) * patch
    pen = _soar(np.hypot(*np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))) / lam)             # (w, w): [dy, dx]
    padded = np.pad(src, ((0, 0), (r, r), (r, r)), mode="edge")
    cost = np.empty((m, npy, npx, w, w), np.float64)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = padded[:, r + dy:r + dy + ny, r + dx:r + dx + nx] - tgt
            ii = np.zeros((m, ny + 1, nx + 1), np.float64)
            ii[:, 1:, 1:] = np.cumsum(np.cumsum(d * d, axis=1, dtype=np.float64), axis=2)
            ssd = ii[:, yhi[:, None], xhi[None, :]] - ii[:, ylo[:, None], xhi[None, :]] - ii[:, yhi[:, None], xlo[None, :]] + ii[:, ylo[:, None], xlo[None, :]]
            cost[:, :, :, dy + r, dx + r] = ssd * pen[dy + r, dx + r]
    flat = cost.reshape(m, npy, npx, w * w)
    k = flat.argmin(axis=-1)
    bdy, bdx = k // w - r, k % w - r
    best = np.take_along_axis(flat, k[..., None], -1)[..., 0]

    def sub(minus, plus, inside):
        c_minus = np.take_along_axis(flat, np.clip(k + minus, 0, w * w - 1)[..., None], -1)[..., 0]
        c_plus = np.take_along_axis(flat, np.clip(k + plus, 0, w * w - 1)[..., None], -1)[..., 0]
        d2 = c_minus - 2 * best + c_plus
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(d2 > 1e-12, np.clip(0.5 * (c_minus - c_plus) / d2, -0.5, 0.5), 0.0)
        return np.where(inside, t, 0.0)

    bx = bdx + sub(-1, 1, np.abs(bdx) < r)                           # parabola through three exact costs: sub-cell shift
    by = bdy + sub(-w, w, np.abs(bdy) < r)
    wet = np.array([[tgt[ylo[i]:yhi[i], xlo[j]:xhi[j]].max() >= eps for j in range(npx)] for i in range(npy)])
    bx, by = np.where(wet[None], bx, 0.0), np.where(wet[None], by, 0.0)
    wy = np.maximum(0.0, 1.0 - np.abs(np.arange(ny)[:, None] - yc[None, :]) / step)               # (ny, npy)
    wx = np.maximum(0.0, 1.0 - np.abs(np.arange(nx)[:, None] - xc[None, :]) / step)               # (nx, npx)
    den = np.maximum(wy.sum(1)[:, None] * wx.sum(1)[None, :], 1e-6)                                # dry patches count as a zero shift
    sx = np.einsum("yp,mpq,xq->myx", wy, bx, wx) / den
    sy = np.einsum("yp,mpq,xq->myx", wy, by, wx) / den
    return sx.astype(np.float32), sy.astype(np.float32)


def find_shifts(x: np.ndarray, target: np.ndarray, passes=None, alpha: float = config.SAM_ALPHA, floor: float = config.SAM_FLOOR_IN) -> np.ndarray:
    """Displacement field of every member toward ``target`` (the probability-matched mean): array (M, 2, ny, nx) of (x, y) shifts in cells,
    meaning the aligned member at cell p is the member at p + shift.  Two passes, large patches then small ones, composed."""
    passes = config.SAM_PASSES if passes is None else passes
    m, ny, nx = x.shape
    out = np.zeros((m, 2, ny, nx), np.float32)
    t = _transform(target, floor)
    if not (t > 0).any():
        return out
    eps = float(np.sqrt(floor))
    c = restore(_transform(x, floor), t)                  # the member takes the target's distribution first, so a bias in amount does not drive the shifts
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    gx, gy = np.broadcast_to(gx, (m, ny, nx)), np.broadcast_to(gy, (m, ny, nx))
    tx, ty, cur = np.zeros((m, ny, nx), np.float32), np.zeros((m, ny, nx), np.float32), c
    for patch, rng_cells in passes:
        fx, fy = _pass(cur, t, patch, rng_cells, alpha, eps)
        ax, ay = gx + fx, gy + fy
        tx, ty = fx + _bilinear(tx, ax, ay), fy + _bilinear(ty, ax, ay)          # total(x) = s2(x) + s1(x + s2(x))
        cur = _bilinear(c, gx + tx, gy + ty)
    out[:, 0], out[:, 1] = tx, ty
    return out


def align(x: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Move every member by its displacement field and restore its own distribution of values (peaks the interpolation smoothed away)."""
    m, ny, nx = x.shape
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    moved = _bilinear(x, gx + shifts[:, 0], gy + shifts[:, 1], nan_aware=True)
    return restore(moved, x)


# --------------------------------------------------------------------------- one forecast time
def consensus_shifts(x: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Displacement fields of the members of ``x`` toward their PMM.  Found once per time on one field (the multi-method snowfall) and then
    applied to every method's snowfall, so that the products of all methods align the same way."""
    if np.isnan(x).all() or not (np.nan_to_num(x) > config.SAM_FLOOR_IN).any():
        return np.zeros((x.shape[0], 2) + x.shape[1:], np.float32)
    return find_shifts(x, pmm(x, w))


def products(x: np.ndarray, shifts: np.ndarray, w: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """``{"pmm", "lpmm", "sam", "samlpm"}`` fields (ny, nx) for the members ``x`` (M, ny, nx) and their ``shifts``."""
    ny, nx = x.shape[1:]
    if np.isnan(x).all():
        return {k: np.full((ny, nx), np.nan, np.float32) for k in config.CONSENSUS_STATS}
    if not (np.nan_to_num(x) > 0).any():                                # nothing falls in any member: every product is zero where defined
        z = np.where(np.isnan(weighted_mean(x, w)), np.nan, 0.0).astype(np.float32)
        return {k: z.copy() for k in config.CONSENSUS_STATS}
    aligned = align(x, shifts) if shifts.any() else x
    out = {"pmm": pmm(x, w), "lpmm": lpmm(x, w), "sam": weighted_mean(aligned, w), "samlpm": lpmm(aligned, w)}
    undefined = np.isnan(weighted_mean(x, w))                           # no member has data here: moving the others must not invent any
    return {k: np.where(undefined, np.nan, v).astype(np.float32) for k, v in out.items()}
