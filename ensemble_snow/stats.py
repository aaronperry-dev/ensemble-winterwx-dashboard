"""Ensemble statistics: percentiles, mean, maximum and exceedance probabilities, optionally weighted (used by the combined multi-system view)."""
from __future__ import annotations

import warnings

import numpy as np

from . import config


def window_valid(window_h: int, fhrs: list[int]) -> np.ndarray:
    """Boolean per output hour: is the ``window_h``-hour total complete there?  It needs the run to be at least that old and the hour
    ``window_h`` earlier to be an output hour (or the start of the run).  ``window_h == 0`` (run total) is always valid."""
    have = set(fhrs) | {0}
    return np.array([window_h <= 0 or (f - window_h >= 0 and (f - window_h) in have) for f in fhrs], bool)


def windowed(cum: np.ndarray, window_h: int, fhrs: list[int]) -> np.ndarray:
    """Totals over a rolling window from cumulative totals ``cum`` (T, M, ny, nx) valid at ``fhrs`` (the run starts with 0 snow at hour 0).
    ``window_h == 0`` means the run total.  Times at which the window is not complete (see :func:`window_valid`) are NaN."""
    if window_h <= 0:
        return cum
    out = np.full_like(cum, np.nan)
    index = {f: i for i, f in enumerate(fhrs)}
    for i, ok in enumerate(window_valid(window_h, fhrs)):
        if ok:
            start = fhrs[i] - window_h
            out[i] = cum[i] if start == 0 else cum[i] - cum[index[start]]
    return out


def weighted_percentiles(x: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    """Weighted percentiles along axis 0 of ``x`` (M, ...) with member weights ``w`` (M,).  NaN members are dropped per cell.

    Uses the weighted midpoint plotting positions (cumulative weight minus half the member's own weight, over the total), interpolating
    linearly between sorted values: the Hazen / type-5 definition, which equals ``np.percentile(..., method="hazen")`` for equal weights.
    """
    x = np.asarray(x, np.float64)
    m = x.shape[0]
    order = np.argsort(np.where(np.isnan(x), np.inf, x), axis=0, kind="stable")
    xs = np.take_along_axis(x, order, axis=0)
    valid = ~np.isnan(xs)
    ws = np.where(valid, np.asarray(w, np.float64)[order], 0.0)
    tot = ws.sum(axis=0)
    nvalid = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (np.cumsum(ws, axis=0) - 0.5 * ws) / tot
    out = np.full((len(qs),) + x.shape[1:], np.nan)
    for k, q in enumerate(qs):
        p = q / 100.0
        idx = np.clip(((pos < p) & valid).sum(axis=0), 0, np.maximum(nvalid - 1, 0))      # first sorted value at or beyond p
        lo_i = np.clip(idx - 1, 0, m - 1)
        hi = np.take_along_axis(xs, idx[None], 0)[0]
        lo = np.take_along_axis(xs, lo_i[None], 0)[0]
        p_hi = np.take_along_axis(pos, idx[None], 0)[0]
        p_lo = np.take_along_axis(pos, lo_i[None], 0)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(p_hi > p_lo, (p - p_lo) / (p_hi - p_lo), 1.0)
        v = np.where(idx == 0, hi, lo + np.clip(t, 0.0, 1.0) * (hi - lo))
        out[k] = np.where(nvalid > 0, v, np.nan)
    return out


def sorted_percentiles(x: np.ndarray, levels) -> np.ndarray:
    """Hazen percentiles (``levels`` in 0..100) along axis 0 of ``x`` (M, ...), NaN members ignored per cell.  Sorts once for all levels."""
    x = np.asarray(x, np.float32)
    xs = np.sort(x, axis=0)                                       # NaN sorts last
    n = (~np.isnan(x)).sum(axis=0)
    last = np.maximum(n - 1, 0)
    out = np.empty((len(levels),) + x.shape[1:], np.float32)
    for k, q in enumerate(levels):
        h = np.clip(n * (q / 100.0) + 0.5, 1.0, np.maximum(n, 1))        # 1-based virtual rank
        lo = np.floor(h).astype(np.intp) - 1
        frac = (h - np.floor(h)).astype(np.float32)
        a = np.take_along_axis(xs, lo[None], 0)[0]
        b = np.take_along_axis(xs, np.minimum(lo + 1, last)[None], 0)[0]
        out[k] = a + frac * (b - a)
    return np.where(n[None] > 0, out, np.nan)


def ensemble_stats(x: np.ndarray, weights: np.ndarray | None = None, thresholds=config.THRESHOLDS_IN):
    """``x`` is (M, ...) (members first).  Returns (amounts, probabilities): amounts (len(config.PCT_STATS), ...) in the order of
    ``config.PCT_STATS`` (min, 5th ... 95th percentiles, mean, max); probabilities (len(thresholds), ...) in percent.
    NaN where no member has a value."""
    x = np.asarray(x, np.float32)
    valid = ~np.isnan(x)
    n = valid.sum(axis=0)
    pct_names = [s for s in config.PCT_STATS if s != "mean"]
    levels = [config.PCT_LEVELS[s] for s in pct_names]
    if weights is None:
        pct = sorted_percentiles(x, levels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(x, axis=0)
        prob = np.stack([100.0 * np.where(valid, x >= t - 1e-6, False).sum(axis=0) / np.maximum(n, 1) for t in thresholds])
    else:
        w = np.asarray(weights, np.float64).reshape((-1,) + (1,) * (x.ndim - 1)) * valid
        tot = w.sum(axis=0)
        pct = weighted_percentiles(x, np.asarray(weights, np.float64), levels)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (np.where(valid, x, 0.0) * w).sum(axis=0) / tot
            prob = np.stack([100.0 * (w * np.where(valid, x >= t - 1e-6, False)).sum(axis=0) / tot for t in thresholds])
    by = dict(zip(pct_names, pct))
    by["mean"] = mean
    amounts = np.stack([np.asarray(by[s], np.float32) for s in config.PCT_STATS])
    amounts = np.where(n[None] > 0, amounts, np.nan)
    prob = np.where(n[None] > 0, prob, np.nan).astype(np.float32)
    return amounts, prob
