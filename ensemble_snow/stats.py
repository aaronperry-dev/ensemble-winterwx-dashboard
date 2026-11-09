"""Ensemble statistics: percentiles, mean, maximum and exceedance probabilities, optionally weighted (used by the combined multi-system view)."""
from __future__ import annotations

import warnings

import numpy as np

from . import config


def windowed(cum: np.ndarray, window_h: int, fhrs: list[int]) -> np.ndarray:
    """Totals over a rolling window from cumulative totals ``cum`` (T, M, ny, nx) valid at ``fhrs`` (the run starts with 0 snow at hour 0).
    ``window_h == 0`` means the run total.  Early in the run the window is simply everything since initialisation."""
    if window_h <= 0:
        return cum
    out = np.empty_like(cum)
    for i, f in enumerate(fhrs):
        target = f - window_h
        if target <= 0:
            out[i] = cum[i]
        else:
            j = int(np.argmin([abs(g - target) for g in fhrs]))
            if abs(fhrs[j] - target) > 1e-9:                        # not an output hour: cannot form the exact window
                out[i] = np.nan
            else:
                out[i] = cum[i] - cum[j]
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


def ensemble_stats(x: np.ndarray, weights: np.ndarray | None = None, percentiles=config.PERCENTILES, thresholds=config.THRESHOLDS_IN):
    """``x`` is (M, ...) (members first).  Returns (amounts, probabilities): amounts (len(percentiles)+2, ...) = percentiles, mean, max;
    probabilities (len(thresholds), ...) in percent.  NaN where no member has a value."""
    x = np.asarray(x, np.float32)
    valid = ~np.isnan(x)
    n = valid.sum(axis=0)
    if weights is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pct = np.nanpercentile(x, percentiles, axis=0, method="hazen")      # the same definition the weighted path uses
            mean = np.nanmean(x, axis=0)
            mx = np.nanmax(x, axis=0)
        prob = np.stack([100.0 * np.where(valid, x >= t - 1e-6, False).sum(axis=0) / np.maximum(n, 1) for t in thresholds])
    else:
        w = np.asarray(weights, np.float64).reshape((-1,) + (1,) * (x.ndim - 1)) * valid
        tot = w.sum(axis=0)
        pct = weighted_percentiles(x, np.asarray(weights, np.float64), percentiles)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (np.where(valid, x, 0.0) * w).sum(axis=0) / tot
            mx = np.where(n > 0, np.nanmax(np.where(valid, x, -np.inf), axis=0), np.nan)
            prob = np.stack([100.0 * (w * np.where(valid, x >= t - 1e-6, False)).sum(axis=0) / tot for t in thresholds])
    amounts = np.concatenate([np.asarray(pct), mean[None], mx[None]], axis=0).astype(np.float32)
    amounts = np.where(n[None] > 0, amounts, np.nan)
    prob = np.where(n[None] > 0, prob, np.nan).astype(np.float32)
    return amounts, prob
