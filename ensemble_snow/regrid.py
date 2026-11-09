"""Bilinear regridding of regular lat/lon fields onto the display grid, as two small matrix products (no interpolation library at run time)."""
from __future__ import annotations

import numpy as np

from . import config


def bilinear_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """(len(dst), len(src)) weights for 1-D linear interpolation; ``src`` ascending; destinations outside take the nearest edge value."""
    src = np.asarray(src, np.float64)
    dst = np.clip(np.asarray(dst, np.float64), src[0], src[-1])
    hi = np.clip(np.searchsorted(src, dst, side="right"), 1, len(src) - 1)
    lo = hi - 1
    w = (dst - src[lo]) / (src[hi] - src[lo])
    m = np.zeros((len(dst), len(src)))
    m[np.arange(len(dst)), lo] = 1.0 - w
    m[np.arange(len(dst)), hi] += w
    return m.astype(np.float32)


def crop_slices(lat: np.ndarray, lon: np.ndarray, pad: float = config.CROP_PAD_DEG):
    """Index slices (lat, lon) covering the display grid plus ``pad`` degrees; ``lat`` and ``lon`` ascending."""
    la0, la1 = config.GRID_LAT0 - pad, config.GRID_LAT1 + pad
    lo0, lo1 = config.GRID_LON0 - pad, config.GRID_LON1 + pad
    iy = np.nonzero((lat >= la0) & (lat <= la1))[0]
    ix = np.nonzero((lon >= lo0) & (lon <= lo1))[0]
    if iy.size < 2 or ix.size < 2:
        raise ValueError("model grid does not cover the display area")
    return slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)


class Regridder:
    """Maps ``(..., nlat, nlon)`` arrays on a source grid to the ``(..., ny, nx)`` display grid."""

    def __init__(self, src_lat: np.ndarray, src_lon: np.ndarray):
        dlon, dlat = config.grid_axes()
        self.src_lat, self.src_lon = np.asarray(src_lat), np.asarray(src_lon)
        self.wy = bilinear_matrix(self.src_lat, dlat)             # (ny, nlat)
        self.wx = bilinear_matrix(self.src_lon, dlon)             # (nx, nlon)

    def __call__(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, np.float32)
        return np.matmul(np.matmul(self.wy, a), self.wx.T).astype(np.float32)
