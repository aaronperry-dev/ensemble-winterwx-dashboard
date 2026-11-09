"""A cheap terrain adjustment: how much higher (or lower) the real ground is than the smooth terrain a model sees.

The models are 0.25 deg (about 28 x 20 km) or 0.5 deg here, so a mountain range is a gentle rise and a valley floor is the same height as its
ridges.  Snowfall on the ground is heavier up the slopes, so the viewer can scale the model's amounts by the height of each ~5 km cell above the
model's own terrain.  This module makes the two ingredients, both from open data:

    elevation   mean ground height (m) of 0.05 deg cells, from the open "Terrain Tiles" (Mapzen Terrarium encoding on AWS, SRTM / 3DEP based)
    relief      elevation minus the same elevation smoothed to the model's resolution: what the model cannot see (m)

The adjustment itself (a percentage per 100 m, chosen by the user) is applied in the browser.  Nothing here changes the model data.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 8                      # ~440 m pixels at this latitude: plenty to average into 5 km cells
STEP = 0.05                   # deg: 5 cells per display cell, so the display grid nodes are nodes of this grid too
# Resolution of the terrain each system sees.  The 0.25 deg surface fields of EPS and GEFS, the 0.5 deg files of GEPS; the pooled view is on 0.25.
SMOOTH_DEG = {"eps": 0.25, "gefs": 0.25, "geps": 0.5, "all": 0.25}


def fine_axes() -> tuple[np.ndarray, np.ndarray]:
    lon = np.round(np.arange(config.GRID_LON0, config.GRID_LON1 + STEP / 2, STEP), 6)
    lat = np.round(np.arange(config.GRID_LAT0, config.GRID_LAT1 + STEP / 2, STEP), 6)
    return lon, lat


# --------------------------------------------------------------------------- web-mercator tiles
def _tile_x(lon: float, z: int) -> float:
    return (lon + 180.0) / 360.0 * 2 ** z


def _tile_y(lat: float, z: int) -> float:
    r = np.radians(lat)
    return (1.0 - np.arcsinh(np.tan(r)) / np.pi) / 2.0 * 2 ** z


def tile_range(z: int = ZOOM, margin: float = 0.1) -> tuple[range, range]:
    x0, x1 = _tile_x(config.GRID_LON0 - margin, z), _tile_x(config.GRID_LON1 + margin, z)
    y0, y1 = _tile_y(config.GRID_LAT1 + margin, z), _tile_y(config.GRID_LAT0 - margin, z)
    return range(int(np.floor(x0)), int(np.floor(x1)) + 1), range(int(np.floor(y0)), int(np.floor(y1)) + 1)


def decode_terrarium(png: bytes) -> np.ndarray:
    """(256, 256) metres from a Terrarium PNG: height = R * 256 + G + B / 256 - 32768."""
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float64)
    return (a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0).astype(np.float32)


def _pixel_lonlat(x: int, y: int, z: int) -> tuple[np.ndarray, np.ndarray]:
    n = 256 * 2 ** z
    px = (x * 256 + np.arange(256) + 0.5) / n
    py = (y * 256 + np.arange(256) + 0.5) / n
    lon = px * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * py))))
    return lon, lat


class _Binner:
    """Mean of all tile pixels that fall in each 0.05 deg cell."""

    def __init__(self):
        self.lon, self.lat = fine_axes()
        self.sum = np.zeros((len(self.lat), len(self.lon)))
        self.n = np.zeros_like(self.sum)

    def add(self, h: np.ndarray, plon: np.ndarray, plat: np.ndarray) -> None:
        ix = np.floor((plon - self.lon[0]) / STEP + 0.5).astype(int)
        iy = np.floor((plat - self.lat[0]) / STEP + 0.5).astype(int)
        okx, oky = (ix >= 0) & (ix < len(self.lon)), (iy >= 0) & (iy < len(self.lat))
        h = np.maximum(h, 0.0)                                             # the tiles carry sea-floor depth too: sea level is 0 here
        cell = (iy[oky][:, None] * len(self.lon) + ix[okx][None, :]).ravel()
        vals = h[np.ix_(oky, okx)].ravel()
        m = self.sum.size
        self.sum += np.bincount(cell, weights=vals, minlength=m).reshape(self.sum.shape)
        self.n += np.bincount(cell, minlength=m).reshape(self.sum.shape)

    def result(self) -> np.ndarray:
        return np.where(self.n > 0, self.sum / np.maximum(self.n, 1), 0.0).astype(np.float32)


def fetch_elevation(cache: Path, fetcher=None, workers: int = 12) -> np.ndarray:
    """Mean elevation (m) on the 0.05 deg grid; downloaded once and kept in ``cache``."""
    path = cache / "dem" / f"elev_z{ZOOM}_{STEP}_{config.GRID_LON0}_{config.GRID_LAT0}.npy"
    if path.exists():
        return np.load(path)
    if fetcher is None:
        from .http import Fetcher
        fetcher = Fetcher(workers=workers)
    xs, ys = tile_range()
    binner = _Binner()

    def get(xy):
        x, y = xy
        try:
            return xy, fetcher.get(TILE_URL.format(z=ZOOM, x=x, y=y))
        except Exception as exc:                                           # a missing tile is open sea or outside the data: height 0
            log.warning("terrain tile %d/%d/%d unavailable (%s)", ZOOM, x, y, exc)
            return xy, None

    got = 0
    with ThreadPoolExecutor(workers) as ex:
        for (x, y), png in ex.map(get, [(x, y) for x in xs for y in ys]):
            if png is None:
                continue
            plon, plat = _pixel_lonlat(x, y, ZOOM)
            binner.add(decode_terrarium(png), plon, plat)
            got += 1
    if got < 0.8 * len(xs) * len(ys):
        raise RuntimeError(f"only {got} of {len(xs) * len(ys)} terrain tiles could be downloaded")
    elev = binner.result()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, elev)
    log.info("terrain: %d tiles -> %dx%d cells, highest cell %.0f m", got, elev.shape[1], elev.shape[0], elev.max())
    return elev


def synthetic_elevation() -> np.ndarray:
    from .sources.synthetic import SyntheticSource
    lon, lat = fine_axes()
    return SyntheticSource._elevation_m(*np.meshgrid(lon, lat)).astype(np.float32)


# --------------------------------------------------------------------------- relief
def relief(elev: np.ndarray, smooth_deg: float) -> np.ndarray:
    """Elevation minus the elevation averaged over a box of the model's grid spacing (m, float)."""
    from scipy.ndimage import uniform_filter
    size = max(1, int(round(smooth_deg / STEP)))
    return elev - uniform_filter(elev.astype(np.float64), size=size, mode="nearest")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(np.round(a), dtype="<i2").tobytes()).decode("ascii")


def build(data_dir: Path, cache: Path, synthetic: bool = False, fetcher=None) -> dict | None:
    """Write ``terrain.json`` next to the model data and return its manifest entry (None, with a log line, when no elevation data is available:
    the viewer then simply has no terrain option)."""
    try:
        elev = synthetic_elevation() if synthetic else fetch_elevation(cache, fetcher)
        lon, lat = fine_axes()
        smooth = sorted(set(SMOOTH_DEG.values()))
        doc = {"v": 1, "grid": {"lon0": float(lon[0]), "lat0": float(lat[0]), "dlon": STEP, "dlat": STEP, "nx": len(lon), "ny": len(lat)},
               "elev": _b64(elev), "dz": {f"{s:g}": _b64(relief(elev, s)) for s in smooth}}
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "terrain.json").write_text(json.dumps(doc, separators=(",", ":")))
        return {"file": "terrain.json", "step": STEP, "smooth": {k: f"{v:g}" for k, v in SMOOTH_DEG.items()},
                "source": "synthetic hills" if synthetic else "Terrain Tiles (AWS Open Data; SRTM, 3DEP and other open elevation sources)"}
    except Exception:
        log.exception("terrain adjustment data could not be built; the site goes ahead without it")
        return None
