"""Shaded relief for the basemap: a hillshade of the open "Terrain Tiles" (Mapzen Terrarium on AWS Open Data), as one transparent image.

The image is in Web Mercator pixels (the map's own projection), so the viewer lays it on the map between its corner coordinates and it lines up at every zoom.
Where the ground faces away from the light it is black with some opacity, where it faces the light white with a little, and flat ground (and the sea) is
transparent, so the image darkens and lightens whatever colour is under it, whether that is the model field or the plain land.  Nothing here changes model data.
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config, terrain

log = logging.getLogger(__name__)

ZOOM = 8                       # ~440 m pixels at 44 deg N
PAD_DEG = 1.5                  # the image reaches this far past the model grid (the map shows a little more land than the data cover)
AZIMUTH, ALTITUDE = 315.0, 40.0          # light from the north-west, as on printed maps, 40 deg above the horizon
EXAGGERATION = 2.0             # vertical exaggeration: New England's slopes are gentle at this scale
GAIN_DARK, GAIN_LIGHT = 1.9, 1.1         # opacity per unit of shade difference from flat ground
MAX_ALPHA = 0.8                # fraction of 255; the viewer scales it further with its slider
WEBP_QUALITY = 78
EARTH_CIRC_M = 40075016.686


def _tile_edges(x0: int, x1: int, y0: int, y1: int, z: int) -> tuple[float, float, float, float]:
    """(south, west, north, east) of a block of tiles, in degrees."""
    n = 2 ** z
    lat = lambda y: float(np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))))
    return lat(y1 + 1), x0 / n * 360 - 180, lat(y0), (x1 + 1) / n * 360 - 180


def synthetic_mosaic(shape: tuple[int, int], bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Demo terrain: the synthetic hills plus fractal roughness, evaluated on the mosaic's own pixels."""
    from .sources.synthetic import SyntheticSource
    h, w = shape
    s, we, n, e = bounds
    px = (np.arange(w) + 0.5) / w
    my = (np.arange(h) + 0.5) / h
    lon = we + (e - we) * px
    y_top = np.arcsinh(np.tan(np.radians(n))); y_bot = np.arcsinh(np.tan(np.radians(s)))
    lat = np.degrees(np.arctan(np.sinh(y_top + (y_bot - y_top) * my)))
    elev = SyntheticSource._elevation_m(*np.meshgrid(lon, lat)).astype(np.float64)
    rng = np.random.default_rng(7)
    from scipy.ndimage import zoom
    rough = sum(zoom(rng.normal(size=(max(2, h // k), max(2, w // k))), (h / max(2, h // k), w / max(2, w // k)), order=3)[:h, :w] * a for k, a in ((256, 250), (64, 220), (16, 140), (6, 60)))
    return np.maximum(elev * (0.6 + 0.4 * np.clip(rough / 250 + 1, 0, 2)) + np.where(elev > 20, rough * 0.5, 0), 0).astype(np.float32)


def fetch_mosaic(cache: Path, fetcher=None, workers: int = 12, synthetic: bool = False) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Heights (m, sea level 0) on a Web Mercator pixel grid of tiles, and the (south, west, north, east) edges of that grid."""
    xs, ys = terrain.tile_range(ZOOM, PAD_DEG)
    bounds = _tile_edges(xs[0], xs[-1], ys[0], ys[-1], ZOOM)
    shape = (len(ys) * 256, len(xs) * 256)
    if synthetic:
        return synthetic_mosaic(shape, bounds), bounds
    path = cache / "dem" / f"mosaic_z{ZOOM}_{xs[0]}_{ys[0]}_{len(xs)}x{len(ys)}.npy"
    if path.exists():
        return np.load(path).astype(np.float32), bounds
    if fetcher is None:
        from .http import Fetcher
        fetcher = Fetcher(workers=workers)
    mosaic = np.zeros(shape, np.float32)
    got = 0

    def get(xy):
        return xy, terrain.get_tile(fetcher, ZOOM, *xy)

    with ThreadPoolExecutor(workers) as ex:
        for (x, y), png in ex.map(get, [(x, y) for x in xs for y in ys]):
            if png is None:
                continue
            mosaic[(y - ys[0]) * 256:(y - ys[0] + 1) * 256, (x - xs[0]) * 256:(x - xs[0] + 1) * 256] = np.maximum(terrain.decode_terrarium(png), 0)    # sea floor depth is not relief
            got += 1
    if got < 0.8 * len(xs) * len(ys):
        raise RuntimeError(f"only {got} of {len(xs) * len(ys)} terrain tiles could be downloaded")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.round(mosaic).astype(np.int16))
    log.info("relief: %d tiles -> %dx%d pixels, highest %.0f m", got, shape[1], shape[0], mosaic.max())
    return mosaic, bounds


def shade(height: np.ndarray, bounds: tuple[float, float, float, float], z: int = ZOOM) -> np.ndarray:
    """Shade of every pixel, 0..1 (flat ground = sin(altitude)): the unit surface normal dotted with the direction of the light."""
    from scipy.ndimage import gaussian_filter
    h = gaussian_filter(height.astype(np.float64), 0.8)                       # the source has small-scale noise that would read as grain
    s, _, n, _ = bounds
    ymerc = np.arcsinh(np.tan(np.radians(n))) + (np.arcsinh(np.tan(np.radians(s))) - np.arcsinh(np.tan(np.radians(n)))) * (np.arange(h.shape[0]) + 0.5) / h.shape[0]
    lat = np.arctan(np.sinh(ymerc))
    res = (EARTH_CIRC_M / (256 * 2 ** z)) * np.cos(lat)[:, None]               # metres on the ground per pixel: Mercator stretches both axes by 1 / cos(lat)
    dz_dy_south, dz_dx = np.gradient(h)[0], np.gradient(h)[1]
    dzdx, dzdy = EXAGGERATION * dz_dx / res, -EXAGGERATION * dz_dy_south / res   # x east, y north
    az, alt = np.radians(AZIMUTH), np.radians(ALTITUDE)
    light = (np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt))
    return (-dzdx * light[0] - dzdy * light[1] + light[2]) / np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)


def to_rgba(height: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    d = shade(height, bounds) - np.sin(np.radians(ALTITUDE))
    alpha = np.where(d < 0, np.minimum(1.0, -d * GAIN_DARK), np.minimum(1.0, d * GAIN_LIGHT)) * MAX_ALPHA
    alpha = np.where(height > 0.5, alpha, 0.0)                                    # open water stays clear
    out = np.zeros(height.shape + (4,), np.uint8)
    out[..., :3] = np.where(d < 0, 0, 255)[..., None]
    out[..., 3] = np.round(alpha * 255).astype(np.uint8)
    return out


def build(data_dir: Path, cache: Path, synthetic: bool = False, fetcher=None) -> dict | None:
    """Write ``relief.webp`` next to the model data and return its manifest entry (None, with a log line, when no elevation data is available:
    the viewer then simply has no relief layer)."""
    try:
        from PIL import Image
        height, bounds = fetch_mosaic(cache, fetcher, synthetic=synthetic)
        img = Image.fromarray(to_rgba(height, bounds), "RGBA")
        data_dir.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        img.save(buf, "WEBP", quality=WEBP_QUALITY, alpha_quality=85, method=4)
        (data_dir / "relief.webp").write_bytes(buf.getvalue())
        log.info("relief.webp  %dx%d  %.0f kB", img.width, img.height, len(buf.getvalue()) / 1e3)
        return {"file": "relief.webp", "bounds": [round(b, 5) for b in bounds], "bytes": len(buf.getvalue()),
                "source": "synthetic hills" if synthetic else "Terrain Tiles (AWS Open Data; SRTM, 3DEP and other open elevation sources)"}
    except Exception:
        log.exception("shaded relief could not be built; the site goes ahead without it")
        return None
