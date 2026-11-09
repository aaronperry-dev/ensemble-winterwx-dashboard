"""Static map layers for the page: Natural Earth (public domain) clipped to the display area, simplified and written as compact GeoJSON / JSON.

Outputs in ``<out>/gis``:  land, lakes, states (state / province lines), counties (de-duplicated boundary lines), roads (interstates, US routes) with
route labels, places (cities and towns), peaks, ski areas.  Every file is small enough to be fetched on demand when its switch is turned on.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

import requests
import shapefile
import shapely
from shapely.geometry import box, shape

from . import config

log = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent / "data"
MIRROR = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/{cat}/ne_10m_{name}.{ext}"
LAYERS = {"land": "10m_physical", "lakes": "10m_physical", "admin_1_states_provinces_lines": "10m_cultural",
          "admin_2_counties": "10m_cultural", "roads_north_america": "10m_cultural"}
NE_NAMES = {"Maine", "New Hampshire", "Vermont", "Massachusetts", "Connecticut", "Rhode Island"}
ABBR = {"Maine": "ME", "New Hampshire": "NH", "Vermont": "VT", "Massachusetts": "MA", "Connecticut": "CT", "Rhode Island": "RI",
        "New York": "NY", "New Jersey": "NJ"}
# minimum Leaflet zoom at which a town's dot and name appear, by population
TOWN_ZOOM = ((100_000, 6), (25_000, 8), (10_000, 9), (5_000, 10), (2_500, 11), (0, 12))
WIDE_PAD_DEG = 4.0
ROUTE_ZOOM = {"interstate": 7, "us": 9}


def display_box(pad: float = 0.5):
    return box(config.GRID_LON0 - pad, config.GRID_LAT0 - pad, config.GRID_LON1 + pad, config.GRID_LAT1 + pad)


def download(name: str, cache: Path) -> Path:
    cat = LAYERS[name]
    cache.mkdir(parents=True, exist_ok=True)
    for ext in ("shp", "dbf", "shx"):
        dest = cache / f"ne_10m_{name}.{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        log.info("downloading %s", dest.name)
        with requests.get(MIRROR.format(cat=cat, name=name, ext=ext), stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return cache / f"ne_10m_{name}"


def records(path: Path, clip):
    """(attributes, geometry clipped to ``clip``) for every feature that touches it."""
    r = shapefile.Reader(str(path))
    for sr in r.iterShapeRecords():
        if sr.shape.shapeType == 0:
            continue
        x0, y0, x1, y1 = sr.shape.bbox
        if x1 < clip.bounds[0] or x0 > clip.bounds[2] or y1 < clip.bounds[1] or y0 > clip.bounds[3]:
            continue
        g = shape(sr.shape.__geo_interface__)
        if not g.is_valid:
            g = shapely.make_valid(g)
        g = g.intersection(clip)
        if not g.is_empty:
            yield sr.record.as_dict(), g


def _round(c, nd):
    return [_round(x, nd) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], nd), round(c[1], nd)]


def geojson(items, nd: int = 3) -> dict:
    feats = []
    for props, g in items:
        m = shapely.geometry.mapping(g)
        feats.append({"type": "Feature", "properties": props, "geometry": {"type": m["type"], "coordinates": _round(m["coordinates"], nd)}})
    return {"type": "FeatureCollection", "features": feats}


def write_json(path: Path, obj) -> int:
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    return path.stat().st_size


def _only(g, kinds):
    parts = list(g.geoms) if hasattr(g, "geoms") else [g]
    parts = [p for p in parts if p.geom_type in kinds and not p.is_empty]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.union_all(parts) if kinds[0].startswith("Multi") else shapely.geometrycollections(parts)


def build_land(cache, out, clip):
    items = []
    for _, g in records(download("land", cache), clip):
        g = shapely.simplify(g, 0.004)
        if not g.is_empty:
            items.append(({}, g))
    return write_json(out / "land.json", geojson(items))


def build_lakes(cache, out, clip):
    items = []
    for a, g in records(download("lakes", cache), clip):
        g = shapely.simplify(g, 0.004)
        polys = [p for p in (g.geoms if hasattr(g, "geoms") else [g]) if p.geom_type == "Polygon" and p.area > 0.0015]
        if polys:
            items.append(({"n": a.get("name") or ""}, shapely.MultiPolygon(polys)))
    return write_json(out / "lakes.json", geojson(items))


def build_states(cache, out, clip):
    items = []
    for a, g in records(download("admin_1_states_provinces_lines", cache), clip):
        g = shapely.simplify(g, 0.004)
        if g.geom_type in ("LineString", "MultiLineString"):
            items.append(({}, g))
    return write_json(out / "states.json", geojson(items))


def build_counties(cache, out, clip):
    lines = []
    for a, g in records(download("admin_2_counties", cache), clip.buffer(0.2)):
        if a.get("ISO_A2") not in (None, "US"):
            continue
        lines.append(g.boundary)
    merged = shapely.line_merge(shapely.unary_union(lines))
    merged = shapely.simplify(merged.intersection(clip), 0.004)
    return write_json(out / "counties.json", geojson([({}, merged)]))


def build_roads(cache, out, clip):
    by: dict[tuple[str, str], list] = {}
    for a, g in records(download("roads_north_america", cache), clip):
        if a.get("country") != "United States":
            continue
        cls = {"Interstate": "interstate", "Federal": "us"}.get(a.get("class"))
        if cls is None or (cls == "us" and a.get("type") not in ("Freeway", "Primary", "Secondary")):
            continue
        num = re.sub(r"\s+", "", str(a.get("number") or ""))
        by.setdefault((cls, num), []).append(g)
    feats, labels = [], []
    for (cls, num), geoms in sorted(by.items()):
        merged = shapely.line_merge(shapely.union_all(geoms))
        merged = shapely.simplify(merged, 0.003)
        feats.append(({"c": 0 if cls == "interstate" else 1, "r": (f"I-{num}" if cls == "interstate" else f"US {num}") if num else ""}, merged))
        if num:
            pieces = sorted([p for p in (merged.geoms if hasattr(merged, "geoms") else [merged]) if p.geom_type == "LineString"], key=lambda p: -p.length)
            for k, p in enumerate(pieces[:2]):
                if p.length < 0.35:
                    continue
                pt = p.interpolate(0.5, normalized=True)
                labels.append([f"I-{num}" if cls == "interstate" else f"US {num}", 0 if cls == "interstate" else 1, round(pt.x, 3), round(pt.y, 3), ROUTE_ZOOM[cls] + k])
    n = write_json(out / "roads.json", {"lines": geojson(feats), "labels": labels})
    return n


def _csv_rows(name):
    with open(DATA_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def build_points(out):
    b = display_box(0.0)
    inside = lambda lon, lat: b.bounds[0] <= lon <= b.bounds[2] and b.bounds[1] <= lat <= b.bounds[3]
    places = []
    for r in sorted(_csv_rows("places.csv"), key=lambda r: -int(r["pop"])):
        lon, lat, pop = float(r["lon"]), float(r["lat"]), int(r["pop"])
        if inside(lon, lat):
            places.append([r["name"], r["state"], round(lat, 4), round(lon, 4), pop, next(z for lo, z in TOWN_ZOOM if pop >= lo)])
    peaks = [[r["name"], r["state"], float(r["lat"]), float(r["lon"]), int(r["feet"]), int(r["tier"])] for r in _csv_rows("peaks.csv")
             if inside(float(r["lon"]), float(r["lat"]))]
    ski = [[r["name"], r["state"], float(r["lat"]), float(r["lon"])] for r in _csv_rows("ski_areas.csv") if inside(float(r["lon"]), float(r["lat"]))]
    return write_json(out / "places.json", places), write_json(out / "peaks.json", peaks), write_json(out / "ski.json", ski)


def build_all(out_dir: Path, cache: Path, refresh: bool = False) -> dict[str, int]:
    """Write every layer under ``out_dir/gis``; returns {file: bytes}.  Layers that already exist are kept unless ``refresh``."""
    out = Path(out_dir) / "gis"
    out.mkdir(parents=True, exist_ok=True)
    clip = display_box(0.5)
    wide = display_box(WIDE_PAD_DEG)      # coastline and borders continue well past the data so wide screens show no hard edge
    sizes = {}
    for name, fn in (("land", build_land), ("lakes", build_lakes), ("states", build_states), ("counties", build_counties), ("roads", build_roads)):
        p = out / f"{name}.json"
        if p.exists() and not refresh:
            sizes[name] = p.stat().st_size
            continue
        sizes[name] = fn(cache, out, wide if name in ("land", "lakes", "states") else clip)
        log.info("gis/%s.json  %.0f kB", name, sizes[name] / 1e3)
    sizes["places"], sizes["peaks"], sizes["ski"] = build_points(out)
    return sizes
