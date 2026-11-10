"""Shaded relief: the hillshade geometry and the image written for the basemap."""
import numpy as np
import pytest

from ensemble_snow import relief

BOUNDS = (43.0, -72.0, 44.0, -71.0)


def _ramp(rising_toward: str, n=64, slope=0.15):
    """A tilted plane (m) whose height rises toward 'n', 's', 'e' or 'w'."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    # rows run north to south, columns west to east; slope in metres per pixel
    h = {"n": -yy, "s": yy, "e": xx, "w": -xx}[rising_toward] * slope * 400
    return (h - h.min() + 100).astype(np.float32)


def test_flat_ground_has_the_flat_shade_and_is_transparent():
    flat = np.full((32, 32), 200, np.float32)
    assert np.allclose(relief.shade(flat, BOUNDS), np.sin(np.radians(relief.ALTITUDE)))
    assert (relief.to_rgba(flat, BOUNDS)[..., 3] == 0).all()


def test_slopes_facing_the_north_west_light_are_lit_and_the_others_dark():
    sun = np.sin(np.radians(relief.ALTITUDE))
    c = (slice(20, 44), slice(20, 44))
    lit = relief.shade(_ramp("s"), BOUNDS)[c].mean()          # rising toward the south: the slope faces north
    lit_w = relief.shade(_ramp("e"), BOUNDS)[c].mean()        # rising toward the east: the slope faces west
    dark = relief.shade(_ramp("n"), BOUNDS)[c].mean()         # facing south
    dark_e = relief.shade(_ramp("w"), BOUNDS)[c].mean()       # facing east
    assert lit > sun and lit_w > sun
    assert dark < sun and dark_e < sun


def test_sea_stays_clear_and_shadows_are_black_highlights_white():
    h = _ramp("n")
    h[:, :10] = 0                                              # open water at the left edge
    img = relief.to_rgba(h, BOUNDS)
    assert (img[:, :10, 3] == 0).all()
    shadow = img[30:34, 30:34]
    assert (shadow[..., :3] == 0).all() and shadow[..., 3].max() > 20
    light = relief.to_rgba(_ramp("s"), BOUNDS)[30:34, 30:34]
    assert (light[..., :3] == 255).all() and light[..., 3].max() > 5
    assert img[..., 3].max() <= round(relief.MAX_ALPHA * 255)


def test_build_writes_an_image_and_its_corners(tmp_path):
    from PIL import Image
    info = relief.build(tmp_path / "data", tmp_path / "cache", synthetic=True)
    assert info and info["file"] == "relief.webp"
    s, w, n, e = info["bounds"]
    assert s < 40.0 - relief.PAD_DEG + 0.5 and n > 48.0 + relief.PAD_DEG - 0.5 and w < -75.0 and e > -65.0     # reaches past the model grid
    im = Image.open(tmp_path / "data" / "relief.webp")
    assert im.mode == "RGBA" and im.size[0] % 256 == 0 and im.size[1] % 256 == 0
    assert np.asarray(im)[..., 3].max() > 30                   # there is relief to see
    assert info["bytes"] == (tmp_path / "data" / "relief.webp").stat().st_size


def test_a_missing_elevation_source_leaves_the_site_without_relief(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(relief, "fetch_mosaic", boom)
    assert relief.build(tmp_path, tmp_path) is None
