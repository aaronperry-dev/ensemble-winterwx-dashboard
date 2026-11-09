"""Round trip through real GRIB2 messages written with ecCodes, so decoding, cropping and orientation are exercised without a server."""
import numpy as np
import pytest

eccodes = pytest.importorskip("eccodes")

from ensemble_snow import grib


def make_message(values, lat0=50.0, lat1=38.0, lon0=-80.0, lon1=-60.0, short="tp", step=(0, 6), number=3):
    gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ny, nx = values.shape
    for k, v in (("Ni", nx), ("Nj", ny), ("latitudeOfFirstGridPointInDegrees", lat0), ("latitudeOfLastGridPointInDegrees", lat1),
                 ("longitudeOfFirstGridPointInDegrees", lon0 % 360), ("longitudeOfLastGridPointInDegrees", lon1 % 360),
                 ("iDirectionIncrementInDegrees", (lon1 - lon0) / (nx - 1)), ("jDirectionIncrementInDegrees", (lat0 - lat1) / (ny - 1))):
        eccodes.codes_set(gid, k, v)
    eccodes.codes_set_values(gid, values.ravel().astype(float))
    msg = eccodes.codes_get_message(gid)
    eccodes.codes_release(gid)
    return msg


def test_decode_orientation_and_crop():
    ny, nx = 25, 41                                              # 0.5 degree grid, north to south as GRIB usually is
    lat = np.linspace(50, 38, ny)
    values = np.repeat(lat[:, None], nx, axis=1)                  # value == latitude
    msgs = list(grib.iter_messages(make_message(values) * 2))
    assert len(msgs) == 2
    m = msgs[0]
    assert m.lat[0] < m.lat[-1]                                   # latitude ascending after decode
    np.testing.assert_allclose(m.values[:, 0], m.lat, atol=0.05)
    assert m.lat.min() >= 38.0 and m.lat.max() <= 50.0
    assert m.lon.min() < 0


def test_grid_that_starts_at_minus_180_is_decoded():
    """ECMWF open data starts at 180 deg west; GRIB stores that as 180 E, so first/last longitude alone give a nonsense axis."""
    ny, nx = 721, 1440
    lon = np.linspace(-180.0, 179.75, nx)
    gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    for k, v in dict(Ni=nx, Nj=ny, latitudeOfFirstGridPointInDegrees=90, latitudeOfLastGridPointInDegrees=-90,
                     longitudeOfFirstGridPointInDegrees=-180, longitudeOfLastGridPointInDegrees=179.75,
                     iDirectionIncrementInDegrees=0.25, jDirectionIncrementInDegrees=0.25).items():
        eccodes.codes_set(gid, k, v)
    eccodes.codes_set_values(gid, np.tile(lon, (ny, 1)).ravel())          # value == longitude
    msg = eccodes.codes_get_message(gid)
    eccodes.codes_release(gid)
    m = next(grib.iter_messages(msg))
    np.testing.assert_allclose(m.values[0], m.lon, atol=0.01)
    assert m.lon.min() < -70 and m.lon.max() > -70


def test_global_zero_to_360_grid_is_decoded():
    ny, nx = 721, 1440
    lat = np.linspace(90, -90, ny)
    gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    for k, v in dict(Ni=nx, Nj=ny, latitudeOfFirstGridPointInDegrees=90, latitudeOfLastGridPointInDegrees=-90,
                     longitudeOfFirstGridPointInDegrees=0, longitudeOfLastGridPointInDegrees=359.75,
                     iDirectionIncrementInDegrees=0.25, jDirectionIncrementInDegrees=0.25).items():
        eccodes.codes_set(gid, k, v)
    eccodes.codes_set_values(gid, np.repeat(lat[:, None], nx, axis=1).ravel())      # value == latitude
    msg = eccodes.codes_get_message(gid)
    eccodes.codes_release(gid)
    m = next(grib.iter_messages(msg))
    np.testing.assert_allclose(m.values[:, 0], m.lat, atol=0.01)
