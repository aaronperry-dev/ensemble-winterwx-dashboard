import numpy as np
import pytest

from ensemble_snow import config, slr
from ensemble_snow.fields import Profile

LEVELS = (1000, 925, 850, 700, 600, 500)


def column(t_k, shape=(1, 1, 1), z=None, rh=None):
    ny, nx = shape[-2:]
    lat = np.full((ny, nx), 43.0, np.float32)
    lon = np.full((ny, nx), -71.0, np.float32)
    t = Profile(LEVELS, np.full((len(LEVELS),) + shape, t_k, np.float32))
    return slr.Column(shape=shape, lat=lat, lon=lon, t2_k=np.full(shape, t_k, np.float32), t=t, z=z, rh=rh)


def test_every_documented_method_is_registered():
    assert {"fixed10", "fixed12", "kuchera", "maxt", "cmc_thick", "nbm_thick", "byun", "climo", "cobb"} <= set(slr.METHODS)
    assert all(k in slr.METHODS for k in config.BLEND)


def test_fixed_ratios():
    col = column(265.0)
    assert slr.METHODS["fixed10"].fn(col)[0, 0, 0] == 10.0
    assert slr.METHODS["fixed12"].fn(col)[0, 0, 0] == 12.0


@pytest.mark.parametrize("t_k, expected", [(268.16, 15.0), (271.16, 12.0), (273.16, 8.0)])
def test_kuchera_golden_values(t_k, expected):
    # colder than -2 C gains one point per degree, warmer loses two (Kuchera 2004)
    assert slr.METHODS["kuchera"].fn(column(t_k))[0, 0, 0] == pytest.approx(expected, abs=1e-4)


def test_ratio_is_clipped_to_a_meaningful_range():
    cold = slr.METHODS["kuchera"].fn(column(230.0))
    assert cold.max() <= slr.SLR_MAX


def test_method_needing_missing_input_is_unavailable():
    col = column(268.0)
    assert slr.METHODS["kuchera"].available(col)
    assert not slr.METHODS["byun"].available(col)             # needs the interval precipitation rate
    assert not slr.METHODS["cobb"].available(col)              # needs humidity profiles


def test_adding_a_method_is_one_decorator():
    @slr.register("test_double", "Test x2", ("t2_k",), desc="twice 10:1", ref="test")
    def double(col):
        return np.full(col.shape, 20.0, np.float32)
    try:
        assert slr.METHODS["test_double"].fn(column(268.0)).mean() == 20.0
        with pytest.raises(ValueError):
            slr.register("test_double", "dup", (), desc="", ref="")(double)
    finally:
        del slr.METHODS["test_double"]


def test_interval_ratio_averages_both_ends():
    m = slr.METHODS["kuchera"]
    r = slr.interval_ratio(m, column(268.16), column(270.16))
    assert r[0, 0, 0] == pytest.approx(0.5 * (15.0 + 13.0))
    assert slr.interval_ratio(m, None, column(268.16))[0, 0, 0] == pytest.approx(15.0)


def test_cobb_is_higher_in_the_dendritic_zone():
    from ensemble_snow import thermo
    rh = Profile(LEVELS, np.full((len(LEVELS), 1, 1, 1), 95.0, np.float32))
    zs = Profile((1000, 850, 700), np.stack([np.full((1, 1, 1), thermo.std_atmosphere_height_m(p), np.float32) for p in (1000, 850, 700)]))
    # a column near -15 C through the depth is much more favourable than one near -2 C
    cold = slr.METHODS["cobb"].fn(column(258.16, rh=rh, z=zs))[0, 0, 0]
    warm = slr.METHODS["cobb"].fn(column(271.16, rh=rh, z=zs))[0, 0, 0]
    assert cold > warm
