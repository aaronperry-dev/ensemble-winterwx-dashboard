import numpy as np
import pytest

from ensemble_snow import accum, config, encode, stats


# ---- accumulation bookkeeping
def test_parse_step_range():
    assert accum.parse_step_range("0-6") == (0, 6)
    assert accum.parse_step_range("12") == (0, 12)           # a bare step is accumulated from the start


def test_tracker_from_start_totals():
    t = accum.CumulativeTracker()
    t.add(0, 6, np.array([1.0])); t.add(0, 12, np.array([3.0]))
    assert t.get(12)[0] == 3.0


def test_tracker_six_hour_buckets_are_chained():
    t = accum.CumulativeTracker()
    for s in (0, 6, 12):
        t.add(s, s + 6, np.array([2.0]))
    assert t.get(18)[0] == 6.0


def test_tracker_out_of_order_and_missing():
    t = accum.CumulativeTracker()
    t.add(6, 12, np.array([2.0])); t.add(0, 6, np.array([1.0]))
    assert t.get(12)[0] == 3.0
    with pytest.raises(accum.MissingWindow):
        t.get(24)
    t.add(18, 24, np.array([1.0]))
    assert t.unresolved() == [(18, 24)]


# ---- statistics
def test_stats_match_numpy_and_probabilities():
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 2.0, size=(31, 4, 5)).astype(np.float32)
    amt, prob = stats.ensemble_stats(x)
    assert amt.shape[0] == len(config.PCT_STATS) == 10
    for i, name in enumerate(config.PCT_STATS):                                  # min, 5th, 10th, 25th, median, mean, 75th, 90th, 95th, max
        want = x.mean(axis=0) if name == "mean" else np.percentile(x, config.PCT_LEVELS[name], axis=0, method="hazen")
        np.testing.assert_allclose(amt[i], want, rtol=1e-5, atol=1e-6, err_msg=name)
    np.testing.assert_allclose(amt[0], x.min(axis=0))
    np.testing.assert_allclose(amt[-1], x.max(axis=0))
    np.testing.assert_allclose(prob[1], 100 * (x >= config.THRESHOLDS_IN[1]).mean(axis=0), rtol=1e-5)


def test_percentiles_ignore_missing_members():
    x = np.random.default_rng(7).gamma(2.0, 2.0, size=(25, 3, 3)).astype(np.float32)
    x[:4, 0, 0] = np.nan
    amt, _ = stats.ensemble_stats(x)
    i = config.PCT_STATS.index("p90")
    np.testing.assert_allclose(amt[i], np.nanpercentile(x, 90, axis=0, method="hazen"), rtol=1e-5)


def test_equal_weights_reproduce_unweighted():
    x = np.random.default_rng(2).gamma(2.0, 2.0, size=(20, 3, 3)).astype(np.float32)
    a, p = stats.ensemble_stats(x)
    aw, pw = stats.ensemble_stats(x, np.ones(20))
    np.testing.assert_allclose(a, aw, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(p, pw, rtol=1e-5)


def test_weights_shift_the_median():
    x = np.array([0.0] * 10 + [10.0] * 10, np.float32).reshape(20, 1)
    heavy_low, _ = stats.ensemble_stats(x, np.r_[np.full(10, 3.0), np.full(10, 1.0)])
    heavy_high, _ = stats.ensemble_stats(x, np.r_[np.full(10, 1.0), np.full(10, 3.0)])
    i = config.PCT_STATS.index("p50")
    assert heavy_low[i, 0] < heavy_high[i, 0]


def test_all_nan_cells_stay_nan():
    x = np.full((5, 2), np.nan, np.float32)
    amt, prob = stats.ensemble_stats(x)
    assert np.isnan(amt).all() and np.isnan(prob).all()


def test_rolling_24h_window():
    fh = [6, 12, 18, 24, 30]
    cum = np.arange(1, 6, dtype=np.float32).reshape(5, 1, 1, 1) * 2
    w = stats.windowed(cum, 24, fh)
    assert np.isnan(w[:3]).all()                          # a 24-hour total does not exist before F+24 h
    assert w[3, 0, 0, 0] == cum[3, 0, 0, 0]               # F+24: everything since the start
    assert w[4, 0, 0, 0] == cum[4, 0, 0, 0] - cum[0, 0, 0, 0]


def test_windows_follow_the_available_output_hours():
    """ECMWF is 3-hourly to F+144 h and 6-hourly after: a 3-hour total cannot be formed late in the run, a 6-hour one can."""
    from ensemble_snow.sources.base import forecast_steps
    fh = forecast_steps(168, 3, False, 144, 6)
    assert fh[:3] == [3, 6, 9] and fh[-4:] == [144, 150, 156, 162, 168][-4:] and 147 not in fh
    cum = np.cumsum(np.ones((len(fh), 1, 1, 1), np.float32), axis=0)
    for hours, first, last in ((3, 3, 144), (6, 6, 168), (12, 12, 168), (24, 24, 168), (48, 48, 168), (0, 3, 168)):
        ok = [f for f, v in zip(fh, stats.window_valid(hours, fh)) if v]
        assert ok[0] == first and ok[-1] == last, (hours, ok[0], ok[-1])
        w = stats.windowed(cum, hours, fh)
        assert (~np.isnan(w[:, 0, 0, 0]) == stats.window_valid(hours, fh)).all()
    w3 = stats.windowed(cum, 3, fh)
    assert w3[fh.index(150), 0, 0, 0] != w3[fh.index(150), 0, 0, 0]        # NaN


def test_forecast_steps_are_3_hourly_then_6_hourly():
    from ensemble_snow.sources.base import forecast_steps
    s = forecast_steps(168, 3, True, 144, 6)
    assert s[0] == 0 and s[48] == 144 and s[49] == 150 and s[-1] == 168 and len(s) == 53
    assert forecast_steps(24, 3) == [0, 3, 6, 9, 12, 15, 18, 21, 24]


# ---- bundle format
def test_bundle_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    amt = np.round(rng.gamma(1.5, 2.0, size=(2, 3, 4, 6, 7)), 2).astype(np.float32)
    amt[0, 0, 0, 0, 0] = np.nan
    prob = rng.integers(0, 101, size=(2, 3, 4, 6, 7)).astype(np.float32)
    p = tmp_path / "b.bin.gz"
    encode.write_bundle(p, [{"name": "amt", "data": amt, "dtype": "u16", "lo": 0.0, "step": 0.05},
                            {"name": "prob", "data": prob, "dtype": "u8", "lo": 0.0, "step": 1.0}], {"x": 1})
    meta, out = encode.read_bundle(p)
    assert meta == {"x": 1}
    assert np.isnan(out["amt"][0, 0, 0, 0, 0])
    ok = ~np.isnan(amt)
    assert np.abs(out["amt"][ok] - amt[ok]).max() <= 0.025 + 1e-6
    np.testing.assert_array_equal(out["prob"], prob)


def test_progress_bars_are_silent(capsys):
    """CI logs must not fill with one line per progress-bar redraw (a full EPS run was ~57 000 lines)."""
    import ensemble_snow  # noqa: F401  (importing the package switches the bars off)
    from multiurl.base import progress_bar
    from tqdm.auto import tqdm
    with tqdm(total=10, desc="x") as bar:
        bar.update(5)
    bar = progress_bar(total=100, desc="download.grib2")
    bar.update(50)
    bar.close()
    out = capsys.readouterr()
    assert out.err == "" and out.out == ""


# --------------------------------------------------------------------------- terrain adjustment
def _terrarium_png(h):
    """A 256 x 256 Terrarium PNG for a height field (m)."""
    import io

    from PIL import Image
    v = np.round((h + 32768.0) * 256.0).astype(np.int64)
    rgb = np.stack([v // 65536, (v // 256) % 256, v % 256], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "PNG")
    return buf.getvalue()


def test_terrarium_decoding_round_trips_heights():
    from ensemble_snow import terrain
    h = np.linspace(-50, 1900, 256 * 256).reshape(256, 256)
    back = terrain.decode_terrarium(_terrarium_png(h))
    assert np.abs(back - h).max() < 0.01


def test_elevation_is_binned_to_the_fine_grid_with_sea_at_zero(tmp_path):
    """Every tile is a slope that rises with longitude; sea-floor depth counts as sea level; a missing tile is height 0."""
    from ensemble_snow import terrain
    from ensemble_snow.http import NotFound

    class Fake:
        def __init__(self):
            self.n = 0

        def get(self, url, byte_range=None):
            self.n += 1
            parts = url.split("/")
            z, x = int(parts[-3]), int(parts[-2])
            if self.n == 3:
                raise NotFound(url)
            n = 256 * 2 ** z
            lon = ((x * 256 + np.arange(256) + 0.5) / n) * 360.0 - 180.0
            return _terrarium_png(np.tile(np.where(lon < -70.0, -200.0, (lon + 70.0) * 100.0), (256, 1)))

    fake = Fake()
    elev = terrain.fetch_elevation(tmp_path, fake)
    lon, lat = terrain.fine_axes()
    assert elev.shape == (len(lat), len(lon))
    west, east = elev[:, lon < -71.0], elev[:, (lon > -69.0) & (lon < -66.0)]
    assert west.max() == 0.0                              # depth below sea level is clipped to 0
    want = (lon[(lon > -69.0) & (lon < -66.0)] + 70.0) * 100.0
    assert np.abs(east.mean(axis=0) - want).max() < 10    # the slope comes back to within a pixel's worth
    assert (tmp_path / "dem").exists() and terrain.fetch_elevation(tmp_path, object()) is not None     # second call is served from the cache


def test_relief_is_zero_on_a_plain_and_positive_on_a_ridge():
    from ensemble_snow import terrain
    lon, lat = terrain.fine_axes()
    flat = np.full((len(lat), len(lon)), 300.0)
    assert np.abs(terrain.relief(flat, 0.25)).max() < 1e-6
    ridge = flat.copy()
    ridge[:, 100] = 900.0                                  # a one-cell (5 km) ridge: the model, 25 km wide, averages it away
    r = terrain.relief(ridge, 0.25)
    assert r[80, 100] > 400 and r[80, 130] == pytest.approx(0.0, abs=1e-6)
    assert terrain.relief(ridge, 0.5)[80, 100] > r[80, 100]    # a coarser model sees even less of it


def test_terrain_build_writes_a_small_file_and_survives_failure(tmp_path):
    import json

    from ensemble_snow import terrain
    entry = terrain.build(tmp_path / "data", tmp_path, synthetic=True)
    doc = json.loads((tmp_path / "data" / "terrain.json").read_text())
    assert entry["file"] == "terrain.json" and set(entry["smooth"]) == {"eps", "gefs", "geps", "all"}
    g = doc["grid"]
    assert (g["nx"], g["ny"]) == (201, 161) and set(doc["dz"]) == {"0.25", "0.5"}
    assert len(doc["elev"]) > 1000 and (tmp_path / "data" / "terrain.json").stat().st_size < 600_000
    class Broken:
        def get(self, *a, **k):
            raise RuntimeError("offline")
    assert terrain.build(tmp_path / "d2", tmp_path / "c2", synthetic=False, fetcher=Broken()) is None      # no elevation data: no terrain option, no crash
