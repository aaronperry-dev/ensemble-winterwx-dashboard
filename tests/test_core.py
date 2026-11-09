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
    np.testing.assert_allclose(amt[2], np.percentile(x, 50, axis=0, method="hazen"), rtol=1e-5)
    np.testing.assert_allclose(amt[-1], x.max(axis=0))
    np.testing.assert_allclose(prob[1], 100 * (x >= config.THRESHOLDS_IN[1]).mean(axis=0), rtol=1e-5)


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
    assert heavy_low[2, 0] < heavy_high[2, 0]


def test_all_nan_cells_stay_nan():
    x = np.full((5, 2), np.nan, np.float32)
    amt, prob = stats.ensemble_stats(x)
    assert np.isnan(amt).all() and np.isnan(prob).all()


def test_rolling_24h_window():
    fh = [6, 12, 18, 24, 30]
    cum = np.arange(1, 6, dtype=np.float32).reshape(5, 1, 1, 1) * 2
    w = stats.windowed(cum, 24, fh)
    assert w[3, 0, 0, 0] == cum[3, 0, 0, 0]               # first 24 h: everything since the start
    assert w[4, 0, 0, 0] == cum[4, 0, 0, 0] - cum[0, 0, 0, 0]


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
