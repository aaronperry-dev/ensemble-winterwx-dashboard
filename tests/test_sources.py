from datetime import datetime

import numpy as np
import pytest

from ensemble_snow.sources import gefs, geps

IDX = """1:0:d=2026110900:APCP:surface:0-6 hour acc fcst:
2:1000:d=2026110900:CSNOW:surface:6 hour fcst:
3:1500:d=2026110900:TMP:2 m above ground:6 hour fcst:
4:2600:d=2026110900:UGRD:10 m above ground:6 hour fcst:
5:3000:d=2026110900:DPT:2 m above ground:6 hour fcst:
"""


def test_gefs_urls():
    c = datetime(2026, 11, 9, 0)
    assert gefs.url(c, "s", 0, 24).endswith("gefs.20261109/00/atmos/pgrb2sp25/gec00.t00z.pgrb2s.0p25.f024")
    assert gefs.url(c, "a", 7, 6).endswith("gefs.20261109/00/atmos/pgrb2ap5/gep07.t00z.pgrb2a.0p50.f006")


def test_gefs_idx_selection_and_range_coalescing():
    recs = gefs.parse_idx(IDX)
    assert [r["var"] for r in recs] == ["APCP", "CSNOW", "TMP", "UGRD", "DPT"]
    assert recs[0]["end"] == 999 and recs[-1]["end"] is None
    want = gefs.select(recs, lambda v, lev: (v, lev) in gefs.SURFACE_WANT)
    assert [r["var"] for r in want] == ["APCP", "CSNOW", "TMP", "DPT"]
    # APCP+CSNOW+TMP are adjacent and fetched as one request, DPT separately
    assert gefs.coalesce(want) == [(0, 2599, 3), (3000, None, 1)]


def test_geps_water_equivalent_check():
    precip = np.full((10, 10), 10.0, np.float32)
    assert geps.snow_is_water_equivalent(precip * 0.8, precip)
    assert not geps.snow_is_water_equivalent(precip * 8.0, precip)         # a depth in cm-like units would be ~10x the liquid
