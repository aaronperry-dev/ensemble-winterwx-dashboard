"""The combined view must keep working with whatever data exist: a missing system, missing hours, older runs, a failing download."""
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from ensemble_snow import build, compute, config, encode
from ensemble_snow.sources import synthetic as syn

CYCLE = datetime(2026, 11, 9, 12)


class Degraded(syn.SyntheticSource):
    """A synthetic system with something wrong: ``fail`` (the download raises), ``drop`` (hours whose precipitation is missing),
    ``drop_members`` ({hour: members with no data}), or ``blank`` (hours with nothing at all, not even a step)."""
    plan: dict = {}

    def load(self, cycle, steps, workers=8):
        p = self.plan.get(self.key, {})
        if p.get("fail"):
            raise RuntimeError("download failed")
        ens = super().load(cycle, steps, workers)
        keep = []
        for st in ens.steps:
            if st.fhr in p.get("blank", ()):
                continue
            if "scale" in p and st.precip_cum_mm is not None:         # a system that forecasts much more (or less) than the others
                st.precip_cum_mm = st.precip_cum_mm * p["scale"]
                if st.snow_liq_cum_mm is not None:
                    st.snow_liq_cum_mm = st.snow_liq_cum_mm * p["scale"]
            if st.fhr in p.get("drop", ()):
                st.precip_cum_mm = None
                st.snow_liq_cum_mm = None
            for m in p.get("drop_members", {}).get(st.fhr, ()):
                st.precip_cum_mm[m] = np.nan
                if st.snow_liq_cum_mm is not None:
                    st.snow_liq_cum_mm[m] = np.nan
            keep.append(st)
        ens.steps = keep
        return ens


@pytest.fixture
def degrade(monkeypatch):
    def setup(plan):
        Degraded.plan = plan
        monkeypatch.setattr(syn, "SyntheticSource", Degraded)
    yield setup
    Degraded.plan = {}


def run(tmp_path, keys, max_fhr=12):
    build.build_site(tmp_path, keys, synthetic=True, max_fhr=max_fhr, cache=tmp_path / "c", force_cycle=CYCLE, common=True, workers=2, with_gis=False)
    return json.loads((tmp_path / "data" / "manifest.json").read_text())


def blend_p50(tmp_path, man, name):
    e = man["ensembles"][name]
    _, blocks = encode.read_bundle(tmp_path / "data" / e["files"]["blend"]["file"])
    return blocks["amt"][man["windows"].index(0)][man["stats"].index("p50")]       # run total, p50: (T, ny, nx)


def test_one_system_fails_the_others_are_still_blended(tmp_path, degrade):
    degrade({"gefs": {"fail": True}})
    man = run(tmp_path, ["eps", "gefs", "geps"])
    assert set(man["ensembles"]) == {"eps", "geps", "all"}
    a = man["ensembles"]["all"]
    assert [s["key"] for s in a["systems"]] == ["eps", "geps"] and a["members"] == 51 + 21
    assert set(a["contrib"]) == {"eps,geps"}


def test_missing_hours_are_filled_in_and_the_totals_never_fall(tmp_path, degrade):
    degrade({"eps": {"drop": {6, 9}}, "geps": {"blank": {9}}})
    man = run(tmp_path, ["eps", "gefs", "geps"])
    assert 6 not in man["ensembles"]["eps"]["fhrs"] and 9 not in man["ensembles"]["eps"]["fhrs"]
    assert 9 not in man["ensembles"]["geps"]["fhrs"]
    a = man["ensembles"]["all"]
    assert a["fhrs"] == [3, 6, 9, 12]                                      # every hour any system has
    assert dict(zip(a["fhrs"], a["contrib"])) == {3: "eps,gefs,geps", 6: "gefs,geps", 9: "gefs", 12: "eps,gefs,geps"}    # which have data of their own
    assert dict(zip(a["fhrs"], a["filled"])) == {3: "", 6: "eps", 9: "eps,geps", 12: ""}                                  # and which are filled in
    _, blocks = encode.read_bundle(tmp_path / "data" / a["files"]["blend"]["file"])
    amt = blocks["amt"][man["windows"].index(0)]                            # run total
    for stat in ("mean", "p50", "p90", "max"):
        v = np.nan_to_num(amt[man["stats"].index(stat)])
        assert (np.diff(v, axis=0) >= -0.06).all(), stat                    # a running total never falls from one hour to the next


def test_totals_do_not_zigzag_when_one_system_has_every_other_hour(tmp_path, degrade):
    """The bug seen on the page: a system present at some hours and absent at others made the pooled totals rise and fall."""
    degrade({"eps": {"drop": {3, 9, 15}, "scale": 3.0}})
    man = run(tmp_path, ["eps", "gefs"], max_fhr=18)
    a = man["ensembles"]["all"]
    assert a["fhrs"] == [3, 6, 9, 12, 15, 18]
    _, blocks = encode.read_bundle(tmp_path / "data" / a["files"]["blend"]["file"])
    amt = blocks["amt"][man["windows"].index(0)]
    for stat in ("mean", "p50", "p75", "max"):
        v = np.nan_to_num(amt[man["stats"].index(stat)])
        assert (np.diff(v, axis=0) >= -0.06).all(), stat


def test_members_missing_at_one_hour_do_not_poison_later_hours(tmp_path, degrade):
    degrade({"gefs": {"drop_members": {6: [0, 1, 2]}}})
    man = run(tmp_path, ["eps", "gefs"])
    e = man["ensembles"]["gefs"]
    assert e["fhrs"] == [3, 6, 9, 12]
    p50 = blend_p50(tmp_path, man, "gefs")
    assert np.isfinite(p50).all() and (np.diff(p50, axis=0) >= -0.06).all()
    assert set(man["ensembles"]) == {"eps", "gefs", "all"}


def test_a_hole_in_the_data_only_blanks_the_windows_that_need_it(tmp_path, degrade):
    degrade({"eps": {"drop": {6}}})
    man = run(tmp_path, ["eps"])
    e = man["ensembles"]["eps"]
    assert e["fhrs"] == [3, 9, 12] and any("hour(s) 6" in n for n in e["notes"])
    assert "all" not in man["ensembles"]                                    # one system: nothing to blend


def test_older_run_is_lined_up_by_valid_time(tmp_path):
    lon, lat = config.grid_axes()
    LON, LAT = np.meshgrid(lon, lat)
    steps = list(range(0, 37, 3))
    kept = {}
    for key, cyc in (("eps", CYCLE - timedelta(hours=12)), ("geps", CYCLE)):
        ens = syn.SyntheticSource(key).load(cyc, steps)
        kept[key] = (ens, compute.compute_snowfall(ens, LAT, LON))
    (tmp_path / "data").mkdir()
    a = build.combine(kept, tmp_path)
    assert a["init"] == "2026-11-09T12:00:00Z" and a["fhrs"][0] == 3 and a["fhrs"][-1] == 24     # the older EPS run ends at F+24 of the newest
    assert set(a["contrib"]) == {"eps,geps"} and any("ends at F+24 h" in n for n in a["notes"])  # the pool stops there rather than lose a system
    assert any("12 h before the newest" in n for n in a["notes"])
    assert [s["init"] for s in a["systems"]] == ["2026-11-09T00:00:00Z", "2026-11-09T12:00:00Z"]
    # the older system's totals count from the newest run's start: the snowiest member at F+3 is the larger of EPS (F+15 minus F+12 of its run) and GEPS
    e_res, g_res = kept["eps"][1], kept["geps"][1]
    eps_first = e_res.methods["blend"][e_res.fhrs.index(15)] - e_res.methods["blend"][e_res.fhrs.index(12)]
    want = np.maximum(eps_first.max(axis=0), g_res.methods["blend"][g_res.fhrs.index(3)].max(axis=0))
    _, blocks = encode.read_bundle(tmp_path / "data" / a["files"]["blend"]["file"])
    mx = blocks["amt"][config.WINDOWS_H.index(0)][list(config.ALL_STATS).index("max")]
    assert np.abs(mx[0] - want).max() < 0.06
    assert np.isfinite(mx).all() and (np.diff(mx, axis=0) >= -0.06).all()


def test_a_run_too_old_is_left_out_not_fatal(tmp_path):
    lon, lat = config.grid_axes()
    LON, LAT = np.meshgrid(lon, lat)
    steps = list(range(0, 37, 3))
    kept = {}
    for key, cyc in (("eps", CYCLE - timedelta(hours=48)), ("gefs", CYCLE), ("geps", CYCLE)):
        ens = syn.SyntheticSource(key).load(cyc, steps)
        kept[key] = (ens, compute.compute_snowfall(ens, LAT, LON))
    (tmp_path / "data").mkdir()
    a = build.combine(kept, tmp_path)
    assert [s["key"] for s in a["systems"]] == ["gefs", "geps"]
    assert any("too old" in n for n in a["notes"])
    assert build.combine({"eps": kept["eps"], "geps": kept["geps"]}, tmp_path) is None


class Fake:
    def __init__(self, done):
        self.done = done

    def is_complete(self, cycle, max_fhr):
        if isinstance(self.done, Exception):
            raise self.done
        return cycle in self.done


def test_cycle_choice_with_a_missing_or_late_system():
    now = datetime(2026, 11, 9, 20)
    c12, c00 = datetime(2026, 11, 9, 12), datetime(2026, 11, 9, 0)
    # EPS never complete: the other two share 12 UTC
    got = build.pick_cycles({"eps": Fake(set()), "gefs": Fake({c12, c00}), "geps": Fake({c12, c00})}, now, 168)
    assert got == {"gefs": c12, "geps": c12}
    # GEPS only has the older run: its newest run is used with the others' newest unless a shared complete run is recent enough
    got = build.pick_cycles({"eps": Fake({c12, c00}), "gefs": Fake({c12, c00}), "geps": Fake({c00})}, now, 168)
    assert got == {"eps": c00, "gefs": c00, "geps": c00}
    # the shared run is more than 12 h older than the newest one: each system uses its own, and the blend lines them up
    got = build.pick_cycles({"eps": Fake({c12, c00}), "gefs": Fake({c12, c00}), "geps": Fake({c00 - timedelta(hours=12)})}, now, 168)
    assert got["eps"] == c12 and got["gefs"] == c12 and got["geps"] == datetime(2026, 11, 8, 12)
    # a server that cannot be reached counts as "not complete", nothing more
    got = build.pick_cycles({"eps": Fake(RuntimeError("down")), "gefs": Fake({c12})}, now, 168)
    assert got == {"gefs": c12}
    assert build.pick_cycles({"eps": Fake(set())}, now, 168) == {}
