"""GEPS must survive an unreachable server: fail fast, switch to a mirror, resume where it stopped, and deliver what it has."""
from datetime import datetime

import numpy as np
import pytest
import requests

from ensemble_snow import config, http
from ensemble_snow.grib import Msg
from ensemble_snow.http import Fetcher, HostDown, NotFound
from ensemble_snow.sources import geps

CYCLE = datetime(2026, 11, 9, 12)
LAT, LON = np.arange(38.0, 50.01, 0.5), np.arange(-77.0, -62.99, 0.5)


def fake_messages(var, level, fhr):
    """21 members; precipitation 1 mm per hour since the start, ASNOW 0.5 mm per hour."""
    v = {"APCP": 1.0 * fhr, "ASNOW": 0.5 * fhr, "TMP": 271.0 if level == "2m" else 260.0, "RH": 90.0, "PRES": 100000.0, "HGT": 5000.0}[var]
    return [Msg({"APCP": "tp", "ASNOW": "asnow", "TMP": "t", "RH": "r", "PRES": "sp", "HGT": "gh"}[var], "surface", 0, (0, fhr), m,
                np.full((len(LAT), len(LON)), v, np.float32), LAT, LON, "", "kg m**-2" if var in ("APCP", "ASNOW") else "") for m in range(1, 22)]


class FakeHttp:
    """Serves files by URL.  ``down`` = hosts that never answer, ``missing_hours`` = hours every server fails on."""
    def __init__(self, down=(), failing=lambda fhr: False):
        self.down, self.failing, self.asked = set(down), failing, []

    def host_down(self, u):
        return any(h in u for h in self.down)

    def _check(self, u):
        if self.host_down(u):
            raise HostDown("down")

    def exists(self, u):
        self.asked.append(("HEAD", u))
        return not self.host_down(u)

    def get(self, u, byte_range=None):
        self._check(u)
        fhr = int(u.rsplit("_P", 1)[1][:3])
        self.asked.append(("GET", fhr))
        if self.failing(fhr):
            raise RuntimeError("giving up on " + u)
        if "PRES_SFC" in u:
            raise NotFound(u)
        return u.encode()


@pytest.fixture(autouse=True)
def decode(monkeypatch):
    def fake(data):
        u = data.decode()
        name = u.rsplit("/", 1)[1]
        var, _lt, level = name.split("_")[2:5]
        fhr = int(name.split("_P")[1][:3])
        return iter(fake_messages(var, level, fhr))
    monkeypatch.setattr(geps, "iter_messages", fake)


def source(http_, **over):
    s = geps.GepsSource(http_)
    s.PAUSE_S, s.PASSES = 0.0, over.pop("passes", 2)
    return s


STEPS = [0, 3, 6, 9, 12]


# ------------------------------------------------------------------------------------------------------------------ the fail-fast limit
def test_a_host_that_never_connects_is_given_up_on_quickly(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    f = Fetcher(trip_after=3, tries=4, cooldown=60)
    calls = []

    def boom(method, url, **kw):
        calls.append(url)
        raise requests.ConnectTimeout("timed out")
    monkeypatch.setattr(f.session, "request", boom)
    with pytest.raises(HostDown):
        f.get("https://dd.example.org/a")
    assert len(calls) == 3                                     # not 4 tries per file for ever: the host is declared down at the third failure
    with pytest.raises(HostDown):
        f.get("https://dd.example.org/b")
    assert len(calls) == 3                                     # nothing sent while it is down
    assert f.host_down("https://dd.example.org/c") and not f.host_down("https://other.example.org/c")
    assert f.exists("https://dd.example.org/a") is False


def test_an_answer_resets_the_count_and_a_404_counts_as_an_answer(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    f = Fetcher(trip_after=2, tries=3)
    seq = iter([requests.ConnectTimeout("x"), type("R", (), {"status_code": 404})(), requests.ConnectTimeout("x"),
                type("R", (), {"status_code": 200, "content": b"ok", "raise_for_status": lambda self: None})()])

    def req(method, url, **kw):
        r = next(seq)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(f.session, "request", req)
    with pytest.raises(NotFound):
        f.get("https://h.example.org/a")                       # 1 failure, then the 404 resets the count
    assert f.get("https://h.example.org/b") == b"ok"           # 1 failure again (not 2 in a row): retried and fine
    assert not f.host_down("https://h.example.org/b")


def test_the_cool_down_ends_and_one_more_failure_trips_it_again(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    f = Fetcher(trip_after=2, tries=3, cooldown=0.0)
    monkeypatch.setattr(f.session, "request", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("no")))
    for _ in range(3):                                         # each probe after the cool-down: one try only, then down again
        with pytest.raises(HostDown):
            f.get("https://h.example.org/a")


# ------------------------------------------------------------------------------------------------------------------ servers
def test_when_the_main_server_is_down_the_mirror_delivers_everything():
    h = FakeHttp(down=["dd.weather.gc.ca"])
    s = source(h)
    assert s.is_complete(CYCLE, 12) is True
    assert "hpfx" in s.good
    ens = s.load(CYCLE, STEPS)
    assert not ens.partial and ens.fhrs == STEPS and len(ens.members) == 21
    assert ens.steps[-1].precip_cum_mm.shape[0] == 21 and np.nanmax(ens.steps[-1].precip_cum_mm) == pytest.approx(12.0, rel=1e-3)


def test_the_url_names_the_run_date_for_the_dated_servers():
    u = geps.url(CYCLE, "APCP", "SFC", "0", 6, geps.MIRRORS[1])
    assert u.startswith("https://hpfx.collab.science.gc.ca/20261109/WXO-DD/ensemble/geps/grib2/raw/12/006/CMC_geps-raw_APCP_SFC_0_latlon0p5x0p5_2026110912_P006_allmbrs.grib2")
    assert geps.url(CYCLE, "APCP", "SFC", "0", 6) == geps.url(CYCLE, "APCP", "SFC", "0", 6, geps.BASE)


def test_a_server_that_dies_half_way_is_replaced_for_the_rest():
    class Dying(FakeHttp):
        def get(self, u, byte_range=None):
            if "dd.weather.gc.ca" in u and len(self.asked) > 40:
                self.down.add("dd.weather.gc.ca")
            return super().get(u, byte_range)
    h = Dying()
    s = source(h)
    ens = s.load(CYCLE, STEPS, workers=1)
    assert not ens.partial and ens.fhrs == STEPS
    assert "hpfx" in s.good


# ------------------------------------------------------------------------------------------------------------------ partial and resumable
def test_hours_that_cannot_be_fetched_leave_a_shorter_run_not_a_failure():
    h = FakeHttp(failing=lambda fhr: fhr >= 9)
    ens = source(h).load(CYCLE, STEPS)
    assert ens.partial and [f for f in ens.fhrs if f > 0 and ens.steps[ens.fhrs.index(f)].precip_cum_mm is not None] == [3, 6]
    assert any("incomplete" in n and "F+9" in n and "F+12" in n for n in ens.notes)


def test_nothing_at_all_is_an_error():
    h = FakeHttp(down=["dd.weather.gc.ca", "hpfx.collab.science.gc.ca"])
    with pytest.raises(RuntimeError, match="nothing could be downloaded"):
        source(h).load(CYCLE, STEPS)


def test_a_second_attempt_fetches_only_what_the_first_did_not_get(tmp_path):
    first = FakeHttp(failing=lambda fhr: fhr >= 9)
    s1 = source(first)
    s1.part_root, s1.part_tag = tmp_path, "abc"
    assert s1.load(CYCLE, STEPS).partial
    kept = sorted(p.name for p in (tmp_path / "parts" / "geps" / "abc_2026110912").glob("f*.pkl.gz"))
    assert kept == ["f000.pkl.gz", "f003.pkl.gz", "f006.pkl.gz"]           # finished hours only

    second = FakeHttp()
    s2 = source(second)
    s2.part_root, s2.part_tag = tmp_path, "abc"
    ens = s2.load(CYCLE, STEPS)
    assert not ens.partial and ens.fhrs == STEPS
    assert sorted({v for k, v in second.asked if k == "GET"}) == [9, 12]   # 0, 3, 6 came from disk


def test_a_new_run_clears_the_old_parts(tmp_path):
    s = source(FakeHttp())
    s.part_root, s.part_tag = tmp_path, "abc"
    s.load(CYCLE, [0, 3])
    s.load(datetime(2026, 11, 10, 0), [0, 3])
    assert [p.name for p in (tmp_path / "parts" / "geps").iterdir()] == ["abc_2026111000"]


def test_the_time_budget_stops_a_download_that_goes_on_too_long():
    s = source(FakeHttp())
    s.BUDGET_S = -1.0
    with pytest.raises(RuntimeError, match="nothing could be downloaded"):
        s.load(CYCLE, STEPS)
