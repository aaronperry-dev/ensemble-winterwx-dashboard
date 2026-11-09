import json
from datetime import datetime

import numpy as np

from ensemble_snow import build, config, encode


def test_synthetic_build_end_to_end(tmp_path):
    m = build.build_site(tmp_path, ["eps", "gefs", "geps"], synthetic=True, max_fhr=24, cache=tmp_path / "c", force_cycle=datetime(2026, 11, 9, 0),
                         common=True, workers=2, with_gis=False)
    man = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert man["synthetic"] and set(man["ensembles"]) == {"eps", "gefs", "geps", "all"}
    assert (tmp_path / "index.html").exists() and (tmp_path / "js" / "main.js").exists() and (tmp_path / ".nojekyll").exists()
    e = man["ensembles"]["eps"]
    assert e["fhrs"] == [3, 6, 9, 12, 15, 18, 21, 24] and "blend" in e["methods"] and all(k in man["methods"] for k in e["methods"])
    assert man["windows"] == [3, 6, 12, 24, 48, 0] and man["stats"] == list(config.ALL_STATS)
    assert e["window_times"]["24"] == [24] and e["window_times"]["3"][0] == 3 and e["window_times"]["0"] == e["fhrs"]
    meta, blocks = encode.read_bundle(tmp_path / "data" / e["files"]["blend"]["file"])
    g = man["grid"]
    W, S, T = len(config.WINDOWS_H), len(man["stats"]), 8
    assert blocks["amt"].shape == (W, S, T, g["ny"], g["nx"])
    assert blocks["prob"].shape == (W, len(man["thresholds"]), T, g["ny"], g["nx"])
    amt = blocks["amt"][man["windows"].index(0)]                   # run total
    st = {n: amt[i] for i, n in enumerate(man["stats"])}
    ok = ~np.isnan(st["min"])
    order = ["min", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "max"]
    for lo, hi in zip(order, order[1:]):                           # the percentiles are ordered at every point and time
        assert (st[lo][ok] <= st[hi][ok] + 1e-3).all(), (lo, hi)
    assert (st["min"][ok] <= st["mean"][ok] + 1e-3).all() and (st["mean"][ok] <= st["max"][ok] + 1e-3).all()
    assert (np.diff(np.nan_to_num(st["p50"]), axis=0) >= -0.06).all()   # the median total never falls with time (up to quantisation)
    for k in ("pmm", "lpmm", "sam", "samlpm"):                      # consensus products never exceed the snowiest member
        assert (st[k][ok] <= st["max"][ok] + 0.06).all(), k
    assert st["pmm"][ok].max() >= st["mean"][ok].max() - 1e-3      # and probability matching restores peaks the mean smooths away
    w24 = blocks["amt"][man["windows"].index(24)]                  # a 24-hour total exists only at F+24 h in a 24-hour run
    assert np.isnan(w24[0, :7]).all() and not np.isnan(w24[0, 7]).all()
    assert m["ensembles"]["all"]["members"] == sum(man["ensembles"][k]["members"] for k in ("eps", "gefs", "geps"))


def test_skip_if_deployed(monkeypatch):
    cycles = {"eps": datetime(2026, 11, 9, 0), "gefs": datetime(2026, 11, 9, 0)}
    fp = build.code_fingerprint()
    live = {"fingerprint": fp, "synthetic": False, "ensembles": {k: {"init": "2026-11-09T00:00:00Z"} for k in cycles}}

    class R:
        def __init__(self, body, status=200): self.body, self.status_code = body, status
        def json(self): return self.body

    class Http:
        def __init__(self, r): self.r = r
        def get(self, url, timeout=0): return self.r

    assert build.already_deployed("https://x.github.io/r", cycles, fp, Http(R(live)))
    assert not build.already_deployed("https://x.github.io/r", cycles, "other", Http(R(live)))              # code changed
    assert not build.already_deployed("https://x.github.io/r", {**cycles, "eps": datetime(2026, 11, 9, 12)}, fp, Http(R(live)))
    assert not build.already_deployed("https://x.github.io/r", cycles, fp, Http(R({}, 404)))


def test_parallel_and_serial_alignment_agree(monkeypatch):
    """Worker processes must produce exactly what the serial path does (they only share memory by forking)."""
    rng = np.random.default_rng(11)
    T, M, ny, nx = 4, 12, 15, 17
    gy, gx = np.mgrid[0:ny, 0:nx]
    cum = np.zeros((T, M, ny, nx), np.float32)
    for m in range(M):
        cx, cy = 8 + rng.normal(0, 2.5), 7 + rng.normal(0, 2)
        blob = 6 * np.exp(-(((gx - cx) ** 2) / 18 + ((gy - cy) ** 2) / 10))
        cum[:, m] = blob[None] * np.linspace(0.3, 1.0, T)[:, None, None] * rng.uniform(0.7, 1.3)
    fhrs = [3, 6, 9, 12]
    monkeypatch.setenv("ENSNOW_PROCESSES", "1")
    serial = build.compute_shifts(cum, fhrs, None)
    monkeypatch.setenv("ENSNOW_PROCESSES", "2")
    parallel = build.compute_shifts(cum, fhrs, None)
    assert set(serial) == set(config.WINDOWS_H)
    for w in serial:
        np.testing.assert_array_equal(serial[w], parallel[w])
    assert np.abs(serial[0]).max() > 0                              # the members really were displaced


def _counting(monkeypatch):
    """Count downloads (Source.load) and heavy computations (process_ensemble) in a build."""
    from ensemble_snow.sources.synthetic import SyntheticSource
    calls = {"load": 0, "process": 0}
    real_load, real_proc = SyntheticSource.load, build.process_ensemble

    def load(self, *a, **k):
        calls["load"] += 1
        return real_load(self, *a, **k)

    def proc(*a, **k):
        calls["process"] += 1
        return real_proc(*a, **k)

    monkeypatch.setattr(SyntheticSource, "load", load)
    monkeypatch.setattr(build, "process_ensemble", proc)
    return calls


def _run(tmp_path, name, **kw):
    return build.build_site(tmp_path / name, ["eps", "gefs"], synthetic=True, max_fhr=12, cache=tmp_path / "cache", force_cycle=datetime(2026, 11, 9, 0),
                            common=True, workers=2, with_gis=False, use_cache=True, **kw)


def test_unchanged_pipeline_reuses_the_finished_data_files(tmp_path, monkeypatch):
    """A page-only change must not download or recompute anything: the cached data files are reused and only the page is new."""
    calls = _counting(monkeypatch)
    first = _run(tmp_path, "a")
    assert calls == {"load": 2, "process": 2}
    monkeypatch.setattr(build, "code_fingerprint", lambda: "new-page-version")          # the viewer changed, the pipeline did not
    second = _run(tmp_path, "b")
    assert calls == {"load": 2, "process": 2}                                            # nothing fetched, nothing recomputed
    assert second["fingerprint"] == "new-page-version" and first["fingerprint"] != "new-page-version"
    assert second["ensembles"].keys() == first["ensembles"].keys()
    for e in first["ensembles"].values():                                                # identical data files
        for f in e["files"].values():
            assert (tmp_path / "a" / "data" / f["file"]).read_bytes() == (tmp_path / "b" / "data" / f["file"]).read_bytes()
    assert (tmp_path / "b" / "index.html").exists() and (tmp_path / "b" / ".nojekyll").exists()
    assert json.loads((tmp_path / "b" / "data" / "manifest.json").read_text())["fingerprint"] == "new-page-version"


def test_changed_processing_code_reuses_the_download_only(tmp_path, monkeypatch):
    """Changing the statistics / snow-ratio code recomputes the products but does not download again."""
    calls = _counting(monkeypatch)
    _run(tmp_path, "a")
    monkeypatch.setattr(build, "pipeline_fingerprint", lambda: "new-processing")         # fetch_fingerprint is untouched
    _run(tmp_path, "b")
    assert calls == {"load": 2, "process": 4}


def test_changed_fetch_code_or_new_cycle_downloads_again(tmp_path, monkeypatch):
    calls = _counting(monkeypatch)
    _run(tmp_path, "a")
    monkeypatch.setattr(build, "pipeline_fingerprint", lambda: "x")
    monkeypatch.setattr(build, "fetch_fingerprint", lambda: "y")                          # fetcher / GRIB reader / grid changed
    _run(tmp_path, "b")
    assert calls["load"] == 4
    build.build_site(tmp_path / "c", ["eps", "gefs"], synthetic=True, max_fhr=12, cache=tmp_path / "cache", force_cycle=datetime(2026, 11, 9, 12),
                     common=True, workers=2, with_gis=False, use_cache=True)             # a newer model run
    assert calls["load"] == 6


def test_cache_can_be_disabled_and_old_entries_are_pruned(tmp_path, monkeypatch):
    calls = _counting(monkeypatch)
    _run(tmp_path, "a")
    monkeypatch.setenv("ENSNOW_NO_CACHE", "1")
    _run(tmp_path, "b")
    assert calls["load"] == 4                                                             # --no-cache: everything fetched again
    monkeypatch.delenv("ENSNOW_NO_CACHE")
    dec = list((tmp_path / "cache" / "decoded").glob("eps_*.pkl.gz"))
    assert len(dec) == 1                                                                  # one download per system is kept
    assert len(list((tmp_path / "cache" / "bundles").iterdir())) == 1                     # and one finished set


def test_corrupt_cache_falls_back_to_downloading(tmp_path, monkeypatch):
    calls = _counting(monkeypatch)
    _run(tmp_path, "a")
    for p in (tmp_path / "cache" / "decoded").glob("*.pkl.gz"):
        p.write_bytes(b"not a gzip file")
    for p in (tmp_path / "cache" / "bundles").rglob("*.bin.gz"):
        p.unlink()                                                                        # incomplete finished set -> ignored
    _run(tmp_path, "b")
    assert calls["load"] == 4 and calls["process"] == 4
