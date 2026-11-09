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
    assert e["fhrs"] == [6, 12, 18, 24] and "blend" in e["methods"] and all(k in man["methods"] for k in e["methods"])
    meta, blocks = encode.read_bundle(tmp_path / "data" / e["files"]["blend"]["file"])
    g = man["grid"]
    W, S, T = len(config.WINDOWS_H), len(man["stats"]), 4
    assert blocks["amt"].shape == (W, S, T, g["ny"], g["nx"])
    assert blocks["prob"].shape == (W, len(man["thresholds"]), T, g["ny"], g["nx"])
    amt = blocks["amt"][0]                                         # run total
    p10, p50, p90, mx = amt[0], amt[2], amt[4], amt[6]
    ok = ~np.isnan(p10)
    assert (p10[ok] <= p50[ok] + 1e-3).all() and (p50[ok] <= p90[ok] + 1e-3).all() and (p90[ok] <= mx[ok] + 1e-3).all()
    assert (np.diff(np.nan_to_num(amt[2]), axis=0) >= -0.06).all()     # the median total never falls with time (up to quantisation)
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
