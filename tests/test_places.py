"""The town list behind the map labels and the search box."""
import csv
import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import pytest

from ensemble_snow import gis

NODE = shutil.which("node")
STATES = {"CT", "MA", "ME", "NH", "RI", "VT"}


def _plain(s):
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFD", s).lower().replace("'", "").replace(".", "")).strip()


def test_places_cover_all_six_states_without_duplicates():
    rows = list(csv.DictReader(open(gis.DATA_DIR / "places.csv", newline="")))
    assert len(rows) > 1800
    assert {r["state"] for r in rows} == STATES
    keys = [(_plain(r["name"]), r["state"]) for r in rows]
    assert len(keys) == len(set(keys)), "a town is listed twice"
    for r in rows:
        assert 40.9 <= float(r["lat"]) <= 47.6 and -73.8 <= float(r["lon"]) <= -66.8, r
        assert int(r["pop"]) >= 0
    for want in ("Stowe", "Rangeley", "Killington", "Amherst", "North Conway"):
        assert any(r["name"] == want for r in rows), want


def test_places_file_is_small_and_sorted_big_first(tmp_path):
    gis.build_points(tmp_path)
    places = json.loads((tmp_path / "places.json").read_text())
    assert (tmp_path / "places.json").stat().st_size < 200_000
    assert places[0][0] == "Boston"
    assert [p[4] for p in places] == sorted((p[4] for p in places), reverse=True)
    small = [p for p in places if p[4] == 0]
    assert small and all(p[5] == 11 for p in small)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_search_matching(tmp_path):
    gis.build_points(tmp_path)
    script = Path(__file__).parent / "search.test.mjs"
    r = subprocess.run([NODE, str(script), str(tmp_path / "places.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
