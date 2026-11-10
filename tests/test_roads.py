"""Road and water layers: which Natural Earth records become lines and shields, and the sea polygon."""
import shutil
import subprocess
from pathlib import Path

import pytest
import shapely
from shapely.geometry import LineString, box

from ensemble_snow import gis

NODE = shutil.which("node")


def rec(cls, typ, number, state="Maine", country="United States"):
    return {"class": cls, "type": typ, "number": number, "state": state, "country": country}


@pytest.mark.parametrize("a, want", [
    (rec("Interstate", "Freeway", "95"), (0, "", "95")),
    (rec("Interstate", "Tollway", "90", "Massachusetts"), (0, "", "90")),
    (rec("Federal", "Primary", "2", "Vermont"), (1, "", "2")),
    (rec("State", "Primary", "101", "New Hampshire"), (2, "NH", "101")),
    (rec("State", "Secondary", "9 W", "New York"), (2, "NY", "9W")),
    (rec("State", "Freeway", "15", "Connecticut"), (2, "CT", "15")),
    (rec("Interstate", "Freeway", "10", "Quebec", "Canada"), None),            # Quebec autoroutes are filed as Interstate
    (rec("State", "Primary", "9", "Ohio"), None),                              # state routes only for the states shown
    (rec("State", "Other Paved", "9"), None),                                  # minor roads are not drawn
    (rec("Other", "Primary", ""), None),
    (rec("Interstate", "Secondary", "BUS25", "Colorado"), None),
])
def test_which_roads_are_drawn(a, want):
    assert gis._road_class(a) == want


def test_roads_file_has_lines_and_shield_labels_in_priority_order(tmp_path, monkeypatch):
    long = LineString([(-72.0, 41.0), (-72.0, 44.0)])
    short = LineString([(-71.0, 42.0), (-71.0, 42.1)])
    rows = [(rec("Interstate", "Freeway", "91", "Vermont"), long), (rec("State", "Primary", "9", "Maine"), long),
            (rec("Federal", "Primary", "5", "Vermont"), long), (rec("State", "Primary", "7", "Maine"), short),
            (rec("Interstate", "Freeway", "BUS91", "Vermont"), long)]
    monkeypatch.setattr(gis, "download", lambda name, cache: None)
    monkeypatch.setattr(gis, "records", lambda path, clip: iter(rows))
    gis.build_roads(tmp_path, tmp_path, box(-80, 38, -60, 50))
    import json
    d = json.loads((tmp_path / "roads.json").read_text())
    classes = sorted(f["properties"]["c"] for f in d["lines"]["features"])
    assert classes == [0, 0, 1, 2, 2]                                          # BUS91 keeps its line
    labels = d["labels"]
    assert [l[0] for l in labels] == sorted(l[0] for l in labels)               # interstates first, then US, then state
    assert {(l[0], l[1], l[2]) for l in labels} == {(0, "", "91"), (1, "", "5"), (2, "ME", "9")}     # no BUS91, and the short ME 7 is too short
    first91 = [l for l in labels if l[2] == "91"]
    assert len(first91) == 3 and sorted(l[5] for l in first91) == [7, 8, 8]      # a long route: one shield early, two more as you zoom in
    assert all(l[3] == -72.0 for l in labels)
    assert d["minzoom"] == {"0": 5, "1": 8, "2": 9}                              # JSON turns the class numbers into keys; the page indexes with them


def test_sea_is_the_box_minus_the_land(tmp_path, monkeypatch):
    land = box(-72, 40, -70, 42)
    monkeypatch.setattr(gis, "download", lambda name, cache: None)
    monkeypatch.setattr(gis, "records", lambda path, clip: iter([({}, land)]))
    clip = box(-75, 38, -66, 46)
    gis.build_sea(tmp_path, tmp_path, clip)
    import json
    from shapely.geometry import shape
    d = json.loads((tmp_path / "sea.json").read_text())
    sea = shapely.unary_union([shape(f["geometry"]) for f in d["features"]])
    assert sea.contains(shapely.Point(-74, 39)) and not sea.contains(shapely.Point(-71, 41))
    assert sea.area == pytest.approx(clip.area - land.area, rel=0.02)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_shields_are_drawn_per_class_and_state():
    script = Path(__file__).parent / "shields.test.mjs"
    r = subprocess.run([NODE, str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
