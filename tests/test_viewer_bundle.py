"""The browser decoder (site/js/bundle.js) must read exactly what the Python encoder writes."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ensemble_snow import encode

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_decoder_matches_python(tmp_path):
    rng = np.random.default_rng(5)
    amt = np.round(rng.gamma(1.5, 3.0, size=(2, 3, 4, 9, 11)), 2).astype(np.float32)
    amt[1, 2, 3, 4, 5] = np.nan
    prob = rng.integers(0, 101, size=(2, 2, 4, 9, 11)).astype(np.float32)
    slr = np.round(rng.uniform(5, 25, size=(4, 9, 11)), 1).astype(np.float32)
    p = tmp_path / "f.bin.gz"
    encode.write_bundle(p, [{"name": "amt", "data": amt, "dtype": "u16", "lo": 0.0, "step": 0.05},
                            {"name": "prob", "data": prob, "dtype": "u8", "lo": 0.0, "step": 1.0},
                            {"name": "slr", "data": slr, "dtype": "u8", "lo": 0.0, "step": 0.2}], {"stats": ["a", "b", "c"]})
    _, py = encode.read_bundle(p)
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({k: [None if np.isnan(v) else float(v) for v in a.ravel()] for k, a in py.items()}))
    script = Path(__file__).parent / "bundle.test.mjs"
    r = subprocess.run([NODE, str(script), str(p), str(expected)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
