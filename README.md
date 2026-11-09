# New England snowfall ensembles

A static dashboard for snowfall percentiles and probabilities across New England from three ensemble systems:

| System | Centre | Members | Source used here |
|---|---|---|---|
| **EPS** (ENS) | ECMWF | 51 | ECMWF open data, AWS mirror (`ecmwf-opendata`), CC BY 4.0 |
| **GEFS** | NOAA/NCEP | 31 | `noaa-gefs-pds` on AWS, byte-range reads guided by `.idx` files |
| **GEPS** | ECCC/CMC | 21 | MSC Datamart (`dd.weather.gc.ca`), Open Government Licence – Canada |

A Python pipeline turns each run into compact files; a vanilla-JS page (Leaflet, no CDN, no framework) shows them on a pannable, zoomable map with a vector basemap and GIS layers. GitHub Actions rebuilds the site after each 00/12 UTC run and deploys it to GitHub Pages. **No forecast data is committed to git.**

```
fetch GRIB2 ──► regrid to 0.25° New England grid ──► snow per member per SLR method ──► percentiles / mean / max / P(≥ x in)
                                                                                         │
                          Pages artifact  ◄── manifest.json + one gzip bundle per (system, method) ◄┘
```

## What the page shows

* **Amount** – 10th, 25th, 50th, 75th, 90th percentile, mean or maximum snowfall, as a run total or a rolling 24 h total, every 6 h to F+168.
* **Chance** – the share of members reaching 1, 3, 6, 12, 18 or 24 in.
* **Ratio** – the effective snow-to-liquid ratio of the chosen method (median member).
* Ensemble: EPS, GEFS, GEPS, or **All** (the three pooled; each system carries equal total weight, so the 21 GEPS members count for more apiece than the 51 EPS members).
* Snow-to-liquid method (see below), per system.
* A bottom dock with a **fan chart** (10–90 % and 25–75 % bands, median, mean, maximum) for the point you click. The chart is also the time control: click or drag it, or use ←/→. A table view of the same numbers is one button away.
* Layers: counties, state lines, highways with route shields, towns, peaks, ski areas. Light and dark appearance. All state is in the URL hash, so a view can be shared.

## Run it locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                                   # 27 tests, incl. the browser decoder under node

python -m ensemble_snow.build --out public --synthetic   # demo data (a made-up nor'easter), no network needed for the models
python -m http.server -d public 8000                     # open http://localhost:8000
```

The first build downloads Natural Earth shapefiles (~40 MB, cached in `.cache/`) for the basemap.

**Before trusting a live build**, run `python -m ensemble_snow.build --probe`. It reports which cycles are on each server and which parameters the ECMWF index lists. See *Status of the data fetchers* below.

```bash
python -m ensemble_snow.build --out public                 # newest complete run common to all systems
python -m ensemble_snow.build --out public --independent   # each system's own newest run
python -m ensemble_snow.build --ensembles gefs --max-fhr 96 --workers 16
```

## Deploying to github.io

1. Push this repository to GitHub. In **Settings → Pages**, set the source to **GitHub Actions**.
2. Run **Actions → Update snowfall ensemble dashboard → Run workflow** once with *probe* ticked to check the data sources, then once with *force* ticked to build and publish.
3. After that the workflow runs by itself at about 08:30 and 20:30 UTC (the time the 00/12 UTC ECMWF run is out), with retries an hour and two later. An attempt that finds the live site already showing the newest runs from the same code finishes in about a minute without deploying (`--skip-if-deployed`). A push to `main` rebuilds too.

The deployed site is a few MB of static files: the page, vendored Leaflet and font, ~100 kB of GIS JSON (loaded lazily) and, per (system, method), a bundle of roughly 150–250 kB that the browser fetches only when selected.

## Snow-to-liquid ratio (SLR) methods

Snow depth = Σ over 6 h intervals of (snow-type liquid) × (SLR). The snow-type liquid comes from the model's own snow accumulation or precipitation-type flag where it has one (EPS `sf`, GEPS `ASNOW` when it passes a unit check, GEFS `CSNOW`), otherwise from a wet-bulb ramp (all snow at Tw ≤ 0 °C, none at ≥ 2 °C). Ratios are evaluated at both ends of an interval and averaged. Every ratio is clipped to 3–40:1.

| Key | Method | Needs | Notes |
|---|---|---|---|
| `fixed10`, `fixed12` | 10:1, 12:1 | – | references |
| `kuchera` | Kuchera | T2m, T profile | 12 + (271.16 K − Tmax), doubled penalty when warmer |
| `maxt` | Max-T-aloft | T2m, T profile | Veals et al. (2025) polynomial. **Experimental**: coefficients are those of the reference project; fitted to Alaska data |
| `cmc_thick` | CMC thickness | T profile | Murphy-factor style layer thickness |
| `nbm_thick` | NBM thickness | T profile | NBM winter-suite thickness bins |
| `byun` | Byun | T2m, interval precipitation | depends on the interval's own rate |
| `climo` | Climatological | – | Baxter et al. (2005) regional mean; ignores the forecast |
| `cobb` | Cobb | T and RH profiles | layer-weighted, **without the lift term** (no vertical velocity here) |
| `blend` | Multi-method mean | – | mean ratio of `config.BLEND` (12:1, Kuchera, Max-T, CMC, NBM, Cobb) |

A method a system cannot compute (a missing input) is dropped for that system and the page only offers what exists.

### Adding a method

Everything is driven by the registry in `ensemble_snow/slr.py`; the page reads the manifest, so **no front-end change is needed**.

```python
# ensemble_snow/slr.py
@register("mymethod", "My method", ("t2_k", "t"),           # key, label, the Column inputs it needs
          desc="One sentence shown under the method menu.", ref="Author (year)")
def slr_mymethod(col):                                       # col: Column with arrays shaped (members, ny, nx)
    tmax = tmax_column_k(col.t2_k, col.t, col.ground_hpa())  # helpers: tmax_column_k, tmax_aloft_c, _thickness
    return _clip(12.0 + 0.8 * (273.15 - tmax))               # inches of snow per inch of liquid
```

Optionally add the key to `config.BLEND` to include it in the consensus, and add a case to `tests/test_slr.py`. Other `Column` inputs: `t2_k`, `ps_hpa`, `t`, `rh`, `z` (profiles on pressure levels, `Profile.at(level)`), and `precip_mm3h` for `interval=True` methods.

## Data pipeline details

* **Grid.** −75…−65 °E, 40…48 °N at 0.25° (41 × 33 cells). Native fields are cropped and bilinearly regridded with two matrix products.
* **Accumulation.** Totals since initialisation are rebuilt from each GRIB message's `stepRange`, handling both from-start and 6-hourly-bucket conventions.
* **Files.** `data/manifest.json` lists systems, methods, grid, times and files. Each `data/<system>_<method>.bin.gz` is an `ESN1` bundle (gzip → JSON header → 2-D-delta, byte-shuffled, quantised blocks: amounts 0.05 in, probabilities 1 %, ratios 0.2). `site/js/bundle.js` decodes it; `tests/test_viewer_bundle.py` checks the JavaScript against the Python encoder.
* **Statistics** are computed in Python over members (Hazen percentiles, weighted for the pooled view), so the browser never sees individual members.

## Status of the data fetchers (read this)

The model fetchers (`sources/eps.py`, `gefs.py`, `geps.py`) were written from the providers' documentation in a sandbox that could not reach the model servers. The parsing, accumulation, regridding, statistics, encoding and page are tested (synthetic data in the native grids, real ecCodes GRIB2 round trips, byte-range planning); the **first live run is the real test**. Likely places to adjust, all isolated in those three files:

* ECMWF index field names and whether geopotential arrives as `gh` or `z` (`--probe` prints what the index lists).
* GEFS APCP bucket convention (handled generically from `stepRange`, not verified live).
* GEPS `ASNOW` units, `PRES_SFC_0` availability and `HGT` units (guarded by a plausibility check and unit autodetection).
* **Volume.** An EPS run at F+168 is on the order of 20 GB of GRIB2 through the client; GEFS reads only the needed byte ranges. This is heavy for a hosted runner. `--max-fhr 120`, fewer members or fewer variables are the first levers.
* MSC Datamart keeps only a short window of runs, so a missed build cannot be recovered later for GEPS.

## Caveats

* 0.25° (and 0.5° for GEFS upper air) does not resolve terrain: mountain snowfall is underestimated and valleys overestimated. The ratio methods are generic, not calibrated to these models.
* The ensembles are raw model output, not post-processed or bias corrected. Percentiles describe the members, not calibrated probabilities.
* The synthetic mode exists to exercise the code; its numbers mean nothing and the page shows a banner when it is used.

## Layout

```
ensemble_snow/       pipeline (config, slr, compute, stats, encode, build, gis) and sources/ (eps, gefs, geps, synthetic)
site/                the page: index.html, app.css, js/ (bundle, colors, field-layer, layers, chart, main), vendor/ (Leaflet, font)
tests/               pytest (+ node decoder check)
.github/workflows/   update.yml (build and deploy), tests.yml
```

## Credits and licences

Forecast data: ECMWF open data (CC BY 4.0), NOAA/NCEP (US government work), ECCC/MSC (Open Government Licence – Canada). Basemap: Natural Earth (public domain). Places and peaks: GeoNames (CC BY 4.0) and the lists inherited from the reference RRFS winter viewer. Leaflet (BSD-2-Clause) and Atkinson Hyperlegible Next (SIL OFL 1.1) are vendored with their licences. SLR formulas are generalised from that reference project; see the references in `ensemble_snow/slr.py`.
