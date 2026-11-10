# New England snowfall ensembles

A static dashboard for snowfall percentiles and probabilities across New England from three ensemble systems:

| System | Centre | Members | Source used here |
|---|---|---|---|
| **EPS** (ENS) | ECMWF | 50 or 51 (the open-data index may omit the control member) | ECMWF open data, AWS mirror (`ecmwf-opendata`), CC BY 4.0 |
| **GEFS** | NOAA/NCEP | 31 | `noaa-gefs-pds` on AWS, byte-range reads guided by `.idx` files |
| **GEPS** | ECCC/CMC | 21 | MSC Datamart (`dd.weather.gc.ca`), Open Government Licence – Canada |

A Python pipeline turns each run into compact files; a vanilla-JS page (Leaflet, no CDN, no framework) shows them on a pannable, zoomable map with a vector basemap and GIS layers. GitHub Actions rebuilds the site after each 00/12 UTC run and deploys it to GitHub Pages. **No forecast data is committed to git.**

```
fetch GRIB2 ──► regrid to 0.25° New England grid ──► snow per member per SLR method ──► percentiles / mean / PMM / LPMM / SAM / SAM–LPM / P(≥ x in)
                                                                                         │
                          Pages artifact  ◄── manifest.json + one gzip bundle per (system, method) ◄┘
```

## What the page shows

* **Amount** – the minimum, 5th, 10th, 25th, 50th (the median), mean, 75th, 90th, 95th percentile or maximum member, for an accumulation period of **3, 6, 12, 24 or 48 hours, or the run total**. Time is a **slider in 3-hour steps** to F+144 h, then 6-hourly to F+168 h (ECMWF publishes 6-hourly fields after F+144 h, so 3-hour totals end there). A period only appears at times when it is complete: a 24-hour total starts at F+24 h.
* **Consensus** – four ways of drawing one map that keeps realistic amounts (see below): **PMM**, **LPMM**, **SAM** and **SAM–LPM**.
* **Chance** – the share of members reaching 0.1, 1, 2, 3, 4, 6, 8, 12, 18 or 24 in in the chosen period (the edges of the NWS snowfall legend).
* **Ratio** – the effective snow-to-liquid ratio of the chosen method (mean member).
* Colours: snowfall uses the NWS "Total Snow Forecast" legend of the Portland/Gray office (sampled from its map) in both appearances; chances use ColorBrewer **BuPu**, a cool sequential ramp that cannot be mistaken for the amounts.
* Ensemble: EPS, GEFS, GEPS, or **All** (the three pooled; each system carries equal total weight, so the 21 GEPS members count for more apiece than the 50-51 EPS members).
* Snow-to-liquid method (see below), per system.
* A bottom dock with the time slider and a **fan chart** (5–95 %, 10–90 % and 25–75 % bands, 50th percentile, mean, maximum) for the point you click. The chart is a second time control (click or drag it); ←/→ step the slider. A table view of all 14 statistics at the point is one button away.
* **Town search** – the box in the header completes town and city names as you type (add a state to narrow it: "wilton nh"). Picking one moves the chosen point and the map there; the chart and the readout then show the ensemble values at that point. The name is kept in the URL.
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

The deployed site is about 60 MB of static files: the page, vendored Leaflet and font, ~100 kB of GIS JSON (loaded lazily) and, per (system, method), a bundle of about 1.4–1.8 MB (it holds every period, statistic and time; about 60 MB for all four systems) that the browser fetches only when selected.

### Rebuilds that do not download the model again

Changing the page (anything in `site/`) or the statistics / snow-ratio code does **not** re-pull the ~18 GB of ECMWF data.  Two caches are kept between workflow runs (GitHub's `actions/cache`, never committed or published):

| Cache | Holds | Used when |
|---|---|---|
| `.cache/bundles` | the finished data files of the last build (~10 MB) | the Python pipeline and the model runs are unchanged, i.e. only `site/` changed: the build finishes in about a minute |
| `.cache/parts` | GEPS forecast hours already downloaded (a few MB each) | an attempt that failed half way: the next one fetches only the hours still missing |
| `.cache/decoded` | the newest download of each system, already reduced to the New England grid (~100-200 MB) | the fetchers, GRIB reader, grid and level settings are unchanged (`sources/*.py`, `grib.py`, `regrid.py`, `accum.py`, `thermo.py`, `http.py`, `fields.py`, grid settings in `config.py`): products are recomputed from it without any download |

A new model run, a change to the fetch code, or a corrupt / missing cache falls back to downloading.  To force a full refetch run the workflow by hand with **fresh_data** ticked (locally: `--no-cache`).  The caches keep only the newest download of each system, and GitHub evicts old snapshots by itself.

## Snow-to-liquid ratio (SLR) methods

Snow depth = Σ over the 3 h (6 h late in the run) intervals of (snow-type liquid) × (SLR). The snow-type liquid comes from the model's own snow accumulation or precipitation-type flag where it has one (EPS `sf`, GEPS `ASNOW` when it passes a unit check, GEFS `CSNOW`), otherwise from a wet-bulb ramp (all snow at Tw ≤ 0 °C, none at ≥ 2 °C). Ratios are evaluated at both ends of an interval and averaged. Every ratio is clipped to 3–40:1.

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
* **Accumulation.** Totals since initialisation are rebuilt from each GRIB message's `stepRange`, handling both from-start and 6-hourly-bucket conventions. Any 3, 6, 12, 24 or 48 h window is the difference of two totals.
* **Files.** `data/manifest.json` lists systems, methods, grid, times and files. Each `data/<system>_<method>.bin.gz` is an `ESN1` bundle (gzip → JSON header → 2-D-delta, byte-shuffled, quantised blocks: amounts 0.05 in, probabilities 1 %, ratios 0.2). `site/js/bundle.js` decodes it; `tests/test_viewer_bundle.py` checks the JavaScript against the Python encoder.
* **Statistics** are computed in Python over members (Hazen percentiles, weighted for the pooled view), so the browser never sees individual members.
* **ECMWF upper-air fields** (temperature, humidity, height: about three quarters of that download) are fetched every 6 h and interpolated in time for the 3-hourly steps in between; surface fields (precipitation, snowfall, 2 m temperature, dew point, pressure) are fetched every 3 h. Change `EPS_PL_STEP_H` in `config.py` to 3 to fetch everything (about 50 % more data).

## Consensus products: PMM, LPMM, SAM, SAM–LPM

All four are computed in `ensemble_snow/consensus.py` for every method, period and time, on the display grid, and shipped as ordinary fields. The plain mean smooths snowfall where members disagree about the storm track, so its peaks are too low and its swath too wide. These products repair that:

| Product | What it does | Source |
|---|---|---|
| **PMM** | Keeps the *pattern* of the ensemble mean and refits its *values* to the pooled distribution of every member over the map (the n-th largest mean receives the n-th largest pooled value). | Ebert 2001; NWS FDTD guide |
| **LPMM** | The same, patch by patch: 5 × 5-cell patches (about 140 × 100 km) take their pool from the 3 × 3 patches around them (about 375 km), so snow from one storm is not given to another. `LPMM_PATCH_CELLS` | Clark 2017; Snook et al. 2020 |
| **SAM** | Each member is moved toward the PMM by a smooth displacement field found in two passes (24-cell then 9-cell patches, search ±6 then ±3 cells, SOAR penalty on long shifts, α = 0.1), its own distribution of values is restored, and the moved members are averaged. `SAM_PASSES`, `SAM_ALPHA` | Lee et al. 2024, *Weather and Forecasting* 39, 1545–1558 |
| **SAM–LPM** | The aligned members combined by LPM instead of averaged. | Lee et al. 2024 |

Choices the papers leave open are mine: the patch sizes are the paper's 600 km and 225 km converted to this coarse grid (and its member-to-mean alignment); amounts are square-root transformed above 0.01 in before matching; and **the displacement fields are found once per period and time on the multi-method snowfall and applied to every method's members** (a storm's position error does not depend on the snow-to-liquid ratio), which keeps the cost independent of the number of methods. The pooled "All" view uses the member weights throughout. These maps are consensus pictures: they do not conserve totals and are not the likeliest outcome at each point, so the page says to use percentiles and chances for point decisions.

## Status of the data fetchers (read this)

The model fetchers (`sources/eps.py`, `gefs.py`, `geps.py`) were written from the providers' documentation in a sandbox that could not reach the model servers. The parsing, accumulation, regridding, statistics, encoding and page are tested (synthetic data in the native grids, real ecCodes GRIB2 round trips, byte-range planning); the **first live run is the real test**. Likely places to adjust, all isolated in those three files:

* ECMWF index field names and whether geopotential arrives as `gh` or `z` (`--probe` prints what the index lists).
* GEFS APCP bucket convention (handled generically from `stepRange`, not verified live).
* GEPS `ASNOW` units, `PRES_SFC_0` availability and `HGT` units (guarded by a plausibility check and unit autodetection).
* **Volume.** An EPS run at F+168 is on the order of 20 GB of GRIB2 through the client; GEFS reads only the needed byte ranges. This is heavy for a hosted runner. `--max-fhr 120`, fewer members or fewer variables are the first levers.
* MSC Datamart keeps only a short window of runs, so a missed build cannot be recovered later for GEPS.
* GEPS reliability.  GitHub's runners are sometimes refused by `dd.weather.gc.ca` (connections time out).  Connections give up after 15 s; six failed connections in a row mark a host down for a minute (`http.Fetcher`), and each file is then tried on the next server (`geps.MIRRORS`: `dd` /today, `hpfx.collab.science.gc.ca` dated, `dd` dated; the one that last worked goes first).  GEPS downloads use 4 connections, hours that fail are retried twice after a pause, and the whole download stops after 25 minutes.  Whatever arrived is used: the run is marked `partial` in the manifest (with a note on the page), is not stored as a finished download and never counts as "already deployed", so the next scheduled build tries again and resumes from `.cache/parts`.  A system whose data end before 60 % of the longest one's (`BLEND_MIN_SHARE`) stays in its own view but is left out of **All**, so it cannot cut the pooled forecast short.  The mirror paths were written from MSC's directory conventions and could not be tried from the sandbox.

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

Forecast data: ECMWF open data (CC BY 4.0), NOAA/NCEP (US government work), ECCC/MSC (Open Government Licence – Canada). Basemap: Natural Earth (public domain). Places and peaks: GeoNames (CC BY 4.0, places of 500 people or more), about 900 further town and village names from public ZIP-code place lists (`ensemble_snow/data/places.csv`, population 0 = not on file), and the lists inherited from the reference RRFS winter viewer. Leaflet (BSD-2-Clause) and Atkinson Hyperlegible Next (SIL OFL 1.1) are vendored with their licences. SLR formulas are generalised from that reference project; see the references in `ensemble_snow/slr.py`.

## Terrain adjustment (experimental, off by default)

The models see smooth terrain (about 25 km boxes for the ECMWF and GEFS data, 50 km for the GEPS data), so a ridge and the valley beside it get the same snow. Tick **Terrain adjustment** in the controls to scale the snowfall by elevation:

- `ensemble_snow/terrain.py` reads ground heights (AWS Terrarium tiles, zoom 8, about 600 m), averages them to 0.05° cells and, for each model grid, subtracts the same heights averaged over the model's box. The result, the relief in metres, goes to `data/terrain.json` (about 250 kB). The finished height array is kept in `.cache/dem`.
- `site/js/terrain.js` multiplies amounts by `1 + strength/100 × relief/100`, limited to 0.5–2. The default strength is 6 % per 100 m and the slider goes from 0 to 15. Chances of at least X inches use the model's chance of at least X divided by the factor, interpolated between the thresholds. The snow-to-liquid ratio is not changed.
- It rescales snow the model already has. It cannot move a storm, change rain to snow or add snow where the model has none. The strength is a rule of thumb that has not been fitted to observations; compare it against snow reports before relying on it.
- If the height download fails, the build continues without it and the control is hidden.

## Working with whatever data exist

The page must keep working when a model family, a forecast hour, a few members or a snow-to-liquid method is missing.

* **A family that fails** (no complete run, a download that errors) is left out and logged; the others are built and blended as usual. The page starts on **All** when it exists and says which family is not in the build.
* **Missing hours.** A forecast hour with no usable precipitation is skipped and the next hour covers the gap (a note says which hours). Members with no data at one hour are blank at that hour only.
* **The combined view keeps every family in the pool at every hour**, each with equal total weight, so the running totals only ever grow. A family that lacks an hour another has (a failed file, or the 6-hourly steps after F+144 h against 3-hourly ones) is filled in with the straight line between its neighbouring hours, and the time label says so. Dropping the family from those hours instead would make the pooled totals rise and fall with the pool's make-up. The pool ends where the earliest-ending family's data end (the later hours stay in each family's own view). A method that one family cannot compute (missing inputs) is pooled from the families that can.
* **Runs of different ages.** If the families cannot share one initialisation time (a run not yet complete), each uses its own newest complete run and the combined view lines them up by valid time: an older run is moved by its age and its snow is counted from the newest run's start. Runs more than `BLEND_MAX_LAG_H` (24 h) older than the newest are left out of the combined view but stay available on their own. Because an older run ends earlier on the common clock, the combined view then stops at that hour too.
* A dataset file that cannot be loaded in the browser falls back to the next best one with a note, instead of an empty map.
