"""Central configuration: display grid, forecast steps, ensembles, statistics and the snowfall methods blended into the consensus.

Everything a maintainer is likely to tune lives here.  Nothing in this file touches the network.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- display grid
# A regular lat/lon grid.  0.25 deg is the finest native resolution of the three systems (GEPS is 0.5 deg, so it is
# interpolated); finer would only invent detail and make the files larger.  Row 0 is the SOUTHERN row.
GRID_LON0, GRID_LON1 = -75.0, -65.0
GRID_LAT0, GRID_LAT1 = 40.0, 48.0
GRID_STEP = 0.25

# Where the map opens: the six New England states (south, west, north, east).
NEW_ENGLAND_BOUNDS = (40.9, -73.8,47.6, -66.8)
NEW_ENGLAND_STATES = {"CT", "ME", "MA", "NH", "RI", "VT"}

# Native data are cropped to the display grid plus this margin before interpolation.
CROP_PAD_DEG = 1.0


def grid_axes() -> tuple[np.ndarray, np.ndarray]:
    lon = np.round(np.arange(GRID_LON0, GRID_LON1 + GRID_STEP / 2, GRID_STEP), 6)
    lat = np.round(np.arange(GRID_LAT0, GRID_LAT1 + GRID_STEP / 2, GRID_STEP), 6)
    return lon, lat


# --------------------------------------------------------------------------- forecast steps
OUT_STEP_H = 6                  # maps are produced every 6 h (accumulation windows are exact multiples of this)
DEFAULT_MAX_FHR = 168           # 7 days.  Beyond that the member spread is so wide that snowfall maps say little.
WINDOWS_H = (0, 24)             # 0 = run total (since initialisation); 24 = rolling 24-hour totals

# --------------------------------------------------------------------------- statistics shipped to the browser
PERCENTILES = (10, 25, 50, 75, 90)
THRESHOLDS_IN = (1, 3, 6, 12, 18, 24)          # exceedance probabilities: P(snowfall >= x inches)
AMT_STEP_IN = 0.05                             # u16 quantisation of snowfall (max 3276 in)
SLR_STEP = 0.2                                 # u8 quantisation of the effective ratio (max ~50:1)
MIN_LIQUID_FOR_SLR_IN = 0.02                   # effective ratio is only reported where this much snow-liquid has fallen

# --------------------------------------------------------------------------- precipitation type
# Used only where a system does not provide its own snow flag / snow accumulation: the snow share of the liquid falls
# linearly from 1 at a wet-bulb temperature of TW_ALL_SNOW to 0 at TW_ALL_RAIN (50 % at 1 C by default).
TW_ALL_SNOW_C = 0.0
TW_ALL_RAIN_C = 2.0

# --------------------------------------------------------------------------- Cobb layer ratio
COBB_TOP_HPA = 500              # NBM integrates to 300 hPa; 500 keeps the download small and identical for all three systems

# --------------------------------------------------------------------------- the consensus ("blend")
# The multi-method mean uses these methods (those a system cannot compute are dropped for that system and the manifest says so).
BLEND = ("fixed12", "kuchera", "maxt", "cmc_thick", "nbm_thick", "cobb")


# --------------------------------------------------------------------------- ensembles
@dataclass(frozen=True)
class Ensemble:
    key: str
    label: str            # long name
    short: str            # name used on the page
    center: str
    n_members: int
    cycles: tuple[int, ...]
    native: str
    license: str
    url: str


ENSEMBLES: dict[str, Ensemble] = {
    "eps": Ensemble("eps", "ECMWF Ensemble Prediction System (ENS)", "EPS", "ECMWF", 51, (0, 12), "0.25 deg, 51 members",
                    "CC BY 4.0 (ECMWF open data)", "https://www.ecmwf.int/en/forecasts/datasets/open-data"),
    "gefs": Ensemble("gefs", "NCEP Global Ensemble Forecast System", "GEFS", "NOAA / NCEP", 31, (0, 6, 12, 18),
                     "0.25 deg surface, 0.5 deg upper air, 31 members", "US government work (NOAA Open Data)",
                     "https://registry.opendata.aws/noaa-gefs/"),
    "geps": Ensemble("geps", "Global Ensemble Prediction System", "GEPS", "ECCC / CMC", 21, (0, 12), "0.5 deg, 21 members",
                     "Open Government Licence - Canada (MSC Datamart)", "https://eccc-msc.github.io/open-data/msc-data/nwp_geps/readme_geps_en"),
}

# Pressure levels fetched (hPa).  Kuchera / max-T-aloft / Cobb read T (and RH); the thickness methods read Z.
T_LEVELS = (1000, 925, 850, 700, 600, 500)
RH_LEVELS = (925, 850, 700, 600, 500)
Z_LEVELS = (1000, 850, 700)
