"""Plain data containers shared by the sources and the snowfall computation.  All arrays are on the display grid.

Shapes:  surface fields ``(M, ny, nx)`` (M = members);  profiles ``(L, M, ny, nx)`` on ``levels`` (hPa, any order).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Profile:
    levels: tuple[int, ...]
    data: np.ndarray                      # (L, M, ny, nx)

    def __post_init__(self):
        if self.data.shape[0] != len(self.levels):
            raise ValueError("profile levels do not match the data")

    def sorted_desc(self) -> "Profile":
        """Bottom (highest pressure) first."""
        order = np.argsort([-p for p in self.levels])
        return Profile(tuple(self.levels[i] for i in order), self.data[order])

    def at(self, level: int) -> np.ndarray | None:
        return self.data[self.levels.index(level)] if level in self.levels else None


@dataclass
class StepFields:
    """Everything the snowfall computation needs at one forecast hour."""
    fhr: int
    precip_cum_mm: np.ndarray | None = None        # liquid-equivalent precipitation since initialisation
    snow_liq_cum_mm: np.ndarray | None = None      # the model's own snow-type liquid since initialisation (if it publishes one)
    snow_frac: np.ndarray | None = None            # instantaneous snow share of the precipitation, 0..1 (if the model publishes a type)
    t2m_k: np.ndarray | None = None
    rh2m: np.ndarray | None = None                 # percent
    td2m_k: np.ndarray | None = None
    ps_hpa: np.ndarray | None = None
    t: Profile | None = None                       # K
    rh: Profile | None = None                      # percent
    z: Profile | None = None                       # geopotential height, gpm


@dataclass
class EnsembleData:
    key: str
    cycle: datetime                                # initialisation time (UTC, naive)
    members: list[int]
    steps: list[StepFields]                        # ascending fhr, includes fhr 0 when the system publishes it
    notes: list[str] = field(default_factory=list)  # what was estimated / missing, shown on the page
    ptype_source: str = ""                         # "native flag" | "native snow accumulation" | "wet-bulb ramp"
    synthetic: bool = False

    @property
    def fhrs(self) -> list[int]:
        return [s.fhr for s in self.steps]
