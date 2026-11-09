"""The interface every ensemble source implements."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..fields import EnsembleData


class Source(Protocol):
    key: str

    def is_complete(self, cycle: datetime, max_fhr: int) -> bool:
        """True when the data for the last forecast hour we need has been published."""

    def load(self, cycle: datetime, steps: list[int], workers: int = 8) -> EnsembleData:
        """Download and decode the member fields for the forecast hours in ``steps`` (ascending; may include 0)."""


def forecast_steps(max_fhr: int, step: int = 3, with_zero: bool = True, coarse_after: int | None = None, coarse_step: int = 6) -> list[int]:
    """Forecast hours ``step`` apart, then ``coarse_step`` apart after hour ``coarse_after`` (ECMWF's 3-hourly fields end at 144 h)."""
    out = list(range(0 if with_zero else step, max_fhr + 1, step))
    if coarse_after is None:
        return out
    fine = [h for h in out if h <= coarse_after]
    last = fine[-1] if fine else 0
    return fine + list(range(last + coarse_step, max_fhr + 1, coarse_step))
