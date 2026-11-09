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


def forecast_steps(max_fhr: int, step: int = 6, with_zero: bool = True) -> list[int]:
    return list(range(0 if with_zero else step, max_fhr + 1, step))
