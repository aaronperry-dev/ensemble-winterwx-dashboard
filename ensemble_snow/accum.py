"""Accumulated model fields (precipitation) come in different conventions: from the start of the run (ECMWF, probably MSC), or in
buckets that reset every few hours (NCEP: "0-6", "6-12", ...).  :class:`CumulativeTracker` turns any of them into the total since
initialisation, using the (start, end) hour pair the GRIB message carries.
"""
from __future__ import annotations

import re

import numpy as np


def parse_step_range(text: str, end_hint: int | None = None) -> tuple[int, int]:
    """'6-12' -> (6, 12); '24' -> (0, 24) (a bare step is taken as accumulated from the start of the forecast)."""
    t = str(text).strip()
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", t)
    if m:
        return int(m.group(1)), int(m.group(2))
    if re.fullmatch(r"\d+", t):
        return 0, int(t)
    if end_hint is not None:
        return 0, int(end_hint)
    raise ValueError(f"cannot read step range {text!r}")


class MissingWindow(Exception):
    """A window starts at an hour for which no total is known (a step is missing from the download)."""


class CumulativeTracker:
    def __init__(self):
        self.cum: dict[int, np.ndarray | float] = {0: 0.0}
        self._pending: list[tuple[int, int, np.ndarray]] = []
        self._start_of: dict[int, int] = {}

    def add(self, start: int, end: int, value: np.ndarray) -> None:
        if end == start:                                 # an instantaneous record (e.g. hour 0): nothing accumulated yet
            self.cum.setdefault(end, 0.0)
            return
        self._pending.append((start, end, value))
        self._resolve()

    def _resolve(self) -> None:
        progress = True
        while progress:
            progress = False
            for item in list(self._pending):
                start, end, value = item
                if start in self.cum:
                    # several windows may end at the same hour (e.g. "0-6" and "3-6"): the one starting earliest wins
                    if end not in self.cum or self._start_of.get(end, 10**9) > start:
                        self.cum[end] = self.cum[start] + value
                        self._start_of[end] = start
                    self._pending.remove(item)
                    progress = True

    def get(self, end: int) -> np.ndarray | float:
        if end not in self.cum:
            raise MissingWindow(f"no accumulation ending at hour {end}")
        return self.cum[end]

    def unresolved(self) -> list[tuple[int, int]]:
        return [(s, e) for s, e, _ in self._pending]
