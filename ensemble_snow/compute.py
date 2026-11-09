"""Snowfall from ensemble data: for every member and every forecast interval,
``snow = snow-type liquid x ratio``, summed over time, once per snow-to-liquid-ratio method (and for the multi-method blend)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from . import config, slr, thermo
from .fields import EnsembleData, StepFields

log = logging.getLogger(__name__)
MM_PER_IN = 25.4


@dataclass
class SnowResult:
    fhrs: list[int]                                   # output hours (multiples of config.OUT_STEP_H, > 0)
    liquid_in: np.ndarray                             # (T, M, ny, nx) cumulative snow-type liquid, inches
    methods: dict[str, np.ndarray]                    # key -> (T, M, ny, nx) cumulative snowfall, inches ("blend" included)
    blend_members: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ptype_source: str = ""


def _column(step: StepFields, lat, lon, shape, precip_mm3h=None) -> slr.Column:
    return slr.Column(shape=shape, lat=lat, lon=lon, t2_k=step.t2m_k, ps_hpa=step.ps_hpa, t=step.t, rh=step.rh, z=step.z, precip_mm3h=precip_mm3h)


def _snow_fraction(step: StepFields, native: bool = True) -> tuple[np.ndarray, str]:
    """Instantaneous snow share at one step: the model's own flag if it has one (and ``native``), else the wet-bulb ramp."""
    if native and step.snow_frac is not None:
        return np.clip(step.snow_frac, 0.0, 1.0), "native flag"
    if step.t2m_k is None or (step.td2m_k is None and step.rh2m is None):
        raise ValueError(f"hour {step.fhr}: no precipitation-type flag and no 2 m temperature/humidity to estimate one")
    t_c = step.t2m_k - 273.15
    td_c = (step.td2m_k - 273.15) if step.td2m_k is not None else thermo.td_from_rh_c(t_c, step.rh2m)
    p = step.ps_hpa if step.ps_hpa is not None else 1000.0
    return thermo.snow_fraction_thermo(thermo.wet_bulb_c(t_c, td_c, p)), "wet-bulb ramp"


def compute_snowfall(ens: EnsembleData, lat: np.ndarray, lon: np.ndarray, methods: dict | None = None) -> SnowResult:
    methods = slr.METHODS if methods is None else methods
    steps = ens.steps
    if not steps or steps[-1].fhr <= 0:
        raise ValueError("no forecast steps")
    M = len(ens.members)
    ny, nx = lat.shape
    shape = (M, ny, nx)
    zero = np.zeros(shape, np.float32)

    prev: StepFields | None = steps[0] if steps[0].fhr == 0 else None
    prev_fhr = 0
    prev_precip = zero if prev is None or prev.precip_cum_mm is None else prev.precip_cum_mm
    prev_snow_liq = zero if prev is None or prev.snow_liq_cum_mm is None else prev.snow_liq_cum_mm
    prev_col = _column(prev, lat, lon, shape) if prev is not None else None

    cum_liq = zero.copy()
    cum = {k: zero.copy() for k in methods}
    cum["blend"] = zero.copy()
    avail: set[str] | None = None
    source_used: set[str] = set()
    out_fhrs, out_liq = [], []
    out_cum: dict[str, list] = {k: [] for k in cum}

    for st in steps:
        if st.fhr == 0:
            continue
        dt = st.fhr - prev_fhr
        if st.precip_cum_mm is None:
            raise ValueError(f"hour {st.fhr}: no precipitation")
        d_mm = np.clip(st.precip_cum_mm - prev_precip, 0.0, None)
        # --- snow-type liquid of the interval
        if st.snow_liq_cum_mm is not None:
            snow_mm = np.clip(st.snow_liq_cum_mm - prev_snow_liq, 0.0, None)
            snow_mm = np.minimum(snow_mm, d_mm * 1.001 + 1e-6)
            source_used.add("native snow accumulation")
        else:
            f1, src = _snow_fraction(st)
            native = src == "native flag"
            try:                                       # the start of the interval, by the same method as its end
                f0 = _snow_fraction(prev, native)[0] if prev is not None and (prev.snow_frac is not None if native else prev.t2m_k is not None) else f1
            except ValueError:
                f0 = f1
            snow_mm = d_mm * 0.5 * (f0 + f1)
            source_used.add(src)
        snow_liq_in = snow_mm / MM_PER_IN
        cum_liq += snow_liq_in

        # --- ratios
        col = _column(st, lat, lon, shape, precip_mm3h=d_mm * (3.0 / dt))
        ratios = {}
        for k, m in methods.items():
            if m.available(col):
                ratios[k] = slr.interval_ratio(m, prev_col, col)
        if avail is None:
            avail = set(ratios)
        else:
            avail &= set(ratios)
        for k, r in ratios.items():
            cum[k] += snow_liq_in * np.nan_to_num(r, nan=0.0)
        bl = [ratios[k] for k in config.BLEND if k in ratios]
        if bl:
            with np.errstate(all="ignore"):
                rb = np.nanmean(np.stack(bl), axis=0)
            cum["blend"] += snow_liq_in * np.nan_to_num(rb, nan=0.0)

        if st.fhr % config.OUT_STEP_H == 0:
            out_fhrs.append(st.fhr)
            out_liq.append(cum_liq.copy())
            for k in cum:
                out_cum[k].append(cum[k].copy())
        prev, prev_fhr, prev_precip, prev_col = st, st.fhr, st.precip_cum_mm, col
        prev_snow_liq = st.snow_liq_cum_mm if st.snow_liq_cum_mm is not None else prev_snow_liq

    avail = avail or set()
    notes = list(ens.notes)
    dropped = [k for k in methods if k not in avail]
    if dropped:
        notes.append("Not available for this system (missing inputs): " + ", ".join(methods[k].label for k in dropped))
    blend_members = [k for k in config.BLEND if k in avail]
    result_methods = {k: np.stack(v) for k, v in out_cum.items() if k == "blend" or k in avail}
    return SnowResult(out_fhrs, np.stack(out_liq), result_methods, blend_members, notes, " + ".join(sorted(source_used)))


def effective_ratio(snow_in: np.ndarray, liquid_in: np.ndarray) -> np.ndarray:
    """Run-total snow / snow-type liquid per member, NaN where less than ``MIN_LIQUID_FOR_SLR_IN`` of snow-liquid has fallen."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = snow_in / liquid_in
    return np.where(liquid_in >= config.MIN_LIQUID_FOR_SLR_IN, r, np.nan).astype(np.float32)
