"""A synthetic nor'easter, perturbed per member, delivered in the conventions of each real system.

Used by the tests and by ``--synthetic`` demo builds (the page flags these builds clearly).  The three flavours deliberately exercise the three
precipitation-type routes of :mod:`ensemble_snow.compute`:  ``eps`` -> native snow accumulation, ``gefs`` -> native snow flag with bucketed
precipitation accumulations, ``geps`` -> wet-bulb ramp.  Fields are generated on each system's own native grid and then regridded, so the
interpolation path is exercised as well.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from .. import config, thermo
from ..accum import CumulativeTracker
from ..fields import EnsembleData, Profile, StepFields
from ..regrid import Regridder

# (native lat step, native lon step, has native snow accumulation, has native snow flag, bucket accumulations)
FLAVOURS = {
    "eps": dict(res=0.25, native_snow_acc=True, native_flag=False, bucket=False, spread=0.8),
    "gefs": dict(res=0.5, native_snow_acc=False, native_flag=True, bucket=True, spread=1.1),
    "geps": dict(res=0.5, native_snow_acc=False, native_flag=False, bucket=False, spread=1.0),
}
SYN_LEVELS_T = config.T_LEVELS
SYN_LEVELS_RH = config.RH_LEVELS
SYN_LEVELS_Z = config.Z_LEVELS
# lapse of temperature relative to 2 m (K) by level
_DT = {1000: 0.8, 925: -5.0, 850: -10.0, 700: -20.5, 600: -29.0, 500: -38.5}


class SyntheticSource:
    def __init__(self, key: str, seed: int = 0):
        if key not in FLAVOURS:
            raise KeyError(key)
        self.key, self.seed = key, seed
        self.flav = FLAVOURS[key]
        self.n_members = config.ENSEMBLES[key].n_members

    def is_complete(self, cycle, max_fhr):
        return True

    # ------------------------------------------------------------------ physical fields
    @staticmethod
    def _elevation_m(lon, lat):
        def hill(x0, y0, h, sx, sy):
            return h * np.exp(-(((lon - x0) / sx) ** 2 + ((lat - y0) / sy) ** 2))
        return hill(-71.3, 44.2, 900, 0.5, 0.5) + hill(-72.8, 44.0, 800, 0.35, 0.8) + hill(-73.1, 42.3, 500, 0.3, 0.5) + hill(-69.5, 45.8, 500, 0.5, 0.5)

    def _rate_mm_h(self, lon, lat, t_h, p):
        """Precipitation rate (mm/h) of two systems for one member's parameters ``p``."""
        out = 0.0
        for k, (t0, wid, amp, x0, y0, vx, vy) in enumerate(((50.0, 16.0, 2.6, -77.5, 38.0, 0.115, 0.095), (132.0, 12.0, 1.1, -75.0, 41.0, 0.08, 0.05))):
            tt = t_h - (t0 + p["dt"])
            env = np.exp(-(tt / wid) ** 2)
            cx = x0 + vx * (t_h - p["dt"]) + p["dx"]
            cy = y0 + vy * (t_h - p["dt"]) + p["dy"]
            ax, ay = lon - cx, lat - cy
            para = (ax * 0.8 + ay * 0.6)                  # along the SW-NE axis
            perp = (-ax * 0.6 + ay * 0.8)
            out = out + amp * p["amp"] * env * np.exp(-(para ** 2 / (2 * 2.6 ** 2) + perp ** 2 / (2 * 0.95 ** 2)))
        return out

    def _t2m_k(self, lon, lat, t_h, p, elev):
        base = 273.15 + 2.6 - 1.15 * (lat - 42.0) - 0.45 * np.maximum(0.0, -lon - 70.8) - 0.0065 * elev
        storm = 1.0 * np.exp(-((t_h - 50.0 - p["dt"]) / 22.0) ** 2) * (1.0 / (1.0 + np.exp(-(lon + 70.8 + p["dx"] * 0.3) / 0.9)))
        diurnal = 1.2 * np.cos(2 * np.pi * (t_h - 15.0) / 24.0)
        return (base + 2.4 * storm + diurnal + p["dT"]).astype(np.float32)

    # ------------------------------------------------------------------ loading
    def load(self, cycle: datetime, steps: list[int], workers: int = 8) -> EnsembleData:
        rng = np.random.default_rng(self.seed * 1000 + sum(map(ord, self.key)) + cycle.hour)
        M = self.n_members
        res = self.flav["res"]
        pad = config.CROP_PAD_DEG
        nlon = np.round(np.arange(config.GRID_LON0 - pad, config.GRID_LON1 + pad + res / 2, res), 6)
        nlat = np.round(np.arange(config.GRID_LAT0 - pad, config.GRID_LAT1 + pad + res / 2, res), 6)
        LON, LAT = np.meshgrid(nlon, nlat)
        elev = self._elevation_m(LON, LAT)
        ps = thermo.std_atmosphere_hpa(elev).astype(np.float32)
        rg = Regridder(nlat, nlon)
        sp = self.flav["spread"]
        params = [dict(dx=rng.normal(0, 0.55 * sp), dy=rng.normal(0, 0.45 * sp), dt=rng.normal(0, 4.0 * sp), amp=float(np.exp(rng.normal(0, 0.28 * sp))),
                       dT=rng.normal(0, 0.9 * sp)) for _ in range(M)]
        params[0] = dict(dx=0.0, dy=0.0, dt=0.0, amp=1.0, dT=0.0)                   # the control

        # hourly (3-hourly sub-stepping) integration of precipitation and snow-type liquid on the native grid
        h_all = np.arange(0, max(steps) + 1, 3)
        cum_p = np.zeros((len(h_all), M) + LON.shape, np.float32)
        cum_s = np.zeros_like(cum_p)
        for i, h in enumerate(h_all):
            if i == 0:
                continue
            for m in range(M):
                r = self._rate_mm_h(LON, LAT, h - 1.5, params[m]).astype(np.float32) * 3.0
                t2 = self._t2m_k(LON, LAT, h - 1.5, params[m], elev)
                frac = thermo.snow_fraction_thermo(thermo.wet_bulb_c(t2 - 273.15, t2 - 273.15 - 1.5, ps))
                cum_p[i, m] = cum_p[i - 1, m] + r
                cum_s[i, m] = cum_s[i - 1, m] + r * frac

        tracker = CumulativeTracker()
        out_steps: list[StepFields] = []
        prev_h = 0
        for fh in steps:
            i = int(np.argmin(np.abs(h_all - fh)))
            P = cum_p[i]
            if self.flav["bucket"] and fh > 0:                     # NCEP style: windows that reset every 6 h, resolved back to totals
                tracker.add(prev_h, fh, P - cum_p[int(np.argmin(np.abs(h_all - prev_h)))])
                P = tracker.get(fh)
            prev_h = fh
            t2 = np.stack([self._t2m_k(LON, LAT, fh, params[m], elev) for m in range(M)])
            storm = np.stack([np.clip(self._rate_mm_h(LON, LAT, fh, params[m]) / 1.2, 0, 1) for m in range(M)]).astype(np.float32)
            rh2 = np.clip(55 + 45 * storm, 0, 100).astype(np.float32)
            tprof = np.stack([t2 + _DT[l] + (3.2 * storm * (l in (850, 925)) * (t2 > 272.0)) for l in SYN_LEVELS_T]).astype(np.float32)
            rhprof = np.stack([np.clip(45 + 55 * storm - 4 * k, 5, 100) for k, _ in enumerate(SYN_LEVELS_RH)]).astype(np.float32)
            # geopotential height by hypsometric integration from 1000 hPa
            z = {1000: np.full_like(t2, 110.0)}
            tl = dict(zip(SYN_LEVELS_T, tprof))
            for lo, hi in ((1000, 850), (850, 700)):
                tm = 0.5 * (tl[lo] + tl[hi]) if lo != 1000 else 0.5 * (tl[1000] + tl[925]) * 0.5 + 0.25 * (tl[925] + tl[850])
                z[hi] = z[lo] + thermo.RD_OVER_G * tm * np.log(lo / hi)
            sf = StepFields(fh)
            sf.precip_cum_mm = rg(P) if fh > 0 else rg(np.zeros_like(P))
            if self.flav["native_snow_acc"]:
                sf.snow_liq_cum_mm = rg(cum_s[i])
            if self.flav["native_flag"] and fh > 0:
                sf.snow_frac = rg(thermo.snow_fraction_thermo(thermo.wet_bulb_c(t2 - 273.15, t2 - 273.15 - 1.5, ps[None])) * (storm > 0.02))
            sf.t2m_k, sf.rh2m = rg(t2), rg(rh2)
            sf.ps_hpa = np.broadcast_to(rg(ps), (M,) + rg(ps).shape).copy()
            sf.t = Profile(SYN_LEVELS_T, rg(tprof))
            sf.rh = Profile(SYN_LEVELS_RH, rg(rhprof))
            sf.z = Profile(SYN_LEVELS_Z, rg(np.stack([z[l] for l in SYN_LEVELS_Z]).astype(np.float32)))
            out_steps.append(sf)
        return EnsembleData(self.key, cycle, list(range(M)), out_steps, ["Synthetic demonstration data: a made-up storm, not a forecast."], synthetic=True)
