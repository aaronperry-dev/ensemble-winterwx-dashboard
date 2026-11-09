// Terrain adjustment: scale the model's snowfall by how much higher (or lower) the real ground is than the smooth terrain the model sees.
// A rule of thumb, not a downscaling model: it changes amounts only, never where or when the snow falls and never rain against snow.
//   relief dz (m) = mean ground height of a ~5 km cell minus the same heights averaged over the model's grid box   (made by ensemble_snow/terrain.py)
//   factor        = 1 + strength/100 * dz/100, limited to 0.5 ... 2          (strength is a percentage per 100 m)

const int16 = b64 => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Int16Array(u.buffer); };

export async function loadTerrain(url) {
  const res = await fetch(url); if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const j = await res.json();
  return { grid: j.grid, elev: int16(j.elev), dz: Object.fromEntries(Object.entries(j.dz).map(([k, v]) => [k, int16(v)])) };
}

export const factor = (dz, strength) => Math.max(0.5, Math.min(2, 1 + (strength / 100) * (dz / 100)));

// A coarse model field on the fine terrain grid (bilinear; NaN where any of the four neighbours is missing).
export function toFine(values, grid, fine) {
  const out = new Float32Array(fine.nx * fine.ny), { nx, ny } = grid;
  for (let y = 0; y < fine.ny; y++) {
    const gy = Math.max(0, Math.min(ny - 1, (fine.lat0 + y * fine.dlat - grid.lat0) / grid.dlat)), y0 = Math.min(ny - 2, Math.floor(gy)), ay = gy - y0;
    for (let x = 0; x < fine.nx; x++) {
      const gx = Math.max(0, Math.min(nx - 1, (fine.lon0 + x * fine.dlon - grid.lon0) / grid.dlon)), x0 = Math.min(nx - 2, Math.floor(gx)), ax = gx - x0, i = y0 * nx + x0;
      out[y * fine.nx + x] = (1 - ay) * ((1 - ax) * values[i] + ax * values[i + 1]) + ay * ((1 - ax) * values[i + nx] + ax * values[i + nx + 1]);
    }
  }
  return out;
}

// Amounts (percentiles, consensus products): the same field on the fine grid, each cell times its factor.
export function adjustAmount(values, grid, T, dz, strength) {
  const v = toFine(values, grid, T.grid);
  for (let i = 0; i < v.length; i++) v[i] *= factor(dz[i], strength);
  return { values: v, grid: T.grid };
}

// Chance of at least X inches: at a cell whose snowfall is scaled by f, that is the model's chance of at least X/f.  The model chances are known at
// the fixed thresholds only, so the in-between value is interpolated (linearly in the logarithm of the threshold); beyond the ends it is held at
// the first value, or decays with the square of the threshold above the last.
export function adjustChance(fields, thresholds, idx, grid, T, dz, strength) {
  const K = thresholds.length, fine = fields.map(f => toFine(f, grid, T.grid)), n = fine[0].length, out = new Float32Array(n), lt = thresholds.map(Math.log);
  for (let i = 0; i < n; i++) {
    const tt = thresholds[idx] / factor(dz[i], strength);
    let p;
    if (tt <= thresholds[0]) p = fine[0][i];
    else if (tt >= thresholds[K - 1]) p = fine[K - 1][i] * (thresholds[K - 1] / tt) ** 2;
    else {
      let k = 0; while (thresholds[k + 1] < tt) k++;
      const a = (Math.log(tt) - lt[k]) / (lt[k + 1] - lt[k]);
      p = fine[k][i] + a * (fine[k + 1][i] - fine[k][i]);
    }
    out[i] = Math.max(0, Math.min(100, p));
  }
  return { values: out, grid: T.grid };
}
