// Decoder for the ESN1 bundles written by ensemble_snow/encode.py:
//   "ESN1" | uint32 LE header length | header JSON | block payloads, the whole file gzip-compressed.
// Each block is quantised (value = lo + step * code, the top code means "missing"), delta-coded along x then y, and, for 16-bit data, byte-shuffled.

async function gunzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export function undelta(a, ny, nx) {
  // typed arrays wrap modulo 2^8 / 2^16 on assignment, which is exactly the arithmetic of the encoder
  const plane = ny * nx;
  for (let p = 0; p < a.length; p += plane) {
    for (let y = 1; y < ny; y++) {
      const r = p + y * nx, q = r - nx;
      for (let x = 0; x < nx; x++) a[r + x] += a[q + x];
    }
    for (let y = 0; y < ny; y++) {
      const r = p + y * nx;
      for (let x = 1; x < nx; x++) a[r + x] += a[r + x - 1];
    }
  }
  return a;
}

export function parseBundle(raw) {
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (String.fromCharCode(raw[0], raw[1], raw[2], raw[3]) !== "ESN1") throw new Error("not an ESN1 bundle");
  const hl = dv.getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(raw.subarray(8, 8 + hl)));
  const base = 8 + hl;
  const blocks = {};
  for (const b of head.blocks) {
    const n = b.dims.reduce((s, v) => s * v, 1);
    const pl = raw.subarray(base + b.offset, base + b.offset + b.nbytes);
    let data;
    if (b.dtype === "u16") {
      data = new Uint16Array(n);
      for (let i = 0; i < n; i++) data[i] = (pl[i] << 8) | pl[n + i];
    } else {
      data = pl.slice(0, n);
    }
    const [ny, nx] = b.dims.slice(-2);
    undelta(data, ny, nx);
    blocks[b.name] = { data, dims: b.dims, lo: b.lo, step: b.step, missing: b.missing };
  }
  return { meta: head.meta, blocks };
}

export async function loadBundle(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  let bytes = new Uint8Array(await res.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) bytes = await gunzip(bytes);     // some servers already undo the gzip
  return parseBundle(bytes);
}

// One 2-D field (ny*nx floats, NaN = missing) from a bundle.
//   kind "pct":  stat index s of window w at time t        kind "prob": threshold index k       kind "slr": run-total effective ratio
export function extract(bundle, spec, grid) {
  const plane = grid.ny * grid.nx;
  const out = new Float32Array(plane);
  let blk, off;
  if (spec.kind === "pct") {
    blk = bundle.blocks.amt; const [W, S, T] = blk.dims; off = (((spec.w * S + spec.s) * T) + spec.t) * plane;
  } else if (spec.kind === "prob") {
    blk = bundle.blocks.prob; const [W, K, T] = blk.dims; off = (((spec.w * K + spec.k) * T) + spec.t) * plane;
  } else {
    blk = bundle.blocks.slr; off = spec.t * plane;
  }
  const { data, lo, step, missing } = blk;
  for (let i = 0; i < plane; i++) { const q = data[off + i]; out[i] = q === missing ? NaN : lo + step * q; }
  return out;
}

// The statistics of one cell (bilinear in the grid) across all times, for the chart: {p10: [..T], ...}
export function series(bundle, grid, lat, lon, w) {
  const { fx, fy } = cellPos(grid, lat, lon);
  const blk = bundle.blocks.amt, [W, S, T, ny, nx] = blk.dims, plane = ny * nx;
  const x0 = Math.max(0, Math.min(nx - 2, Math.floor(fx))), y0 = Math.max(0, Math.min(ny - 2, Math.floor(fy)));
  const ax = Math.max(0, Math.min(1, fx - x0)), ay = Math.max(0, Math.min(1, fy - y0));
  const names = bundle.meta.stats, res = {};
  for (let s = 0; s < S; s++) {
    const arr = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      const o = (((w * S + s) * T) + t) * plane;
      const v = (xx, yy) => { const q = blk.data[o + yy * nx + xx]; return q === blk.missing ? NaN : blk.lo + blk.step * q; };
      arr[t] = (1 - ay) * ((1 - ax) * v(x0, y0) + ax * v(x0 + 1, y0)) + ay * ((1 - ax) * v(x0, y0 + 1) + ax * v(x0 + 1, y0 + 1));
    }
    res[names[s]] = arr;
  }
  return res;
}

export function cellPos(grid, lat, lon) { return { fx: (lon - grid.lon0) / grid.dlon, fy: (lat - grid.lat0) / grid.dlat }; }

// Bilinear value of a field at a point (NaN outside the grid or next to missing cells).
export function sample(field, grid, lat, lon) {
  const { fx, fy } = cellPos(grid, lat, lon);
  if (fx < 0 || fy < 0 || fx > grid.nx - 1 || fy > grid.ny - 1) return NaN;
  const x0 = Math.min(grid.nx - 2, Math.floor(fx)), y0 = Math.min(grid.ny - 2, Math.floor(fy));
  const ax = fx - x0, ay = fy - y0, n = grid.nx;
  const a = field[y0 * n + x0], b = field[y0 * n + x0 + 1], c = field[(y0 + 1) * n + x0], d = field[(y0 + 1) * n + x0 + 1];
  return (1 - ay) * ((1 - ax) * a + ax * b) + ay * ((1 - ax) * c + ax * d);
}
