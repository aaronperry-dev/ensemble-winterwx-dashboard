// A Leaflet layer that paints a regular lat/lon grid (bilinear or nearest) through a classed colour scale.
// The lon/lat of a screen pixel in Web Mercator depends on x (lon) or y (lat) alone, so only W + H projections are needed per redraw.
import { rgbaPacked } from "./colors.js";

export const FieldLayer = L.Layer.extend({
  options: { pad: 0.1, opacity: 0.85 },

  initialize(opts) { L.setOptions(this, opts); this._data = null; },

  onAdd(map) {
    this._map = map;
    if (!map.getPane("field")) { const p = map.createPane("field"); p.style.zIndex = 250; }
    this._canvas = L.DomUtil.create("canvas", "field-canvas", map.getPane("field"));
    this._canvas.style.opacity = this.options.opacity;
    map.on("moveend zoomend resize", this._reset, this);
    this._reset();
  },
  onRemove(map) { map.off("moveend zoomend resize", this._reset, this); L.DomUtil.remove(this._canvas); },

  setOpacity(o) { this.options.opacity = o; if (this._canvas) this._canvas.style.opacity = o; },
  // data: { values: Float32Array(ny*nx), grid: {lon0, lat0, dlon, dlat, nx, ny}, scale, smooth }
  setData(d) { this._data = d; this.redraw(); },

  _reset() {
    const m = this._map, s = m.getSize(), px = Math.round(s.x * this.options.pad), py = Math.round(s.y * this.options.pad);
    this._pad = { x: px, y: py };
    const w = s.x + 2 * px, h = s.y + 2 * py, dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    this._scale = dpr;
    this._canvas.width = Math.round(w * dpr); this._canvas.height = Math.round(h * dpr);
    this._canvas.style.width = w + "px"; this._canvas.style.height = h + "px";
    L.DomUtil.setPosition(this._canvas, m.containerPointToLayerPoint([-px, -py]));
    this.redraw();
  },

  redraw() {
    const cv = this._canvas; if (!cv) return;
    const ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const d = this._data; if (!d) return;
    const { values, grid, scale, smooth } = d, { nx, ny } = grid, m = this._map, k = this._scale;
    const fx = new Float32Array(W), fy = new Float32Array(H);
    for (let x = 0; x < W; x++) fx[x] = (m.containerPointToLatLng([x / k - this._pad.x, 0]).lng - grid.lon0) / grid.dlon;
    for (let y = 0; y < H; y++) fy[y] = (m.containerPointToLatLng([0, y / k - this._pad.y]).lat - grid.lat0) / grid.dlat;
    const img = ctx.createImageData(W, H), px = new Uint32Array(img.data.buffer);
    const packed = scale.rgb.map(c => rgbaPacked(c)), edges = scale.edges, ne = edges.length;
    const cls = v => { if (!(v >= edges[0])) return -1; let c = ne - 1; while (v < edges[c]) c--; return c; };
    for (let y = 0; y < H; y++) {
      const gy = fy[y]; if (gy < 0 || gy > ny - 1) continue;
      const y0 = Math.min(ny - 2, Math.floor(gy)), ay = gy - y0, row = y * W;
      for (let x = 0; x < W; x++) {
        const gx = fx[x]; if (gx < 0 || gx > nx - 1) continue;
        let v;
        if (smooth) {
          const x0 = Math.min(nx - 2, Math.floor(gx)), ax = gx - x0, i = y0 * nx + x0;
          v = (1 - ay) * ((1 - ax) * values[i] + ax * values[i + 1]) + ay * ((1 - ax) * values[i + nx] + ax * values[i + nx + 1]);
        } else {
          v = values[Math.round(gy) * nx + Math.round(gx)];
        }
        const c = cls(v);
        if (c >= 0) px[row + x] = packed[c];
      }
    }
    ctx.putImageData(img, 0, 0);
  },
});
