// Basemap and GIS layers: vector land/lakes on a water background, county and state lines, roads, and a decluttered label layer
// (cities, peaks, ski areas, route shields).  Every file is fetched the first time its layer is switched on.

import { shieldSvg, shieldSize } from "./shields.js";

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
// Stacking, bottom to top: land and lakes (behind the model field, z 250), then the water mask over the field, the coast, county and state
// lines, roads, and the labels.  Every line is drawn twice (a wider light or dark casing under the core) so it stays visible over any colour.
// Weights, in CSS pixels, from strongest to lightest: state and international lines 2.8 (casing 6.4), coast 2 (4.6), counties 1.4 (3.8).
// The shaded relief (an image, z 255) sits just over the model field, so it shades the field, and under the water mask.
const PANES = { land: 200, lakes: 210, water: 260, coast: 270, counties: 300, states: 310, roads: 320 };

export class Gis {
  constructor(map, base) {
    this.map = map; this.base = base; this.data = {}; this.lab = {}; this.on = new Set(); this.vec = {}; this.mask = false;
    for (const [n, z] of Object.entries(PANES)) { const p = map.createPane(n); p.style.zIndex = z; p.style.pointerEvents = "none"; }
    this.renderers = Object.fromEntries(Object.keys(PANES).map(n => [n, L.canvas({ pane: n, padding: 0.5 })]));
    const rp = map.createPane("relief"); rp.style.zIndex = 255; rp.style.pointerEvents = "none";
    this.reliefInfo = null; this.reliefOpacity = 0.6;
    this.labelPane = map.createPane("labels"); this.labelPane.style.zIndex = 400; this.labelPane.style.pointerEvents = "none";
    this.labelBox = L.DomUtil.create("div", "lbl-layer", this.labelPane);
    map.on("moveend zoomend", () => this.labels());
    map.on("zoomend", () => { if (this.on.has("roads")) { this.vec.roadsCase?.setStyle(f => this.roadStyle(f, true)); this.vec.roads?.setStyle(f => this.roadStyle(f, false)); } });
  }

  async get(name) {
    if (!this.data[name]) {
      this.data[name] = fetch(`${this.base}/${name}.json`).then(r => { if (!r.ok) throw new Error(`${name}: ${r.status}`); return r.json(); });
    }
    return this.data[name];
  }

  styles() {
    const fill = this.mask ? 0.94 : 0;
    return {
      land: { fill: true, fillColor: css("--land"), fillOpacity: 1, stroke: true, color: css("--coast"), weight: 1 },
      lakes: { fill: true, fillColor: css("--water"), fillOpacity: 1, stroke: true, color: css("--coast"), weight: 0.8 },
      sea: { fill: true, fillColor: css("--water"), fillOpacity: fill, stroke: false },
      lakesTop: { fill: true, fillColor: css("--water"), fillOpacity: fill, stroke: false },
      coastCase: { fill: false, color: css("--line-case"), weight: 4.6, opacity: 0.75, lineJoin: "round" },
      coast: { fill: false, color: css("--coast-line"), weight: 2, lineJoin: "round" },
      countiesCase: { fill: false, color: css("--line-case"), weight: 3.8, opacity: 0.6, lineJoin: "round" },
      counties: { fill: false, color: css("--county"), weight: 1.4, lineJoin: "round" },
      statesCase: { fill: false, color: css("--line-case"), weight: 6.4, opacity: 0.85, lineJoin: "round" },
      states: { fill: false, color: css("--state"), weight: 2.8, lineJoin: "round" },
    };
  }

  line(name, d, pane) {
    const mk = key => L.geoJSON(d, { pane, renderer: this.renderers[pane], interactive: false, style: () => this.styles()[key] });
    this.vec[name + "Case"] = mk(name + "Case"); this.vec[name] = mk(name);
  }

  async baseLayers() {
    const [land, lakes, sea] = await Promise.all([this.get("land"), this.get("lakes"), this.get("sea").catch(() => null)]);
    const add = (key, d, pane, style) => { this.vec[key] = L.geoJSON(d, { pane, renderer: this.renderers[pane], interactive: false, style: () => this.styles()[style || key] }).addTo(this.map); };
    add("land", land, "land"); add("lakes", lakes, "lakes");
    if (sea) add("sea", sea, "water");
    add("lakesTop", lakes, "water");
    this.line("coast", land, "coast");                      // land outline: the coast, drawn above the water mask
    this.vec.coastCase.addTo(this.map); this.vec.coast.addTo(this.map);
  }

  async set(id, show) {
    if (show) this.on.add(id); else this.on.delete(id);
    if (id === "counties" || id === "states") {
      if (show) {
        if (!this.vec[id]) {
          const d = await this.get(id); if (!this.on.has(id)) return;
          this.line(id, d, id);
        }
        this.vec[id + "Case"].addTo(this.map); this.vec[id].addTo(this.map);
      } else { this.vec[id + "Case"]?.remove(); this.vec[id]?.remove(); }
    } else if (id === "relief") {
      if (show && this.reliefInfo) {
        if (!this.vec.relief) {
          const [s0, w0, n0, e0] = this.reliefInfo.bounds;
          this.vec.relief = L.imageOverlay(this.reliefInfo.url, [[s0, w0], [n0, e0]], { pane: "relief", opacity: this.reliefOpacity, interactive: false });
        }
        if (this.on.has("relief")) this.vec.relief.addTo(this.map);
      } else this.vec.relief?.remove();
    } else if (id === "water") {
      this.mask = show; this.restyle();
    } else if (id === "roads") {
      if (show) {
        if (!this.vec.roads) {
          const d = await this.get("roads"); if (!this.on.has(id)) return;
          this.roadLabels = d.labels; this.roadZoom = d.minzoom || { 0: 5, 1: 8, 2: 9 };
          this.vec.roadsCase = L.geoJSON(d.lines, { pane: "roads", renderer: this.renderers.roads, interactive: false, style: f => this.roadStyle(f, true) });
          this.vec.roads = L.geoJSON(d.lines, { pane: "roads", renderer: this.renderers.roads, interactive: false, style: f => this.roadStyle(f, false) });
        }
        this.vec.roadsCase.addTo(this.map); this.vec.roads.addTo(this.map);
      } else { this.vec.roadsCase?.remove(); this.vec.roads?.remove(); }
    } else if (show) this.lab[id] = await this.get(id);       // places, peaks, ski: label sets
    this.labels();
  }

  // Interstates from the first zoom, U.S. routes from zoom 8 and state routes from 9; widths grow with zoom so the field stays readable when zoomed out.
  roadStyle(f, casing) {
    const z = this.map.getZoom(), c = f.properties.c;
    const base = [0.9, 0.6, 0.45][c], grow = [0.3, 0.22, 0.18][c], cap = [3.2, 2.2, 1.7][c];
    const w = Math.max(0.8, Math.min(cap, base + grow * (z - 6)));
    const on = z >= (this.roadZoom?.[c] ?? 5);
    return casing ? { color: css("--road-casing"), weight: w + 2, opacity: on ? 0.9 : 0, lineJoin: "round" }
                  : { color: css(["--road-i", "--road-us", "--road-st"][c]), weight: w, opacity: on ? 1 : 0, lineJoin: "round" };
  }

  setReliefOpacity(o) { this.reliefOpacity = o; this.vec.relief?.setOpacity(o); }

  restyle() {
    const st = this.styles();
    for (const k of Object.keys(st)) this.vec[k]?.setStyle(st[k]);
    this.vec.roadsCase?.setStyle(f => this.roadStyle(f, true));
    this.vec.roads?.setStyle(f => this.roadStyle(f, false));
  }

  // Greedy declutter: candidates in priority order (big cities, route shields, towns, peaks, ski areas), each kept only if its box misses every box kept so far.
  labels() {
    const m = this.map, z = m.getZoom(), box = this.labelBox;
    const b = m.getBounds().pad(0.05), kept = [], frag = document.createDocumentFragment();
    const fits = (x0, y0, w, h) => {
      for (const k of kept) if (x0 < k[2] && x0 + w > k[0] && y0 < k[3] && y0 + h > k[1]) return false;
      kept.push([x0, y0, x0 + w, y0 + h]); return true;
    };
    const place = (lat, lon, text, cls) => {
      if (!b.contains([lat, lon])) return false;
      const p = m.latLngToContainerPoint([lat, lon]);
      const w = text.length * (cls.includes("city") ? 7.6 : 6.8) + 16, h = 16;
      if (!fits(p.x - 4, p.y - h / 2, w, h)) return false;
      const el = document.createElement("div"); el.className = "lbl " + cls;
      const lp = m.containerPointToLayerPoint(p);
      el.style.left = lp.x - 4 + "px"; el.style.top = lp.y + "px";
      el.append(document.createElement("i"), document.createTextNode(text));
      frag.append(el); return true;
    };
    const near = new Map();                                   // where each route already has a shield, so one route is not repeated every few pixels
    const shield = (c, st, num, lat, lon) => {
      if (!b.contains([lat, lon])) return;
      const p = m.latLngToContainerPoint([lat, lon]), { pw: w, ph: h } = shieldSize(num), key = `${c}${st}${num}`;
      if ((near.get(key) || []).some(q => Math.hypot(q.x - p.x, q.y - p.y) < 140)) return;
      if (!fits(p.x - w / 2 - 2, p.y - h / 2 - 2, w + 4, h + 4)) return;
      (near.get(key) || near.set(key, []).get(key)).push(p);
      const el = document.createElement("div"); el.className = "lbl shield";
      const lp = m.containerPointToLayerPoint(p);
      el.style.left = lp.x + "px"; el.style.top = lp.y + "px";
      el.innerHTML = shieldSvg(c, st, num);
      frag.append(el);
    };
    const places = this.on.has("places") && this.lab.places;
    if (places) for (const [n, st, lat, lon, pop, mz] of places) if (pop > 50000 && z >= mz) place(lat, lon, n, "city");
    if (this.on.has("roads") && this.roadLabels) for (const [c, st, num, lon, lat, mz] of this.roadLabels) if (z >= mz) shield(c, st, num, lat, lon);
    if (places) for (const [n, st, lat, lon, pop, mz] of places) if (pop <= 50000 && z >= mz) place(lat, lon, n, "city small");
    if (this.on.has("peaks") && this.lab.peaks) {
      for (const [n, st, lat, lon, ft, mz] of this.lab.peaks) if (z >= mz + 5) place(lat, lon, `${n} ${ft.toLocaleString("en-US")} ft`, "peak");
    }
    if (this.on.has("ski") && this.lab.ski && z >= 8) for (const [n, st, lat, lon] of this.lab.ski) place(lat, lon, n, "ski");
    box.replaceChildren(frag);
  }
}

export function nearestPlace(places, lat, lon, maxKm = 20) {
  let best = null, bd = Infinity;
  const k = Math.cos(lat * Math.PI / 180);
  for (const p of places) {
    const dy = (p[2] - lat) * 111.2, dx = (p[3] - lon) * 111.2 * k, d = Math.hypot(dx, dy);
    if (d < bd && p[4] >= 5000) { bd = d; best = p; }
  }
  return best && bd <= maxKm ? { name: best[0], st: best[1], km: bd } : null;
}
