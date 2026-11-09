// Basemap and GIS layers: vector land/lakes on a water background, county and state lines, roads, and a decluttered label layer
// (cities, peaks, ski areas, route shields).  Every file is fetched the first time its layer is switched on.

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const PANES = { land: 200, lakes: 210, counties: 300, states: 310, roads: 320 };

export class Gis {
  constructor(map, base) {
    this.map = map; this.base = base; this.data = {}; this.lab = {}; this.on = new Set(); this.vec = {};
    for (const [n, z] of Object.entries(PANES)) { const p = map.createPane(n); p.style.zIndex = z; p.style.pointerEvents = "none"; }
    this.renderers = Object.fromEntries(Object.keys(PANES).map(n => [n, L.canvas({ pane: n, padding: 0.5 })]));
    this.labelPane = map.createPane("labels"); this.labelPane.style.zIndex = 400; this.labelPane.style.pointerEvents = "none";
    this.labelBox = L.DomUtil.create("div", "lbl-layer", this.labelPane);
    map.on("moveend zoomend", () => this.labels());
    map.on("zoomend", () => { if (this.on.has("roads")) this.restyle(); });
  }

  async get(name) {
    if (!this.data[name]) {
      this.data[name] = fetch(`${this.base}/${name}.json`).then(r => { if (!r.ok) throw new Error(`${name}: ${r.status}`); return r.json(); });
    }
    return this.data[name];
  }

  styles() {
    return {
      land: { fill: true, fillColor: css("--land"), fillOpacity: 1, stroke: true, color: css("--coast"), weight: 0.8 },
      lakes: { fill: true, fillColor: css("--water"), fillOpacity: 1, stroke: true, color: css("--coast"), weight: 0.6 },
      counties: { fill: false, color: css("--county"), weight: 0.7 },
      states: { fill: false, color: css("--state"), weight: 1.4 },
    };
  }

  async baseLayers() {
    const [land, lakes] = await Promise.all([this.get("land"), this.get("lakes")]);
    this.vec.land = L.geoJSON(land, { pane: "land", renderer: this.renderers.land, interactive: false, style: () => this.styles().land }).addTo(this.map);
    this.vec.lakes = L.geoJSON(lakes, { pane: "lakes", renderer: this.renderers.lakes, interactive: false, style: () => this.styles().lakes }).addTo(this.map);
  }

  async set(id, show) {
    if (show) this.on.add(id); else this.on.delete(id);
    if (id === "counties" || id === "states") {
      if (show) {
        if (!this.vec[id]) {
          const d = await this.get(id); if (!this.on.has(id)) return;
          this.vec[id] = L.geoJSON(d, { pane: id, renderer: this.renderers[id], interactive: false, style: () => this.styles()[id] });
        }
        this.vec[id].addTo(this.map);
      } else if (this.vec[id]) this.vec[id].remove();
    } else if (id === "roads") {
      if (show) {
        if (!this.vec.roads) {
          const d = await this.get("roads"); if (!this.on.has(id)) return;
          this.roadLabels = d.labels;
          this.vec.roadsCase = L.geoJSON(d.lines, { pane: "roads", renderer: this.renderers.roads, interactive: false, style: f => this.roadStyle(f, true) });
          this.vec.roads = L.geoJSON(d.lines, { pane: "roads", renderer: this.renderers.roads, interactive: false, style: f => this.roadStyle(f, false) });
        }
        this.vec.roadsCase.addTo(this.map); this.vec.roads.addTo(this.map);
      } else { this.vec.roadsCase?.remove(); this.vec.roads?.remove(); }
    } else if (show) this.lab[id] = await this.get(id);       // places, peaks, ski: label sets
    this.labels();
  }

  // Interstates from the first zoom, U.S. routes only once zoomed in; widths grow with zoom so the field stays readable when zoomed out.
  roadStyle(f, casing) {
    const z = this.map.getZoom(), i = f.properties.c === 0;
    const w = Math.max(0.7, Math.min(i ? 2.6 : 1.8, (i ? 0.9 : 0.6) + 0.28 * (z - 6)));
    const on = i || z >= 8;
    return casing ? { color: css("--road-casing"), weight: w + 1.8, opacity: on ? 1 : 0 } : { color: i ? css("--road-i") : css("--road-us"), weight: w, opacity: on ? 1 : 0 };
  }

  restyle() {
    const st = this.styles();
    this.vec.land?.setStyle(st.land); this.vec.lakes?.setStyle(st.lakes);
    this.vec.counties?.setStyle(st.counties); this.vec.states?.setStyle(st.states);
    this.vec.roadsCase?.setStyle(f => this.roadStyle(f, true));
    this.vec.roads?.setStyle(f => this.roadStyle(f, false));
  }

  // Greedy declutter: candidates in priority order, each kept only if its box misses every box kept so far.
  labels() {
    const m = this.map, z = m.getZoom(), box = this.labelBox;
    const b = m.getBounds().pad(0.05), kept = [], frag = document.createDocumentFragment();
    const place = (lat, lon, text, cls, prefix = "") => {
      if (!b.contains([lat, lon])) return false;
      const p = m.latLngToContainerPoint([lat, lon]);
      const w = text.length * (cls.includes("city") ? 7.6 : 6.8) + (cls.includes("route") ? 10 : 16), h = 16;
      const x0 = cls.includes("route") ? p.x - w / 2 : p.x - 4, y0 = p.y - h / 2;
      for (const k of kept) if (x0 < k[2] && x0 + w > k[0] && y0 < k[3] && y0 + h > k[1]) return false;
      kept.push([x0, y0, x0 + w, y0 + h]);
      const el = document.createElement("div"); el.className = "lbl " + cls;
      const lp = m.containerPointToLayerPoint(p);
      el.style.left = lp.x + (cls.includes("route") ? 0 : -4) + "px"; el.style.top = lp.y + "px";
      if (!cls.includes("route")) { el.append(document.createElement("i")); }
      el.append(document.createTextNode(text));
      frag.append(el); return true;
    };
    if (this.on.has("places") && this.lab.places) {
      const d = this.lab.places;
      for (const [n, st, lat, lon, pop, mz] of d) if (z >= mz) place(lat, lon, n, "city" + (pop > 50000 ? "" : " small"));
    }
    if (this.on.has("peaks") && this.lab.peaks) {
      const d = this.lab.peaks;
      for (const [n, st, lat, lon, ft, mz] of d) if (z >= mz + 5) place(lat, lon, `${n} ${ft.toLocaleString("en-US")} ft`, "peak");
    }
    if (this.on.has("ski") && this.lab.ski) {
      const d = this.lab.ski;
      if (z >= 8) for (const [n, st, lat, lon] of d) place(lat, lon, n, "ski");
    }
    if (this.on.has("roads") && this.roadLabels)
      for (const [t, c, lon, lat, mz] of this.roadLabels) if (z >= mz) place(lat, lon, t.replace(/^(I|US) /, "").replace(/^I-/, ""), "route " + (c === 0 ? "i" : "us"));
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
