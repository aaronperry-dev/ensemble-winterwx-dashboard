import { loadBundle, extract, series, sample } from "./bundle.js";
import { scales } from "./colors.js";
import { FieldLayer } from "./field-layer.js";
import { Gis, nearestPlace } from "./layers.js";
import { FanChart, tableHtml, fmt, stamp } from "./chart.js";

const $ = id => document.getElementById(id);
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const root = document.documentElement;

const STATS = [["p10", "10th", "Only 1 member in 10 has less than this"], ["p25", "25th", "A quarter of members have less"], ["p50", "Median", "Half the members have more, half less"],
  ["p75", "75th", "A quarter of members have more"], ["p90", "90th", "Only 1 member in 10 has more than this"], ["mean", "Mean", "Average of the members"], ["max", "Max", "The wettest member"]];
const GROUPS = { blend: "Consensus", method: "Methods", reference: "Simple references" };
const LAYERS = [["counties", "Counties", true], ["states", "State lines", true], ["roads", "Highways", true], ["places", "Towns", true], ["peaks", "Peaks", false], ["ski", "Ski areas", false]];
const DEFAULT_PT = [42.3584, -71.0598];

const S = { ens: "eps", method: "blend", prod: "pct", stat: "p50", thr: 1, win: 0, t: 0, pt: DEFAULT_PT, smooth: true, opacity: 0.85, dark: false, layers: new Set(LAYERS.filter(l => l[2]).map(l => l[0])) };
let M, GRID, map, field, gis, chart, scale, bundle, places = [], playing = null, token = 0;
const cache = new Map();

// ---------- state in the URL ----------
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1)), o = {};
  for (const [k, v] of p) o[k] = v;
  return o;
}
function writeHash() {
  const c = map.getCenter(), p = new URLSearchParams({ e: S.ens, m: S.method, p: S.prod, s: S.stat, k: S.thr, w: S.win, t: S.t, pt: S.pt.map(v => v.toFixed(3)).join(","), v: `${c.lat.toFixed(3)},${c.lng.toFixed(3)},${map.getZoom()}`, l: [...S.layers].join(".") });
  history.replaceState(null, "", "#" + p);
}

// ---------- small UI builders ----------
function radio(box, items, value, on, label) {
  box.replaceChildren(...items.map(([v, text, sub, disabled]) => {
    const b = el("button", { type: "button", role: "radio", disabled: !!disabled }, text);
    if (sub) b.append(el("small", {}, sub));
    b.setAttribute("aria-checked", String(v === value)); b.dataset.v = v;
    b.addEventListener("click", () => on(v));
    return b;
  }));
  box.onkeydown = e => {
    const bs = [...box.querySelectorAll("button:not(:disabled)")], i = bs.indexOf(document.activeElement);
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (d && i >= 0) { e.preventDefault(); bs[(i + d + bs.length) % bs.length].focus(); bs[(i + d + bs.length) % bs.length].click(); }
  };
}
const mark = (box, v) => box.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.v === String(v))));

// ---------- data ----------
const key = () => `${S.ens}_${S.method}`;
async function getBundle() {
  const k = key();
  if (!cache.has(k)) {
    const f = M.ensembles[S.ens].files[S.method];
    cache.set(k, loadBundle(`data/${f.file}`).catch(e => { cache.delete(k); throw e; }));
  }
  return cache.get(k);
}

async function refresh(full = true) {
  const my = ++token, loading = $("loading");
  const pending = !cache.has(key());
  if (pending) loading.hidden = false;
  try { bundle = await getBundle(); }
  catch (e) { loading.hidden = false; loading.textContent = "Could not load this dataset. Try another method or reload."; console.error(e); return; }
  if (my !== token) return;
  loading.hidden = true; loading.textContent = "Loading";
  paint(); if (full) updateChart(); updateReadoutAtPoint(); writeHash();
}

function currentSpec() {
  if (S.prod === "pct") return { kind: "pct", w: S.win, s: M.stats.indexOf(S.stat), t: S.t };
  if (S.prod === "prob") return { kind: "prob", w: S.win, k: S.thr, t: S.t };
  return { kind: "slr", t: S.t };
}
function scaleFor() { const sc = scales(S.dark); return S.prod === "prob" ? sc.prob : S.prod === "slr" ? sc.slr : sc.snow; }

function paint() {
  scale = scaleFor();
  S.values = extract(bundle, currentSpec(), GRID);
  field.setData({ values: S.values, grid: GRID, scale, smooth: S.smooth });
  legend(); timeLabels();
}

function describe() {
  const stat = STATS.find(s => s[0] === S.stat);
  const win = S.win ? "24-hour snowfall ending" : "Snowfall since the start of the run to";
  if (S.prod === "pct") return { title: `${stat[1] === "Mean" || stat[1] === "Max" ? stat[1] : stat[1] + (stat[1] === "Median" ? "" : " percentile")} ${S.win ? "24-hour" : "total"} snowfall`, unit: "inches", note: stat[2] };
  if (S.prod === "prob") return { title: `Chance of at least ${M.thresholds[S.thr]} in, ${S.win ? "in 24 hours" : "run total"}`, unit: "% of members", note: "Share of ensemble members reaching the amount" };
  return { title: "Effective snow-to-liquid ratio", unit: ":1", note: "Whole-run snow depth divided by snow-type liquid, median member" };
}

function legend() {
  const d = describe(), sc = scale, box = $("legend");
  const bar = el("div", { className: "bar" }), ticks = el("div", { className: "ticks" });
  sc.colors.forEach((c, i) => {
    bar.append(el("i", { style: `background:${c}` }));
    const e = sc.edges[i];
    ticks.append(el("span", {}, Number.isFinite(e) ? sc.fmt(e) + (i === sc.colors.length - 1 ? "+" : "") : ""));
  });
  box.replaceChildren(el("div", { className: "lt" }, d.title + " ", el("span", {}, `(${d.unit})`)), bar, ticks);
  box.title = d.note;
}

// ---------- time ----------
const E = () => M.ensembles[S.ens];
const validMs = i => Date.parse(E().init) + E().fhrs[i] * 3600e3;
function timeLabels() {
  const ms = validMs(S.t), f = E().fhrs[S.t];
  $("when-utc").textContent = `${stamp(ms)}  ·  F+${String(f).padStart(3, "0")} h`;
  try {
    $("when-local").textContent = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", weekday: "short", hour: "numeric", minute: "2-digit", timeZoneName: "short" }).format(ms) + " in New England";
  } catch { $("when-local").textContent = ""; }
  chart?.setIndex(S.t);
}
function setTime(i) {
  S.t = Math.max(0, Math.min(E().fhrs.length - 1, i));
  if (bundle) { paint(); updateReadoutAtPoint(); writeHash(); }
}
function togglePlay(on = !playing) {
  const b = $("play");
  if (playing) { clearInterval(playing); playing = null; }
  if (on) { playing = setInterval(() => setTime(S.t + 1 >= E().fhrs.length ? 0 : S.t + 1), 650); }
  b.textContent = playing ? "Pause" : "Play"; b.setAttribute("aria-label", playing ? "Pause" : "Play");
}

// ---------- point, chart, readout ----------
let marker;
function setPoint(lat, lon) {
  S.pt = [lat, lon];
  if (!marker) marker = L.circleMarker(S.pt, { radius: 6, weight: 2.5, color: "#fff", fillColor: "#0b0b0b", fillOpacity: 1, pane: "markerPane", interactive: false }).addTo(map);
  else marker.setLatLng(S.pt);
  const np = nearestPlace(places, lat, lon);
  $("where").textContent = np ? `Near ${np.name}, ${np.st}` : "Chosen point";
  $("where-sub").textContent = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}  ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}  ·  click the map to move`;
  if (bundle) { updateChart(); writeHash(); }
}
let lastSeries;
function updateChart() {
  const win = S.win, ser = series(bundle, GRID, S.pt[0], S.pt[1], win);
  const d = { times: E().fhrs.map((_, i) => validMs(i)), fhr: E().fhrs, series: ser, label: win ? "Snow in the 24 h to each time (in)" : "Snow since the run began (in)" };
  lastSeries = d; chart.setData(d); chart.setIndex(S.t);
  if (!$("table-wrap").hidden) $("table-wrap").innerHTML = tableHtml(d);
}
function valueAt(lat, lon) {
  if (!S.smooth) {
    const x = Math.round((lon - GRID.lon0) / GRID.dlon), y = Math.round((lat - GRID.lat0) / GRID.dlat);
    return x < 0 || y < 0 || x >= GRID.nx || y >= GRID.ny ? NaN : S.values[y * GRID.nx + x];
  }
  return sample(S.values, GRID, lat, lon);
}
function showReadout(lat, lon) {
  const box = $("readout"), v = valueAt(lat, lon);
  if (!Number.isFinite(v)) { box.replaceChildren(); return; }
  const unit = S.prod === "prob" ? "%" : S.prod === "slr" ? ":1" : "in";
  const big = el("div", { className: "big" }, S.prod === "slr" ? v.toFixed(1) : fmt(v), el("small", {}, unit));
  const kids = [big];
  if (S.prod !== "slr") {
    const ser = series(bundle, GRID, lat, lon, S.win), t = S.t;
    kids.push(el("div", { className: "trio" }, ...[["10th", "p10"], ["Median", "p50"], ["90th", "p90"]].map(([n, k]) => el("span", {}, n + " ", el("b", {}, fmt(ser[k][t]))))));
    kids.push(el("div", { className: "sub" }, `Snow range, in ${S.win ? "24 h" : "total"}`));
  }
  box.replaceChildren(...kids);
}
const updateReadoutAtPoint = () => { if (S.hover) showReadout(...S.hover); else showReadout(...S.pt); };

// ---------- controls ----------
function buildControls() {
  radio($("ens"), Object.keys(M.ensembles).map(k => [k, M.ensembles[k].short, k === "all" ? "pooled" : `${M.ensembles[k].members} mem.`]), S.ens, v => { S.ens = v; onEnsChange(); });
  buildMethods(); buildProd();
  radio($("win"), [[0, "Run total", "since start"], [1, "24 hours", "rolling"]], S.win, v => { S.win = +v; mark($("win"), v); refresh(); });
  const box = $("layers");
  box.replaceChildren(...LAYERS.map(([id, text]) => {
    const i = el("input", { type: "checkbox", checked: S.layers.has(id) }); i.addEventListener("change", () => { i.checked ? S.layers.add(id) : S.layers.delete(id); gis.set(id, i.checked); writeHash(); });
    return el("label", { className: "check" }, i, el("span", {}, text));
  }));
  $("opacity").value = Math.round(S.opacity * 100);
  $("opacity").addEventListener("input", e => { S.opacity = e.target.value / 100; field.setOpacity(S.opacity); });
  $("smooth").checked = S.smooth;
  $("smooth").addEventListener("change", e => { S.smooth = e.target.checked; paint(); });
  $("prev").onclick = () => { togglePlay(false); setTime(S.t - 1); };
  $("next").onclick = () => { togglePlay(false); setTime(S.t + 1); };
  $("play").onclick = () => togglePlay();
  $("table-btn").onclick = () => {
    const on = $("table-wrap").hidden; $("table-wrap").hidden = !on; $("table-btn").setAttribute("aria-pressed", String(on));
    $("table-btn").textContent = on ? "Chart" : "Table"; if (on) $("table-wrap").innerHTML = tableHtml(lastSeries);
  };
  $("theme").onclick = () => setTheme(!S.dark);
  $("rail-toggle").onclick = () => { const o = $("rail").classList.toggle("open"); $("rail-toggle").setAttribute("aria-expanded", String(o)); setTimeout(() => map.invalidateSize(), 50); };
  credits();
}
function buildMethods() {
  const sel = $("method"), e = E(); sel.replaceChildren();
  for (const g of Object.keys(GROUPS)) {
    const ms = e.methods.filter(m => M.methods[m].group === g); if (!ms.length) continue;
    const og = el("optgroup", { label: GROUPS[g] });
    for (const m of ms) og.append(el("option", { value: m, selected: m === S.method }, M.methods[m].label));
    sel.append(og);
  }
  sel.onchange = () => { S.method = sel.value; methodNote(); refresh(); };
  methodNote();
}
function methodNote() {
  const m = M.methods[S.method]; $("method-desc").textContent = m.desc + (m.ref ? ` (${m.ref})` : "");
}
function buildProd() {
  radio($("prod"), [["pct", "Amount", "percentile"], ["prob", "Chance", "of at least"], ["slr", "Ratio", "snow : liquid"]], S.prod, v => { S.prod = v; buildProd(); $("win-wrap").hidden = v === "slr"; refresh(); });
  const chips = $("prod-opts");
  if (S.prod === "pct") radio(chips, STATS.map(([k, t]) => [k, t]), S.stat, v => { S.stat = v; mark(chips, v); refresh(false); });
  else if (S.prod === "prob") radio(chips, M.thresholds.map((t, i) => [i, `${t} in`]), S.thr, v => { S.thr = +v; mark(chips, v); refresh(false); });
  else chips.replaceChildren();
  $("win-wrap").hidden = S.prod === "slr";
}
function onEnsChange() {
  mark($("ens"), S.ens);
  if (!E().methods.includes(S.method)) S.method = "blend";
  S.t = Math.min(S.t, E().fhrs.length - 1);
  buildMethods(); ensNote(); runline(); refresh();
}
function ensNote() {
  const e = E(), n = $("ens-note");
  n.replaceChildren(`${e.center}, ${e.native}. Precipitation type: ${e.ptype}.`);
  for (const t of e.notes.filter(t => !t.startsWith("Synthetic"))) n.append(el("br"), t);
}
function runline() {
  const e = E(), d = new Date(e.init), gen = new Date(M.generated);
  const f = x => `${DAYS(x)} ${x.getUTCDate()} ${x.toLocaleString("en-US", { month: "short", timeZone: "UTC" })}`;
  $("runline").textContent = `${e.short} run of ${String(d.getUTCHours()).padStart(2, "0")} UTC ${f(d)}, to F+${e.fhrs.at(-1)} h. Page built ${String(gen.getUTCHours()).padStart(2, "0")}:${String(gen.getUTCMinutes()).padStart(2, "0")} UTC ${f(gen)}.`;
}
const DAYS = d => ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][d.getUTCDay()];
function credits() {
  const p = $("credits"); p.replaceChildren();
  for (const k of ["eps", "gefs", "geps"]) { const e = M.ensembles[k]; p.append(el("a", { href: e.url, target: "_blank", rel: "noopener" }, e.short), ` ${e.center}, ${e.license}. `); }
  p.append("Basemap: Natural Earth (public domain). Places and peaks: GeoNames (CC BY 4.0). Ratios computed on a 0.25° grid, so terrain is smoothed.");
}

function setTheme(dark) {
  S.dark = dark; root.dataset.theme = dark ? "dark" : "light";
  $("theme").textContent = dark ? "Light" : "Dark";
  try { localStorage.setItem("ensnow-theme", dark ? "dark" : "light"); } catch {}
  if (gis) { gis.restyle(); }
  if (bundle) { paint(); chart.draw(); }
}

// ---------- start ----------
async function start() {
  const res = await fetch("data/manifest.json"); M = await res.json(); GRID = M.grid;
  if (M.synthetic) $("demo-banner").hidden = false;
  const h = readHash();
  if (h.e && M.ensembles[h.e]) S.ens = h.e;
  if (h.m && M.methods[h.m] && E().methods.includes(h.m)) S.method = h.m;
  if (["pct", "prob", "slr"].includes(h.p)) S.prod = h.p;
  if (h.s && M.stats.includes(h.s)) S.stat = h.s;
  if (h.k !== undefined && +h.k >= 0 && +h.k < M.thresholds.length) S.thr = +h.k;
  if (h.w === "1") S.win = 1;
  if (h.t !== undefined && +h.t >= 0 && +h.t < E().fhrs.length) S.t = +h.t;
  else S.t = Math.max(0, E().fhrs.indexOf(E().fhrs.find(f => f >= 48) ?? E().fhrs.at(-1)));
  if (h.pt) { const q = h.pt.split(",").map(Number); if (q.length === 2 && q.every(Number.isFinite)) S.pt = q; }
  if (h.l !== undefined) S.layers = new Set(h.l.split(".").filter(Boolean));
  let dark = matchMedia("(prefers-color-scheme: dark)").matches;
  try { const t = localStorage.getItem("ensnow-theme"); if (t) dark = t === "dark"; } catch {}
  S.dark = dark; root.dataset.theme = dark ? "dark" : "light"; $("theme").textContent = dark ? "Light" : "Dark";

  const b = M.bounds, grid = [[GRID.lat0, GRID.lon0], [GRID.lat0 + GRID.dlat * (GRID.ny - 1), GRID.lon0 + GRID.dlon * (GRID.nx - 1)]];
  const gb = L.latLngBounds(grid), gbw = gb.pad(0.4);      // the basemap extends 4 degrees past the data
  map = L.map("map", { zoomControl: true, minZoom: 5, maxZoom: 12, zoomSnap: 0.5, zoomDelta: 0.5, zoomAnimation: false, fadeAnimation: false, markerZoomAnimation: false, maxBounds: gbw, maxBoundsViscosity: 1, preferCanvas: true, keyboard: true });
  const v = h.v && h.v.split(",").map(Number);
  if (v && v.length === 3 && v.every(Number.isFinite)) map.setView([v[0], v[1]], v[2]);;
  const fit = () => { map.invalidateSize(); map.setMinZoom(Math.max(5, map.getBoundsZoom(gb, false), map.getBoundsZoom(gbw, true))); };
  fit(); addEventListener("resize", fit);
  if (!(v && v.length === 3 && v.every(Number.isFinite))) map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: [16, 16] });
  field = new FieldLayer({ opacity: S.opacity }); field.addTo(map);
  gis = new Gis(map, "data/gis");
  await gis.baseLayers().catch(e => console.error(e));
  chart = new FanChart($("chart"), $("chart-tip"), i => { togglePlay(false); setTime(i); });

  buildControls(); ensNote(); runline();
  gis.get("places").then(p => { places = p; setPoint(...S.pt); }).catch(() => {});
  for (const [id] of LAYERS) if (S.layers.has(id)) gis.set(id, true).catch(console.error);
  setPoint(...S.pt);
  map.on("click", e => { setPoint(e.latlng.lat, e.latlng.lng); });
  map.on("mousemove", e => { S.hover = [e.latlng.lat, e.latlng.lng]; if (bundle) showReadout(...S.hover); });
  map.on("mouseout", () => { S.hover = null; if (bundle) showReadout(...S.pt); });
  map.on("moveend", () => writeHash());
  addEventListener("keydown", e => {
    if (e.target.closest("input, select, textarea")) return;
    if (e.key === " " && e.target === document.body) { e.preventDefault(); togglePlay(); }
  });
  await refresh();
  window.__ens = { S, map, get bundle() { return bundle; } };       // handle for tests and debugging
}
start().catch(e => { document.body.append(el("pre", {}, "The viewer could not start: " + e.message)); console.error(e); });
