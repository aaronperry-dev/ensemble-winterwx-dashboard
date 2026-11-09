import { loadBundle, extract, series, sample } from "./bundle.js";
import { scales } from "./colors.js";
import { FieldLayer } from "./field-layer.js";
import { Gis, nearestPlace } from "./layers.js";
import { FanChart, tableHtml, fmt, stamp } from "./chart.js";

const $ = id => document.getElementById(id);
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const root = document.documentElement;

const PCT = [
  ["min", "Min", "Minimum", "The smallest of the members at each point"],
  ["p5", "5th", "5th percentile", "Only 1 member in 20 has less than this"],
  ["p10", "10th", "10th percentile", "Only 1 member in 10 has less than this"],
  ["p25", "25th", "25th percentile", "A quarter of the members have less"],
  ["p50", "Median", "Median", "Half the members have more, half less"],
  ["mean", "Mean", "Mean", "The average of the members. Smooth, but it understates heavy amounts where members disagree about where the snow falls"],
  ["p75", "75th", "75th percentile", "A quarter of the members have more"],
  ["p90", "90th", "90th percentile", "Only 1 member in 10 has more than this"],
  ["p95", "95th", "95th percentile", "Only 1 member in 20 has more than this"],
  ["max", "Max", "Maximum", "The snowiest member at each point. A bound, not a forecast: different members set it at different places"]];
const CONS = [
  ["pmm", "PMM", "Probability-matched mean (PMM)", "Keeps the pattern of the ensemble mean but refits its values to the pooled amounts of all members, so the heavy amounts a plain mean smooths away are back."],
  ["lpmm", "LPMM", "Local probability-matched mean (LPMM)", "The same refit within local patches, drawing amounts from the neighbourhood (about 375 km) around each one, so snow from one storm is not given to another area."],
  ["sam", "SAM", "Spatially aligned mean (SAM)", "Each member is first moved toward the PMM so the storm sits in a common place (Lee et al. 2024), then the members are averaged. A cohesive swath whose amounts are an average, not extremes."],
  ["samlpm", "SAM–LPM", "Spatially aligned mean with local probability matching (SAM–LPM)", "SAM, then local probability matching: the aligned position of SAM with the realistic peak amounts of LPMM. Experimental."]];
const CONS_CAUTION = "Consensus maps show one plausible picture, not the likeliest outcome at every point, and they do not conserve totals. Use the percentiles and chances for point decisions.";
const GROUPS = { blend: "Consensus", method: "Methods", reference: "Simple references" };
const LAYERS = [["counties", "Counties", true], ["states", "State lines", true], ["roads", "Highways", true], ["places", "Towns", true], ["peaks", "Peaks", false], ["ski", "Ski areas", false]];
const DEFAULT_PT = [42.3584, -71.0598];
const DAYS = d => ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][d.getUTCDay()];
const winText = w => (w ? `${w}-hour` : "storm-total");

const S = { ens: "eps", method: "blend", prod: "pct", stat: "p50", cons: "pmm", thr: 1, win: 0, t: 0, pt: DEFAULT_PT, smooth: true, opacity: 0.85, dark: false, layers: new Set(LAYERS.filter(l => l[2]).map(l => l[0])) };
let M, GRID, map, field, gis, chart, scale, bundle, places = [], playing = null, token = 0;
const cache = new Map();

// ---------- state in the URL ----------
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1)), o = {};
  for (const [k, v] of p) o[k] = v;
  return o;
}
function writeHash() {
  const c = map.getCenter(), p = new URLSearchParams({ e: S.ens, m: S.method, p: S.prod, s: S.stat, c: S.cons, k: S.thr, w: S.win, t: S.t, pt: S.pt.map(v => v.toFixed(3)).join(","), v: `${c.lat.toFixed(3)},${c.lng.toFixed(3)},${map.getZoom()}`, l: [...S.layers].join(".") });
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

// The accumulation period that applies: the effective ratio is a run total whatever period is chosen.
const activeWin = () => (S.prod === "slr" ? 0 : S.win);
const winIndex = () => M.windows.indexOf(activeWin());
function currentSpec() {
  if (S.prod === "pct") return { kind: "pct", w: winIndex(), s: M.stats.indexOf(S.stat), t: S.t };
  if (S.prod === "cons") return { kind: "pct", w: winIndex(), s: M.stats.indexOf(S.cons), t: S.t };
  if (S.prod === "prob") return { kind: "prob", w: winIndex(), k: S.thr, t: S.t };
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
  const w = winText(activeWin());
  if (S.prod === "pct") { const st = PCT.find(x => x[0] === S.stat); return { title: `${st[2]} ${w} snowfall`, unit: "inches", note: st[3] }; }
  if (S.prod === "cons") { const st = CONS.find(x => x[0] === S.cons); return { title: `${st[1]} ${w} snowfall`, unit: "inches", note: st[3] }; }
  if (S.prod === "prob") return { title: `Chance of at least ${M.thresholds[S.thr]} in, ${w}`, unit: "% of members", note: "Share of ensemble members reaching the amount" };
  return { title: "Effective snow-to-liquid ratio", unit: ":1", note: "Whole-run snow depth divided by snow-type liquid, mean member" };
}

const narrowQuery = matchMedia("(max-width: 860px)");
function legend() {
  const d = describe(), sc = scale, box = $("legend");
  // Class ranges ("12–18") do not fit under a swatch on a phone, so there the numbers sit on the boundaries between swatches, as on a continuous legend.
  const edgeLabels = !!sc.labels && narrowQuery.matches;
  const bar = el("div", { className: "bar" }), ticks = el("div", { className: "ticks" + (sc.labels && !edgeLabels ? " cls" : "") });
  sc.colors.forEach((c, i) => {
    bar.append(el("i", { style: `background:${c}` }));
    const e = sc.edges[i];
    ticks.append(el("span", {}, edgeLabels ? (i === 0 ? "" : sc.fmt(e)) : sc.labels ? sc.labels[i] : Number.isFinite(e) ? sc.fmt(e) + (i === sc.colors.length - 1 ? "+" : "") : ""));
  });
  box.replaceChildren(el("div", { className: "lt" }, d.title + " ", el("span", {}, `(${d.unit})`)), bar, ticks);
  box.classList.toggle("lab", !!sc.labels && !edgeLabels); box.title = d.note;
}
narrowQuery.addEventListener("change", () => { if (scale) legend(); });
// On a phone the title and the Controls button share one row, so the appearance switch moves into the controls sheet.
function placeTheme() {
  const b = $("theme");
  if (narrowQuery.matches) $("rail").querySelector(".sheet-head").insertBefore(b, $("rail-close"));
  else document.querySelector(".top").insertBefore(b, $("rail-toggle"));
}
narrowQuery.addEventListener("change", placeTheme); addEventListener("DOMContentLoaded", placeTheme); placeTheme();

// ---------- time ----------
// S.t indexes the ensemble's full list of output hours.  A period is only complete at some of them (a 24-hour total first exists at F+24 h), so the
// slider steps through the hours listed for the chosen period in the manifest.
const E = () => M.ensembles[S.ens];
const validMs = i => Date.parse(E().init) + E().fhrs[i] * 3600e3;
const validIdx = () => (E().window_times[String(activeWin())] || []).map(f => E().fhrs.indexOf(f)).filter(i => i >= 0);
const snapTime = i => { const v = validIdx(); return !v.length || v.includes(i) ? i : (v.find(x => x >= i) ?? v.at(-1)); };
const localStamp = ms => {
  try { return new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", weekday: "short", hour: "numeric", minute: "2-digit", timeZoneName: "short" }).format(ms); } catch { return ""; }
};
function timeLabels() {
  const ms = validMs(S.t), f = E().fhrs[S.t], w = activeWin(), v = validIdx(), pos = Math.max(0, v.indexOf(S.t));
  $("when-utc").textContent = `Valid ${stamp(ms)} · F+${String(f).padStart(3, "0")} h`;
  // local time first: on a phone this line is cut to one row, and the clock time is what is wanted from it
  const lt = localStamp(ms), what = w ? `${w}-hour snow from ${stamp(ms - w * 3600e3)} to ${stamp(ms)}` : `Snow since the run began, ${stamp(Date.parse(E().init))}`;
  $("when-local").textContent = (lt ? `${lt} in New England · ` : "") + what; $("when-local").title = $("when-local").textContent;
  const t = $("time"); t.value = pos; t.setAttribute("aria-valuetext", `${stamp(ms)}, forecast hour ${f}`);
  chart?.setIndex(pos);
}
function buildTime() {
  const v = validIdx(), t = $("time"), days = $("days");
  t.max = Math.max(0, v.length - 1); t.disabled = v.length < 2; t.value = Math.max(0, v.indexOf(S.t));
  days.replaceChildren();
  v.forEach((i, k) => {
    const d = new Date(validMs(i));
    if (d.getUTCHours() !== 0) return;
    const f = v.length > 1 ? k / (v.length - 1) : 0;
    const sp = el("span", { style: `left: calc(8px + (100% - 16px) * ${f})` }, `${DAYS(d)} ${d.getUTCDate()}`);
    Object.assign(sp.dataset, { f: String(f), long: `${DAYS(d)} ${d.getUTCDate()}`, short: DAYS(d) });
    days.append(sp);
  });
  layoutDays();
  const w = activeWin(), hrs = v.map(i => E().fhrs[i]);
  $("win-note").textContent = !v.length ? "" : w
    ? `${w}-hour totals are complete from F+${hrs[0]} h to F+${hrs.at(-1)} h${hrs.at(-1) < E().fhrs.at(-1) ? `; after that this system publishes ${w < 6 ? "only 6-hourly" : "too coarse"} fields` : ""}.`
    : "Snow from the start of the run to each time.";
}
// Weekday and date under the slider where they fit, the weekday alone where they do not, and no label that would touch its neighbour.
function layoutDays() {
  const days = $("days"), spans = [...days.children], w = days.clientWidth - 16;
  if (!spans.length || w <= 0) return;
  for (const kind of ["long", "short"]) {
    let last = -1e9, ok = true;
    for (const sp of spans) {
      sp.textContent = sp.dataset[kind]; sp.hidden = false;
      const x = 8 + w * +sp.dataset.f, half = sp.offsetWidth / 2;
      if (x - half < last + 6) { ok = false; sp.hidden = true; continue; }
      last = x + half;
    }
    if (ok) return;
  }
}
new ResizeObserver(layoutDays).observe($("days"));
function setTime(i) {
  S.t = snapTime(Math.max(0, Math.min(E().fhrs.length - 1, i)));
  if (bundle) { paint(); updateReadoutAtPoint(); writeHash(); }
}
function stepTime(d) { const v = validIdx(), p = v.indexOf(S.t); setTime(v[Math.max(0, Math.min(v.length - 1, (p < 0 ? 0 : p) + d))]); }
function togglePlay(on = !playing) {
  const b = $("play");
  if (playing) { clearInterval(playing); playing = null; }
  if (on) { playing = setInterval(() => { const v = validIdx(), p = v.indexOf(S.t); setTime(v[p + 1 >= v.length ? 0 : p + 1]); }, 600); }
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
  $("where-sub").textContent = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}  ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}  ·  ${matchMedia("(pointer: coarse)").matches ? "tap" : "click"} the map to move`;
  if (bundle) { updateChart(); writeHash(); }
}
let lastSeries;
function updateChart() {
  const v = validIdx(), w = activeWin(), ser = series(bundle, GRID, S.pt[0], S.pt[1], winIndex()), sub = {};
  for (const k of Object.keys(ser)) sub[k] = Float32Array.from(v, i => ser[k][i]);
  const d = { times: v.map(validMs), fhr: v.map(i => E().fhrs[i]), idx: v, series: sub, label: w ? `${w}-hour snow ending at each time (in)` : "Snow since the run began (in)" };
  lastSeries = d; chart.setData(d); chart.setIndex(Math.max(0, v.indexOf(S.t)));
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
    const ser = series(bundle, GRID, lat, lon, winIndex()), t = S.t;
    kids.push(el("div", { className: "trio" }, ...[["10th", "p10"], ["Median", "p50"], ["90th", "p90"]].map(([n, k]) => el("span", {}, n + " ", el("b", {}, fmt(ser[k][t]))))));
    kids.push(el("div", { className: "sub" }, `Snow range, ${winText(activeWin())}`));
  }
  box.replaceChildren(...kids);
}
const updateReadoutAtPoint = () => { if (S.hover) showReadout(...S.hover); else showReadout(...S.pt); };

// ---------- controls ----------
function buildControls() {
  radio($("ens"), Object.keys(M.ensembles).map(k => [k, M.ensembles[k].short, k === "all" ? "pooled" : `${M.ensembles[k].members} mem.`]), S.ens, v => { S.ens = v; onEnsChange(); });
  buildMethods(); buildProd();
  buildWindows();
  $("time").addEventListener("input", e => { togglePlay(false); const v = validIdx(); setTime(v[+e.target.value]); });
  const box = $("layers");
  box.replaceChildren(...LAYERS.map(([id, text]) => {
    const i = el("input", { type: "checkbox", checked: S.layers.has(id) }); i.addEventListener("change", () => { i.checked ? S.layers.add(id) : S.layers.delete(id); gis.set(id, i.checked); writeHash(); });
    return el("label", { className: "check" }, i, el("span", {}, text));
  }));
  $("opacity").value = Math.round(S.opacity * 100);
  $("opacity").addEventListener("input", e => { S.opacity = e.target.value / 100; field.setOpacity(S.opacity); });
  $("smooth").checked = S.smooth;
  $("smooth").addEventListener("change", e => { S.smooth = e.target.checked; paint(); });
  $("prev").onclick = () => { togglePlay(false); stepTime(-1); };
  $("next").onclick = () => { togglePlay(false); stepTime(1); };
  $("play").onclick = () => togglePlay();
  $("table-btn").onclick = () => {
    const on = $("table-wrap").hidden; $("table-wrap").hidden = !on; $("table-btn").setAttribute("aria-pressed", String(on));
    $("table-btn").textContent = on ? "Chart" : "Table"; if (on) $("table-wrap").innerHTML = tableHtml(lastSeries);
  };
  $("theme").onclick = () => setTheme(!S.dark);
  const sheet = on => { $("rail").classList.toggle("open", on); $("scrim").hidden = !on; $("rail-toggle").setAttribute("aria-expanded", String(on)); if (on) $("rail").scrollTop = 0; setTimeout(() => map.invalidateSize(), 50); };
  $("rail-toggle").onclick = () => sheet(!$("rail").classList.contains("open"));
  $("rail-close").onclick = $("scrim").onclick = () => { sheet(false); $("rail-toggle").focus(); };
  addEventListener("keydown", e => { if (e.key === "Escape" && $("rail").classList.contains("open")) { sheet(false); $("rail-toggle").focus(); } });
  narrowQuery.addEventListener("change", () => { if (!narrowQuery.matches) sheet(false); });
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
function buildWindows() {
  radio($("win"), [...M.windows.filter(w => w > 0), 0].map(w => [w, w ? `${w} h` : "Total", null, !(E().window_times[String(w)] || []).length]), S.win,
    v => { S.win = +v; mark($("win"), v); S.t = snapTime(S.t); buildTime(); refresh(); });
}
function buildProd() {
  radio($("prod"), [["pct", "Amount"], ["cons", "Consensus"], ["prob", "Chance"], ["slr", "Ratio"]], S.prod,
    v => { S.prod = v; S.t = snapTime(S.t); buildProd(); buildTime(); refresh(); });
  const chips = $("prod-opts"), note = $("prod-note");
  if (S.prod === "pct") {
    radio(chips, PCT.map(([k, t]) => [k, t]), S.stat, v => { S.stat = v; mark(chips, v); noteFor(); refresh(false); });
  } else if (S.prod === "cons") {
    radio(chips, CONS.map(([k, t]) => [k, t]), S.cons, v => { S.cons = v; mark(chips, v); noteFor(); refresh(false); });
  } else if (S.prod === "prob") {
    radio(chips, M.thresholds.map((t, i) => [i, `${t} in`]), S.thr, v => { S.thr = +v; mark(chips, v); noteFor(); refresh(false); });
  } else chips.replaceChildren();
  function noteFor() {
    if (S.prod === "pct") note.textContent = PCT.find(x => x[0] === S.stat)[3] + ".";
    else if (S.prod === "cons") note.replaceChildren(CONS.find(x => x[0] === S.cons)[3] + " ", el("em", {}, CONS_CAUTION));
    else if (S.prod === "prob") note.textContent = `Share of the members with at least ${M.thresholds[S.thr]} in of snow in the period.`;
    else note.textContent = "Run-total snow depth divided by snow-type liquid, mean of the members. Needs at least 0.02 in of snow-liquid.";
  }
  noteFor();
  $("win-wrap").hidden = S.prod === "slr";
}
function onEnsChange() {
  mark($("ens"), S.ens);
  if (!E().methods.includes(S.method)) S.method = "blend";
  S.t = snapTime(Math.min(S.t, E().fhrs.length - 1));
  buildMethods(); buildWindows(); buildTime(); ensNote(); runline(); refresh();
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
  if (["pct", "cons", "prob", "slr"].includes(h.p)) S.prod = h.p;
  if (h.s && PCT.some(x => x[0] === h.s)) S.stat = h.s;
  if (h.c && CONS.some(x => x[0] === h.c)) S.cons = h.c;
  if (h.k !== undefined && +h.k >= 0 && +h.k < M.thresholds.length) S.thr = +h.k;
  if (h.w !== undefined && M.windows.includes(+h.w)) S.win = +h.w;
  if (h.t !== undefined && +h.t >= 0 && +h.t < E().fhrs.length) S.t = +h.t;
  else S.t = Math.max(0, E().fhrs.indexOf(E().fhrs.find(f => f >= 48) ?? E().fhrs.at(-1)));
  S.t = snapTime(S.t);
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
  chart = new FanChart($("chart"), $("chart-tip"), i => { togglePlay(false); setTime(validIdx()[i]); });

  buildControls(); buildTime(); ensNote(); runline();
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
