// Fan chart of the ensemble at one point: P5-P95, P10-P90 and P25-P75 bands, median, mean and maximum.
// It is also the time control: click or drag to pick a time, arrow keys step through times.

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

export const fmt = v => (!Number.isFinite(v) ? "–" : v < 10 ? v.toFixed(1) : v.toFixed(0));
export const stamp = ms => { const d = new Date(ms); return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${String(d.getUTCHours()).padStart(2, "0")} UTC`; };

function niceMax(v) {
  if (!(v > 0)) return 1;
  const steps = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 60, 80, 100, 150];
  for (const s of steps) if (v <= s) return s;
  return Math.ceil(v / 50) * 50;
}

export class FanChart {
  constructor(canvas, tip, onPick) {
    this.cv = canvas; this.tip = tip; this.onPick = onPick; this.idx = 0; this.data = null; this.hover = -1;
    new ResizeObserver(() => this.draw()).observe(canvas);
    const pick = e => { const i = this.indexAt(e); if (i >= 0) this.onPick(i); };
    canvas.addEventListener("pointerdown", e => { canvas.setPointerCapture(e.pointerId); this.drag = true; pick(e); });
    canvas.addEventListener("pointermove", e => { this.hover = this.indexAt(e); if (this.drag) pick(e); this.draw(); this.showTip(e); });
    canvas.addEventListener("pointerup", () => { this.drag = false; });
    canvas.addEventListener("pointerleave", () => { this.hover = -1; this.tip.hidden = true; this.draw(); });
    canvas.addEventListener("keydown", e => {
      const d = { ArrowLeft: -1, ArrowRight: 1, Home: -1e9, End: 1e9 }[e.key];
      if (d) { e.preventDefault(); this.onPick(Math.max(0, Math.min(this.data.times.length - 1, this.idx + d))); }
    });
  }

  // data: { times: [ms], fhr: [h], series: {min,p5,p10,p25,p50,mean,p75,p90,p95,max,...: Float32Array}, label }
  setData(d) { this.data = d; this.draw(); }
  setIndex(i) { this.idx = i; this.draw(); }

  geom() {
    const r = this.cv.getBoundingClientRect(), narrow = r.width < 520;
    return { w: r.width, h: r.height, l: narrow ? 30 : 38, r: narrow ? 10 : 12, t: 22, b: 24, narrow };
  }
  xOf(i, g) { const n = this.data.times.length; return g.l + (n < 2 ? 0 : (i / (n - 1)) * (g.w - g.l - g.r)); }
  indexAt(e) {
    if (!this.data) return -1;
    const g = this.geom(), r = this.cv.getBoundingClientRect(), n = this.data.times.length;
    const f = (e.clientX - r.left - g.l) / (g.w - g.l - g.r);
    return Math.max(0, Math.min(n - 1, Math.round(f * (n - 1))));
  }

  showTip(e) {
    const d = this.data, i = this.hover;
    if (!d || i < 0) { this.tip.hidden = true; return; }
    const s = d.series, rows = [["Max", s.max], ["95th", s.p95], ["90th", s.p90], ["Median", s.p50], ["Mean", s.mean], ["10th", s.p10], ["5th", s.p5], ["Min", s.min]];
    this.tip.innerHTML = `<div class="th">${stamp(d.times[i])}</div>` + rows.map(([n, a]) => `<div class="row"><span>${n}</span><b>${fmt(a[i])} in</b></div>`).join("");
    this.tip.hidden = false;
    const g = this.geom(), x = this.xOf(i, g), tw = this.tip.offsetWidth;
    this.tip.style.left = Math.max(4, Math.min(g.w - tw - 4, x + 12 > g.w - tw ? x - tw - 12 : x + 12)) + "px";
    this.tip.style.top = "6px";
  }

  draw() {
    const cv = this.cv, d = this.data, g = this.geom(), dpr = window.devicePixelRatio || 1;
    if (!g.w) return;
    cv.width = Math.round(g.w * dpr); cv.height = Math.round(g.h * dpr);
    const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, g.w, g.h);
    const fs = g.narrow ? 11 : 12;
    c.font = `${fs}px "Atkinson Hyperlegible Next", system-ui, sans-serif`;
    if (!d) return;
    const s = d.series, n = d.times.length;
    // the key above the plot, wrapped onto as many rows as the width needs; the plot starts below it
    const accent0 = css("--accent-ink"), ink20 = css("--ink-2"), ink30 = css("--ink-3");
    const sw = g.narrow ? 14 : 16, gap = g.narrow ? 10 : 14;
    const lineKey = (col, w, dash) => (x0, y0) => { c.strokeStyle = col; c.lineWidth = w; c.setLineDash(dash || []); c.beginPath(); c.moveTo(x0, y0); c.lineTo(x0 + sw, y0); c.stroke(); c.setLineDash([]); };
    const boxKey = v => (x0, y0) => { c.fillStyle = css(v); c.fillRect(x0, y0 - 5, sw, 10); };
    const title = g.narrow ? d.label.replace(/^(\d+)-hour snow ending at each time/, "$1-h snow").replace(/^Snow since the run began/, "Total snow") : d.label;
    const keys = [[null, title], [lineKey(accent0, 2.25), "Median"], [lineKey(ink20, 1.5, [5, 3]), "Mean"], [lineKey(ink30, 1.25, [2, 3]), "Max"],
                  [boxKey("--band-b"), "25–75%"], [boxKey("--band-a"), "10–90%"], [boxKey("--band-c"), "5–95%"]];
    const rowH = g.narrow ? 15 : 0, place = [];
    let lx0 = g.l, row = 0;
    for (const [draw, label] of keys) {
      const wd = (draw ? sw + 5 : 0) + c.measureText(label).width + (draw ? gap : 12);
      if (rowH && lx0 > g.l && lx0 + wd - gap > g.w - g.r) { row++; lx0 = g.l; }
      place.push([draw, label, lx0, 9 + row * rowH]); lx0 += wd;
    }
    g.t = 22 + row * rowH;
    // scale to the 95th percentile; the maximum member can be several times larger and would flatten everything else, so it is clipped and labelled
    const maxAll = Math.max(...s.max.filter(Number.isFinite), 0), top = niceMax(Math.max(...s.p95.filter(Number.isFinite), 0.5) * 1.15);
    const y = v => g.t + (1 - Math.min(v, top) / top) * (g.h - g.t - g.b);
    const hair = css("--hair"), ink2 = css("--ink-2"), ink3 = css("--ink-3"), accent = css("--accent-ink");
    // y grid and labels
    const ticks = top <= 4 ? [0, 1, 2, 3, 4].filter(v => v <= top) : [0, top / 4, top / 2, (3 * top) / 4, top].map(v => Math.round(v * 10) / 10);
    c.textAlign = "right"; c.textBaseline = "middle"; c.fillStyle = ink2; c.strokeStyle = hair; c.lineWidth = 1;
    for (const v of ticks) { const yy = Math.round(y(v)) + 0.5; c.beginPath(); c.moveTo(g.l, yy); c.lineTo(g.w - g.r, yy); c.stroke(); c.fillText(fmt(v), g.l - 6, yy); }
    // x: a tick at each 00 UTC, weekday and date (the weekday alone when the full labels would touch), never two labels on top of each other
    c.textAlign = "center"; c.textBaseline = "top";
    const mids = [];
    for (let i = 0; i < n; i++) if (new Date(d.times[i]).getUTCHours() === 0) mids.push(i);
    const full = i => { const dt = new Date(d.times[i]); return `${DAYS[dt.getUTCDay()]} ${dt.getUTCDate()}`; };
    const gaps = mids.slice(1).map((m, k) => this.xOf(m, g) - this.xOf(mids[k], g));
    const useShort = gaps.length && Math.min(...gaps) < Math.max(...mids.map(i => c.measureText(full(i)).width)) + 10;
    let lastR = -1e9;
    for (const i of mids) {
      const dt = new Date(d.times[i]), text = useShort ? DAYS[dt.getUTCDay()] : full(i), half = c.measureText(text).width / 2;
      const x = Math.round(this.xOf(i, g)) + 0.5;
      c.strokeStyle = hair; c.beginPath(); c.moveTo(x, g.t); c.lineTo(x, g.h - g.b); c.stroke();
      const lx = Math.min(Math.max(x, g.l + half), g.w - 4 - half);
      if (lx - half < lastR + 6) continue;
      lastR = lx + half; c.fillStyle = ink2; c.fillText(text, lx, g.h - g.b + 6);
    }
    const band = (lo, hi, col) => {
      c.beginPath();
      for (let i = 0; i < n; i++) c[i ? "lineTo" : "moveTo"](this.xOf(i, g), y(hi[i]));
      for (let i = n - 1; i >= 0; i--) c.lineTo(this.xOf(i, g), y(lo[i]));
      c.closePath(); c.fillStyle = col; c.fill();
    };
    band(s.p5, s.p95, css("--band-c")); band(s.p10, s.p90, css("--band-a")); band(s.p25, s.p75, css("--band-b"));
    const line = (a, col, w, dash) => {
      c.beginPath(); c.setLineDash(dash || []); c.strokeStyle = col; c.lineWidth = w; c.lineJoin = "round";
      for (let i = 0; i < n; i++) c[i ? "lineTo" : "moveTo"](this.xOf(i, g), y(a[i]));
      c.stroke(); c.setLineDash([]);
    };
    c.save(); c.beginPath(); c.rect(g.l, g.t - 2, g.w - g.l - g.r, g.h - g.t - g.b + 2); c.clip();
    line(s.max, ink3, 1.25, [2, 3]); c.restore();
    if (maxAll > top) { c.textAlign = "right"; c.textBaseline = "bottom"; c.fillStyle = ink3; c.fillText(`Max reaches ${fmt(maxAll)} in, off the chart`, g.w - g.r - 4, g.t + 14); }
    line(s.mean, ink2, 1.5, [5, 3]); line(s.p50, accent, 2.25);
    // current time
    const x = this.xOf(this.idx, g);
    c.strokeStyle = css("--ink"); c.lineWidth = 1.5; c.beginPath(); c.moveTo(x, g.t - 4); c.lineTo(x, g.h - g.b); c.stroke();
    c.fillStyle = accent; c.strokeStyle = css("--surface"); c.lineWidth = 2;
    if (Number.isFinite(s.p50[this.idx])) { c.beginPath(); c.arc(x, y(s.p50[this.idx]), 4.5, 0, 7); c.fill(); c.stroke(); }
    if (this.hover >= 0 && this.hover !== this.idx) { c.strokeStyle = ink3; c.lineWidth = 1; c.beginPath(); c.moveTo(this.xOf(this.hover, g), g.t); c.lineTo(this.xOf(this.hover, g), g.h - g.b); c.stroke(); }
    // the key
    c.textAlign = "left"; c.textBaseline = "middle"; c.fillStyle = ink2;
    for (const [draw, label, kx, ky] of place) { if (draw) { draw(kx, ky); c.fillStyle = ink2; c.fillText(label, kx + sw + 5, ky); } else c.fillText(label, kx, ky); }
    if (document.activeElement === cv) { c.strokeStyle = css("--focus"); c.lineWidth = 2; c.strokeRect(1, 1, g.w - 2, g.h - 2); }
  }
}

export function tableHtml(d) {
  const s = d.series, cols = [["Min", "min"], ["5th", "p5"], ["10th", "p10"], ["25th", "p25"], ["Median", "p50"], ["Mean", "mean"], ["75th", "p75"], ["90th", "p90"], ["95th", "p95"], ["Max", "max"], ["PMM", "pmm"], ["LPMM", "lpmm"], ["SAM", "sam"], ["SAM–LPM", "samlpm"]];
  let h = `<table><caption class="sr">${d.label}, inches, at the chosen point</caption><thead><tr><th scope="col">Valid time</th>${cols.map(c => `<th scope="col">${c[0]}</th>`).join("")}</tr></thead><tbody>`;
  for (let i = 0; i < d.times.length; i++) h += `<tr><th scope="row">${stamp(d.times[i])}</th>${cols.map(c => `<td>${fmt(s[c[1]][i])}</td>`).join("")}</tr>`;
  return h + "</tbody></table>";
}
