// Fan chart of the ensemble at one point: P10-P90 and P25-P75 bands, median, mean and maximum.
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

  // data: { times: [ms], fhr: [h], series: {p10,p25,p50,p75,p90,mean,max: Float32Array}, unit, label }
  setData(d) { this.data = d; this.draw(); }
  setIndex(i) { this.idx = i; this.draw(); }

  geom() {
    const r = this.cv.getBoundingClientRect();
    return { w: r.width, h: r.height, l: 38, r: 12, t: 22, b: 24 };
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
    const s = d.series, rows = [["90th", s.p90], ["Median", s.p50], ["10th", s.p10], ["Mean", s.mean], ["Max", s.max]];
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
    c.font = '12px "Atkinson Hyperlegible Next", system-ui, sans-serif';
    if (!d) return;
    const s = d.series, n = d.times.length;
    // scale to the 90th percentile; the maximum member can be several times larger and would flatten everything else, so it is clipped and labelled
    const maxAll = Math.max(...s.max.filter(Number.isFinite), 0), top = niceMax(Math.max(...s.p90.filter(Number.isFinite), 0.5) * 1.2);
    const y = v => g.t + (1 - Math.min(v, top) / top) * (g.h - g.t - g.b);
    const hair = css("--hair"), ink2 = css("--ink-2"), ink3 = css("--ink-3"), accent = css("--accent-ink");
    // y grid and labels
    const ticks = top <= 4 ? [0, 1, 2, 3, 4].filter(v => v <= top) : [0, top / 4, top / 2, (3 * top) / 4, top].map(v => Math.round(v * 10) / 10);
    c.textAlign = "right"; c.textBaseline = "middle"; c.fillStyle = ink2; c.strokeStyle = hair; c.lineWidth = 1;
    for (const v of ticks) { const yy = Math.round(y(v)) + 0.5; c.beginPath(); c.moveTo(g.l, yy); c.lineTo(g.w - g.r, yy); c.stroke(); c.fillText(fmt(v), g.l - 6, yy); }
    // x: a tick at each 00 UTC, weekday and date
    c.textAlign = "center"; c.textBaseline = "top";
    for (let i = 0; i < n; i++) {
      const dt = new Date(d.times[i]);
      if (dt.getUTCHours() !== 0) continue;
      const x = Math.round(this.xOf(i, g)) + 0.5;
      c.strokeStyle = hair; c.beginPath(); c.moveTo(x, g.t); c.lineTo(x, g.h - g.b); c.stroke();
      c.fillStyle = ink2; c.fillText(`${DAYS[dt.getUTCDay()]} ${dt.getUTCDate()}`, Math.min(Math.max(x, g.l + 14), g.w - 22), g.h - g.b + 6);
    }
    const band = (lo, hi, col) => {
      c.beginPath();
      for (let i = 0; i < n; i++) c[i ? "lineTo" : "moveTo"](this.xOf(i, g), y(hi[i]));
      for (let i = n - 1; i >= 0; i--) c.lineTo(this.xOf(i, g), y(lo[i]));
      c.closePath(); c.fillStyle = col; c.fill();
    };
    band(s.p10, s.p90, css("--band-a")); band(s.p25, s.p75, css("--band-b"));
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
    c.beginPath(); c.arc(x, y(s.p50[this.idx]), 4.5, 0, 7); c.fill(); c.stroke();
    if (this.hover >= 0 && this.hover !== this.idx) { c.strokeStyle = ink3; c.lineWidth = 1; c.beginPath(); c.moveTo(this.xOf(this.hover, g), g.t); c.lineTo(this.xOf(this.hover, g), g.h - g.b); c.stroke(); }
    // legend, direct in the chart
    c.textAlign = "left"; c.textBaseline = "middle"; let lx = g.l, ly = 9;
    const key = (label, draw) => { draw(lx, ly); c.fillStyle = ink2; c.fillText(label, lx + 22, ly); lx += 22 + c.measureText(label).width + 14; };
    c.fillStyle = ink2; c.fillText(d.label, lx, ly); lx += c.measureText(d.label).width + 18;
    key("Median", (x0, y0) => { c.strokeStyle = accent; c.lineWidth = 2.25; c.beginPath(); c.moveTo(x0, y0); c.lineTo(x0 + 16, y0); c.stroke(); });
    key("Mean", (x0, y0) => { c.strokeStyle = ink2; c.lineWidth = 1.5; c.setLineDash([5, 3]); c.beginPath(); c.moveTo(x0, y0); c.lineTo(x0 + 16, y0); c.stroke(); c.setLineDash([]); });
    key("Max", (x0, y0) => { c.strokeStyle = ink3; c.lineWidth = 1.25; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(x0, y0); c.lineTo(x0 + 16, y0); c.stroke(); c.setLineDash([]); });
    key("25–75%", (x0, y0) => { c.fillStyle = css("--band-b"); c.fillRect(x0, y0 - 5, 16, 10); });
    key("10–90%", (x0, y0) => { c.fillStyle = css("--band-a"); c.fillRect(x0, y0 - 5, 16, 10); });
    if (document.activeElement === cv) { c.strokeStyle = css("--focus"); c.lineWidth = 2; c.strokeRect(1, 1, g.w - 2, g.h - 2); }
  }
}

export function tableHtml(d) {
  const s = d.series, cols = [["10th", "p10"], ["25th", "p25"], ["Median", "p50"], ["75th", "p75"], ["90th", "p90"], ["Mean", "mean"], ["Max", "max"]];
  let h = `<table><caption class="sr">${d.label}, inches, at the chosen point</caption><thead><tr><th scope="col">Valid time</th>${cols.map(c => `<th scope="col">${c[0]}</th>`).join("")}</tr></thead><tbody>`;
  for (let i = 0; i < d.times.length; i++) h += `<tr><th scope="row">${stamp(d.times[i])}</th>${cols.map(c => `<td>${fmt(s[c[1]][i])}</td>`).join("")}</tr>`;
  return h + "</tbody></table>";
}
