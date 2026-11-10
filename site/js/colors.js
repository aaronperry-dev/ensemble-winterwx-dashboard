// Colour scales.  Each scale is a list of class edges and one colour per class; values below the first edge are not painted.
//   liquid precip: the NWS "Total Precipitation Forecast" legend of the same office, the same in light and dark
//   snowfall:     the NWS "Total Snow Forecast" legend used by the Portland / Gray, ME office (colours sampled from its map), the same in light and dark
//   probability:  ColorBrewer sequential "BuPu" (light blue to deep purple), 8 of its 9 classes; a different hue from the amounts so the two never read alike
//   snow ratio:   diverging, low ratio (wet, dense snow) red <-> high ratio (dry, fluffy snow) blue, gray midpoint
// The NWS ramps are a fixed convention and is never recoloured.  The probability ramp runs the other way on the dark appearance
// (the lightest colour is the largest value), as a dark surface needs.

const NWS_SNOW = {
  edges: [0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 30, 36],
  colors: ["#bdd7e7", "#6baed6", "#3182bd", "#08519c", "#082694", "#ffff96", "#ffc400", "#ff8700", "#db1400", "#9e0000", "#690000", "#360000"],
  labels: ["<1", "1–2", "2–3", "3–4", "4–6", "6–8", "8–12", "12–18", "18–24", "24–30", "30–36", "36+"],
};
// ColorBrewer BuPu, 9 classes: #f7fcfd #e0ecf4 #bfd3e6 #9ebcda #8c96c6 #8c6bb1 #88419d #810f7c #4d004b.  The near-white first class is dropped.
const BREWER_BUPU = ["#e0ecf4", "#bfd3e6", "#9ebcda", "#8c96c6", "#8c6bb1", "#88419d", "#810f7c", "#4d004b"];
const PROB_EDGES = [5, 10, 20, 30, 40, 50, 70, 90];
const PROB_LABELS = ["5–10", "10–20", "20–30", "30–40", "40–50", "50–70", "70–90", "90+"];
// Total liquid precipitation (QPF): the "Total Precipitation Forecast" legend of the NWS Portland / Gray, ME office (colours sampled from its map), the same in light and dark.
// Its first class (0.00) is unpainted, as on that map.
const QPF_EDGES = [0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30];
const QPF_COLORS = ["#c7e9c0", "#a1d99b", "#74c476", "#31a353", "#006d2c", "#ffff8a", "#ffcc4f", "#fe8d3c", "#fc4e2a", "#d61a1c", "#ad0026", "#700026", "#3b0030", "#4c0073", "#ffdbff"];
const QPF_LABELS = ["0.01–0.09", "0.1–0.24", "0.25–0.49", "0.5–0.99", "1–1.49", "1.5–1.99", "2–2.99", "3–3.99", "4–5.99", "6–7.99", "8–9.99", "10–14.99", "15–19.99", "20–29.99", "30+"];
const SLR_EDGES = [-Infinity, 5, 7, 9, 11, 13, 16, 20, 25];
const SLR_LIGHT = ["#a3262a", "#bb5e59", "#d18f88", "#e2bfb9", "#f0efec", "#bbcade", "#87a6ce", "#5481bd", "#1c5cab"];
const SLR_DARK = ["#f08a80", "#bf756d", "#906159", "#634c47", "#383835", "#505960", "#697b8f", "#839fc0", "#9ec5f4"];

function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

function make(id, edges, colors, fmt, label, labels) {
  return { id, edges, colors, rgb: colors.map(hexToRgb), fmt, label, labels,
    classOf(v) { if (!(v >= edges[0])) return -1; let c = edges.length - 1; while (v < edges[c]) c--; return c; } };
}

export function scales(dark) {
  const ord = a => (dark ? a.slice().reverse() : a);
  return {
    snow: make("snow", NWS_SNOW.edges, NWS_SNOW.colors, v => (v < 10 ? v.toFixed(v % 1 ? 1 : 0) : v.toFixed(0)), "in", NWS_SNOW.labels),
    qpf: make("qpf", QPF_EDGES, QPF_COLORS, v => (v < 1 ? String(+v.toFixed(2)) : v.toFixed(v % 1 ? 1 : 0)), "in", QPF_LABELS),
    prob: make("prob", PROB_EDGES, ord(BREWER_BUPU), v => v.toFixed(0), "%", PROB_LABELS),
    slr: make("slr", SLR_EDGES, dark ? SLR_DARK : SLR_LIGHT, v => (Number.isFinite(v) ? v.toFixed(0) : ""), ":1"),
  };
}

export const rgbaPacked = (rgb, a = 255) => (a << 24) | (rgb[2] << 16) | (rgb[1] << 8) | rgb[0];     // little-endian ImageData word
