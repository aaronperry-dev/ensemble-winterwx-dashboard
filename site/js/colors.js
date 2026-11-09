// Colour scales.  Each scale is a list of class edges and one colour per class; values below the first edge are not painted.
//   snowfall:     the sequential blue ramp (steps 100 -> 700), light = little snow
//   probability:  a second sequential ramp (the palette's orange hue at the same lightness steps)
//   snow ratio:   diverging, low ratio (wet, dense snow) red <-> high ratio (dry, fluffy snow) blue, gray midpoint
// In the dark appearance the sequential ramps run the other way (the lightest colour is the largest value), as the dark surface needs.

const BLUE = { 100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7", 400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281", 700: "#0d366b" };
const ORANGE = ["#fed6c7", "#fec0aa", "#fea98b", "#fe9068", "#f97948", "#e96a38", "#d85a27", "#c74b12", "#b44004", "#9e3703", "#892f02", "#752601", "#611e01"];

const SNOW_EDGES = [0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48];
const SNOW_STEPS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700];
const PROB_EDGES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90];
const PROB_IDX = [0, 1, 2, 3, 4, 5, 7, 9, 11, 12];
const SLR_EDGES = [-Infinity, 5, 7, 9, 11, 13, 16, 20, 25];
const SLR_LIGHT = ["#a3262a", "#bb5e59", "#d18f88", "#e2bfb9", "#f0efec", "#bbcade", "#87a6ce", "#5481bd", "#1c5cab"];
const SLR_DARK = ["#f08a80", "#bf756d", "#906159", "#634c47", "#383835", "#505960", "#697b8f", "#839fc0", "#9ec5f4"];

function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

function make(id, edges, colors, fmt, label) {
  return { id, edges, colors, rgb: colors.map(hexToRgb), fmt, label,
    classOf(v) { if (!(v >= edges[0])) return -1; let c = edges.length - 1; while (v < edges[c]) c--; return c; } };
}

export function scales(dark) {
  const snow = SNOW_STEPS.map(s => BLUE[s]), prob = PROB_IDX.map(i => ORANGE[i]);
  const ord = a => (dark ? a.slice().reverse() : a);
  return {
    snow: make("snow", SNOW_EDGES, ord(snow), v => (v < 10 ? v.toFixed(v % 1 ? 1 : 0) : v.toFixed(0)), "in"),
    prob: make("prob", PROB_EDGES, ord(prob), v => v.toFixed(0), "%"),
    slr: make("slr", SLR_EDGES, dark ? SLR_DARK : SLR_LIGHT, v => (Number.isFinite(v) ? v.toFixed(0) : ""), ":1"),
  };
}

export const rgbaPacked = (rgb, a = 255) => (a << 24) | (rgb[2] << 16) | (rgb[1] << 8) | rgb[0];     // little-endian ImageData word
