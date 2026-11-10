// Route shields drawn as small inline SVGs.
//   Interstate and U.S. route shields: the outlines of MUTCD sign M1-1 and M1-4 (FHWA Standard Highway Signs, 24 x 24 inches; 30 x 24 for three digits) with the
//   FHWA Series D numerals, which narrow to Series C and B when a number does not fit (the rule of the Standard Highway Signs).  The artwork is in mutcd-art.js.
//   State routes: a shape for each New England state's own route marker, simplified (plain outline, number only); change them in STATE_SHAPES.
//   A route not listed gets the round default.

import { ART } from "./mutcd-art.js";

const NS = "http://www.w3.org/2000/svg";
const H = 24;                                    // every shield is drawn in a box 24 units tall (the width is 24, or 30 for three characters) ...
const SCALE = 0.95;                              // ... and shown at this many pixels per unit
const US_SQUARE = true;                          // the U.S. route sign is a white shield on a black square (M1-4); false draws the shield alone with a black outline

// Outlines of the state shields, in a box of w x H.  Each returns an SVG path.
const SHAPES = {
  circle: w => `M${w / 2} 1.2 A${H / 2 - 1.2} ${H / 2 - 1.2} 0 1 1 ${w / 2} ${H - 1.2} A${H / 2 - 1.2} ${H / 2 - 1.2} 0 1 1 ${w / 2} 1.2 Z`,
  // Maine: an upright oval
  oval: w => `M${w / 2} 1.2 A${w / 2 - 1.2} ${H / 2 - 1.2} 0 1 1 ${w / 2} ${H - 1.2} A${w / 2 - 1.2} ${H / 2 - 1.2} 0 1 1 ${w / 2} 1.2 Z`,
  // Vermont: a rectangle whose left and right sides bulge outwards
  bulge: w => `M5 1.5 H${w - 5} Q${w + 0.6} ${H / 2} ${w - 5} ${H - 1.5} H5 Q-0.6 ${H / 2} 5 1.5 Z`,
  // Connecticut: a rectangle with an arched top
  arch: w => `M1.5 ${H - 1.5} V11 Q1.5 1.5 ${w / 2} 1.5 Q${w - 1.5} 1.5 ${w - 1.5} 11 V${H - 1.5} Z`,
  // Massachusetts: a rounded square
  rounded: w => `M7 1.5 H${w - 7} Q${w - 1.5} 1.5 ${w - 1.5} 7 V${H - 7} Q${w - 1.5} ${H - 1.5} ${w - 7} ${H - 1.5} H7 Q1.5 ${H - 1.5} 1.5 ${H - 7} V7 Q1.5 1.5 7 1.5 Z`,
  // Rhode Island: a rectangle with cut corners
  cut: w => `M6 1.5 H${w - 6} L${w - 1.5} 6 V${H - 6} L${w - 6} ${H - 1.5} H6 L1.5 ${H - 6} V6 Z`,
};
export const STATE_SHAPES = { ME: "oval", NH: "circle", VT: "bulge", MA: "rounded", CT: "arch", RI: "cut", NY: "circle" };

const cache = new Map();
const COLORS = { white: "#fff", black: "#000", red: "#af1e2d", blue: "#003f87" };          // MUTCD: white, black, PMS 187 red, PMS 294 blue

export function shieldSize(num) {
  const w = num.length >= 3 ? 30 : 24;
  return { w, h: H, pw: Math.round(w * SCALE), ph: Math.round(H * SCALE) };
}

// ---- FHWA lettering: the first series in which the text fits within maxWidth at the given cap height, else the narrowest, shrunk to fit ----
const CHAIN = [ART.font.D, ART.font.C, ART.font.B];
const advance = (text, font) => [...text].reduce((s, c) => s + (font[c] ? font[c][0] : 0), 0);
const r3 = v => Math.round(v * 1000) / 1000;
function lettering(text, { height, x, y, maxWidth }) {
  let font = CHAIN.find(f => advance(text, f) * height <= maxWidth), h = height;
  if (!font) { font = CHAIN[CHAIN.length - 1]; h = Math.max(height * 0.5, maxWidth / Math.max(advance(text, font), 1e-6)); }
  let pen = x - (advance(text, font) * h) / 2, out = "";
  for (const c of text) {
    const g = font[c];
    if (!g) continue;
    out += `<path d="${g[1]}" transform="translate(${r3(pen)} ${y}) scale(${r3(h)})"/>`;
    pen += g[0] * h;
  }
  return out;
}

/** SVG markup of one shield.  cls: 0 Interstate, 1 U.S. route, 2 state route; st: state abbreviation (state routes only). */
export function shieldSvg(cls, st, num) {
  const key = `${cls}|${cls === 2 ? st : ""}|${num}`;
  if (cache.has(key)) return cache.get(key);
  const { w } = shieldSize(num), wide = w > 24, sx = w / 24;
  let body;
  if (cls === 0) {
    const art = ART.interstate;
    const outer = art[0][0];
    const paths = art.map(([d, f]) => `<path d="${d}" fill="${COLORS[f] || f}"/>`).join("");
    body = `<g transform="scale(${sx} 1)"><path d="${outer}" fill="none" stroke="rgba(0,0,0,.45)" stroke-width="2.6" stroke-linejoin="round"/>${paths}</g>` +
      `<g fill="#fff">${lettering(num, { height: 11, x: w / 2, y: 17.5, maxWidth: wide ? 23.5 : 17.6 })}</g>`;
  } else if (cls === 1) {
    const [sq, shield] = ART.us;
    body = `<g transform="scale(${sx} 1)">` + (US_SQUARE
      ? `<path d="${sq[0]}" fill="${COLORS[sq[1]]}"/><path d="${shield[0]}" fill="${COLORS[shield[1]]}"/>`
      : `<path d="${shield[0]}" fill="${COLORS[shield[1]]}" stroke="#000" stroke-width="1.8" stroke-linejoin="round"/>`) + `</g>` +
      `<g fill="#000">${lettering(num, { height: 12, x: w / 2, y: 17.5, maxWidth: wide ? 25.5 : 20.3 })}</g>`;
  } else {
    const shape = STATE_SHAPES[st] || "circle", d = SHAPES[shape](w);
    body = `<path d="${d}" fill="none" stroke="rgba(0,0,0,.4)" stroke-width="4.4" stroke-linejoin="round"/>` +
      `<path d="${d}" fill="#fff" stroke="#111" stroke-width="1.8" stroke-linejoin="round"/>` +
      `<g fill="#111">${lettering(num, { height: 11.5, x: w / 2, y: 17.75, maxWidth: shape === "circle" ? (wide ? 21 : 14.5) : wide ? 22 : 16.5 })}</g>`;
  }
  const svg = `<svg xmlns="${NS}" width="${Math.round((w + 2) * SCALE)}" height="${Math.round(H * SCALE)}" viewBox="-1 0 ${w + 2} ${H}" data-route="${num}" aria-hidden="true">${body}</svg>`;
  cache.set(key, svg);
  return svg;
}
