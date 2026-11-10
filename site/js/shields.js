// Route shields drawn as small inline SVGs: the Interstate and U.S. route shields, and a shape for each New England state's own routes.
// The state shapes are simplified (plain outline, number only); change them in STATE_SHAPES.  A route not listed gets the round default.

const NS = "http://www.w3.org/2000/svg";
const H = 26;                                    // shields are drawn in a box 26 units tall (the width grows with the number of digits) ...
const SCALE = 0.86;                              // ... and shown at this fraction of that, in pixels

const INTERSTATE = { blue: "#00478f", red: "#c8102e" };

// Outlines are drawn in a box of w x H.  Each returns an SVG path (or an element description for round shapes).
const SHAPES = {
  interstate: w => `M1.5 7.5 Q${w * 0.26} 0.5 ${w / 2} 4 Q${w * 0.74} 0.5 ${w - 1.5} 7.5 L${w - 1.5} 16 Q${w - 1.5} 22.5 ${w / 2} ${H - 0.8} Q1.5 22.5 1.5 16 Z`,
  us: w => `M1.2 7 Q${w * 0.25} 1 ${w / 2} 4.2 Q${w * 0.75} 1 ${w - 1.2} 7 L${w - 1.2} 14.5 Q${w - 1.2} 21.5 ${w / 2} ${H - 0.8} Q1.2 21.5 1.2 14.5 Z`,
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

export function shieldSize(num) {
  const w = num.length >= 3 ? 31 : num.length === 2 ? 26 : 24;
  return { w, h: H, pw: Math.round(w * SCALE), ph: Math.round(H * SCALE) };
}

/** SVG markup of one shield.  cls: 0 Interstate, 1 U.S. route, 2 state route; st: state abbreviation (state routes only). */
export function shieldSvg(cls, st, num) {
  const key = `${cls}|${cls === 2 ? st : ""}|${num}`;
  if (cache.has(key)) return cache.get(key);
  const { w } = shieldSize(num), fs = num.length >= 3 ? 11 : num.length === 2 ? 14 : 15;
  let body, tl = 21, ink = "#111", ty = cls === 0 ? 17.6 : cls === 1 ? 16.5 : 17.2;
  if (cls === 0) {
    const d = SHAPES.interstate(w), id = `ci${w}`;
    body = `<defs><clipPath id="${id}"><path d="${d}"/></clipPath></defs>
      <path d="${d}" fill="none" stroke="rgba(0,0,0,.4)" stroke-width="3.6" stroke-linejoin="round"/>
      <path d="${d}" fill="${INTERSTATE.blue}"/>
      <rect x="0" y="0" width="${w}" height="9.2" fill="${INTERSTATE.red}" clip-path="url(#${id})"/>
      <path d="${d}" fill="none" stroke="#fff" stroke-width="1.7"/>`;
    ink = "#fff";
  } else {
    const shape = cls === 1 ? "us" : (STATE_SHAPES[st] || "circle");
    const d = SHAPES[shape](w);
    body = `<path d="${d}" fill="none" stroke="rgba(0,0,0,.4)" stroke-width="4.4" stroke-linejoin="round"/>` +
      `<path d="${d}" fill="#fff" stroke="#111" stroke-width="1.8" stroke-linejoin="round"/>`;
    ty = 17.4;
    if (shape === "circle") tl = 16;
  }
  const svg = `<svg xmlns="${NS}" width="${Math.round((w + 2) * SCALE)}" height="${Math.round(H * SCALE)}" viewBox="-1 0 ${w + 2} ${H}" aria-hidden="true">${body}` +
    `<text x="${w / 2}" y="${ty}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-weight="700" font-size="${fs}" fill="${ink}"` +
    `${num.length >= 3 ? ` textLength="${tl}" lengthAdjust="spacingAndGlyphs"` : ""}>${num}</text></svg>`;
  cache.set(key, svg);
  return svg;
}
