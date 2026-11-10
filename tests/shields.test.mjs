// shieldSvg / shieldSize from site/js/shields.js, run by tests/test_roads.py
import { shieldSvg, shieldSize, STATE_SHAPES } from "../site/js/shields.js";
const fail = m => { console.error("FAIL " + m); process.exit(1); };

const i95 = shieldSvg(0, "", "95");
if (!i95.startsWith("<svg") || !i95.includes(">95</text>")) fail("interstate text");
if (!i95.includes("#00478f") || !i95.includes("#c8102e")) fail("interstate blue and red");
const us = shieldSvg(1, "", "302");
if (!us.includes(">302</text>") || !us.includes('fill="#fff"')) fail("US route");
if (us.includes("#00478f")) fail("US route must not be blue");
const states = Object.keys(STATE_SHAPES);
if (!["ME", "NH", "VT", "MA", "CT", "RI"].every(s => states.includes(s))) fail("a New England state has no shape");
const shapes = new Set(["ME", "NH", "VT", "MA", "CT", "RI"].map(s => shieldSvg(2, s, "9").match(/<path d="([^"]+)" fill="#fff"/)[1]));
if (shapes.size < 5) fail("state shields should differ in shape, got " + shapes.size);
if (shieldSvg(2, "ZZ", "9") !== shieldSvg(2, "ZZ", "9")) fail("cached");
if (!shieldSvg(2, "ZZ", "9").includes(">9</text>")) fail("unknown state still gets a (round) shield");
const a = shieldSize("9"), b = shieldSize("95"), c = shieldSize("495");
if (!(a.w < b.w && b.w < c.w)) fail("wider for more digits");
if (a.ph !== b.ph || a.ph < 18) fail("height");
if (!shieldSvg(0, "", "9W").includes(">9W</text>")) fail("letter suffix");
console.log("ok");
