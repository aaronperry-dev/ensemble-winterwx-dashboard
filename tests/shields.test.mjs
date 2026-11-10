// shieldSvg / shieldSize from site/js/shields.js, run by tests/test_roads.py
import { shieldSvg, shieldSize, STATE_SHAPES } from "../site/js/shields.js";
const fail = m => { console.error("FAIL " + m); process.exit(1); };

const i95 = shieldSvg(0, "", "95");
if (!i95.startsWith("<svg") || !i95.includes('data-route="95"')) fail("interstate marker");
if ((i95.match(/<path d="[^"]+" transform="translate/g) || []).length !== 2) fail("interstate numerals: one glyph per digit");
if (!i95.includes("#003f87") || !i95.includes("#af1e2d")) fail("interstate blue and red (MUTCD PMS 294 and 187)");
const us = shieldSvg(1, "", "302");
if (!us.includes('data-route="302"') || !us.includes('fill="#fff"') || !us.includes('fill="#000"')) fail("US route: white shield, black numerals");
if (us.includes("#003f87")) fail("US route must not be blue");
const states = Object.keys(STATE_SHAPES);
if (!["ME", "NH", "VT", "MA", "CT", "RI"].every(s => states.includes(s))) fail("a New England state has no shape");
const shapes = new Set(["ME", "NH", "VT", "MA", "CT", "RI"].map(s => shieldSvg(2, s, "9").match(/<path d="([^"]+)" fill="#fff"/)[1]));
if (shapes.size < 5) fail("state shields should differ in shape, got " + shapes.size);
if (shieldSvg(2, "ZZ", "9") !== shieldSvg(2, "ZZ", "9")) fail("cached");
if (!shieldSvg(2, "ZZ", "9").includes('data-route="9"')) fail("unknown state still gets a (round) shield");
const a = shieldSize("9"), b = shieldSize("95"), c = shieldSize("495");
if (!(a.w <= b.w && b.w < c.w)) fail("three characters get the wider (30 x 24) sign");
if (a.ph !== b.ph || a.ph < 18) fail("height");
if (shieldSvg(0, "", "495").split("<path d=").length < 5) fail("three-digit numerals");
if (!shieldSvg(0, "", "9W").includes('data-route="9W"') || (shieldSvg(0, "", "9W").match(/translate/g) || []).length !== 2) fail("letter suffix");
console.log("ok");
