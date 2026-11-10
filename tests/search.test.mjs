// findPlaces / parseQuery from site/js/search.js, run by tests/test_places.py
import { findPlaces, parseQuery } from "../site/js/search.js";
import { readFileSync } from "node:fs";
const places = JSON.parse(readFileSync(process.argv[2], "utf8"));
const fail = m => { console.error("FAIL " + m); process.exit(1); };
const names = (q, n) => findPlaces(places, q, n).map(p => `${p[0]},${p[1]}`);

let r = parseQuery("Wilton, New Hampshire"); if (r.name !== "wilton" || r.st !== "NH") fail("state name " + JSON.stringify(r));
r = parseQuery("wilton nh"); if (r.name !== "wilton" || r.st !== "NH") fail("state code " + JSON.stringify(r));
r = parseQuery("Portland"); if (r.st !== null) fail("no state");
r = parseQuery("Washington Maine"); if (r.name !== "washington" || r.st !== "ME") fail("two-word " + JSON.stringify(r));

if (names("port")[0] !== "Portland,ME") fail("bigger first: " + names("port"));            // prefix matches, largest first
if (!names("portland", 5).includes("Portland,CT")) fail("both Portlands");
if (names("portland me")[0] !== "Portland,ME" || names("portland me").some(n => !n.endsWith(",ME"))) fail("state filter");
if (!names("conway").includes("North Conway,NH")) fail("word prefix: " + names("conway"));  // "North Conway" found by "conway"
if (names("stowe")[0] !== "Stowe,VT") fail("small town: " + names("stowe"));
if (names("zzzz").length) fail("no match");
if (names("  ").length) fail("blank");
if (names("o'", 8).length < 0) fail("odd input");
console.log("ok");
