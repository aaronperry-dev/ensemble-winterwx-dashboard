// usage: node bundle.test.mjs <bundle.bin.gz> <expected.json>   (run by tests/test_viewer_bundle.py)
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { pathToFileURL } from "node:url";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const { parseBundle } = await import(pathToFileURL(resolve(here, "../site/js/bundle.js")).href);
const [file, expectedFile] = process.argv.slice(2);
const raw = new Uint8Array(gunzipSync(readFileSync(file)));
const { blocks } = parseBundle(raw);
const expected = JSON.parse(readFileSync(expectedFile, "utf8"));
let bad = 0;
for (const [name, want] of Object.entries(expected)) {
  const b = blocks[name];
  for (let i = 0; i < want.length; i++) {
    const q = b.data[i], got = q === b.missing ? null : b.lo + b.step * q;
    const same = want[i] === null ? got === null : got !== null && Math.abs(got - want[i]) < 1e-4;
    if (!same && bad++ < 5) console.error(`${name}[${i}]: got ${got}, want ${want[i]}`);
  }
}
if (bad) { console.error(`${bad} mismatches`); process.exit(1); }
console.log("bundle decoder OK");
