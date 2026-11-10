// Town search: type a few letters, pick from the list, and the chosen point moves there.
// Matches the start of the name first, then the start of any word in it, then (only if nothing else matches) anywhere in it; bigger places come first.
// A state can be added to narrow it down: "wilton nh", "wilton, new hampshire".

const STATES = { CT: "connecticut", MA: "massachusetts", ME: "maine", NH: "new hampshire", RI: "rhode island", VT: "vermont" };
const plain = s => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[.'’]/g, "").replace(/[^a-z0-9]+/g, " ").trim();

// Split "wilton nh" into the name part and a state code (or null).
export function parseQuery(text) {
  let q = plain(text);
  for (const [code, name] of Object.entries(STATES)) {
    if (q.endsWith(" " + name)) return { name: q.slice(0, -name.length - 1).trim(), st: code };
    if (q.endsWith(" " + code.toLowerCase())) return { name: q.slice(0, -3).trim(), st: code };
  }
  return { name: q, st: null };
}

// places: [name, state, lat, lon, population, zoom].  Returns up to `limit` of them, best first.
export function findPlaces(places, text, limit = 8) {
  const { name: q, st } = parseQuery(text);
  if (!q) return [];
  const hits = [];
  for (const p of places) {
    if (st && p[1] !== st) continue;
    const n = p._n || (p._n = plain(p[0]));
    let rank;
    if (n === q) rank = 0;
    else if (n.startsWith(q)) rank = 1;
    else if (n.includes(" " + q)) rank = 2;
    else if (n.includes(q)) rank = 3;
    else continue;
    hits.push([rank, p]);
  }
  // a match in the middle of a word ("stow" in Kingstown) is shown only when nothing starts with what was typed
  const best = hits.length ? Math.min(...hits.map(h => h[0])) : 0;
  if (best < 3) for (let i = hits.length - 1; i >= 0; i--) if (hits[i][0] === 3) hits.splice(i, 1);
  hits.sort((a, b) => a[0] - b[0] || b[1][4] - a[1][4] || a[1][0].localeCompare(b[1][0]));
  return hits.slice(0, limit).map(h => h[1]);
}

export function initSearch({ input, list, getPlaces, onPick }) {
  let items = [], active = -1;
  const label = p => `${p[0]}, ${p[1]}`;
  const close = () => { list.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); active = -1; };
  const mark = () => {
    [...list.children].forEach((li, i) => li.setAttribute("aria-selected", i === active ? "true" : "false"));
    if (active >= 0) { input.setAttribute("aria-activedescendant", list.children[active].id); list.children[active].scrollIntoView({ block: "nearest" }); }
    else input.removeAttribute("aria-activedescendant");
  };
  const pick = p => { input.value = label(p); close(); onPick(p); };
  const render = () => {
    const text = input.value;
    items = text.trim() ? findPlaces(getPlaces(), text) : [];
    list.replaceChildren();
    if (!items.length) {
      if (text.trim()) {
        const li = document.createElement("li"); li.className = "none"; li.setAttribute("role", "presentation");
        li.textContent = getPlaces().length ? "No town by that name" : "Towns are still loading";
        list.append(li); list.hidden = false; input.setAttribute("aria-expanded", "true");
      } else close();
      return;
    }
    items.forEach((p, i) => {
      const li = document.createElement("li");
      li.id = `search-opt-${i}`; li.setAttribute("role", "option"); li.setAttribute("aria-selected", "false");
      const a = document.createElement("span"); a.textContent = p[0];
      const b = document.createElement("small"); b.textContent = p[1];
      li.append(a, b);
      li.addEventListener("pointerdown", e => { e.preventDefault(); pick(p); });      // before the input loses focus
      li.addEventListener("pointermove", () => { if (active !== i) { active = i; mark(); } });
      list.append(li);
    });
    active = 0; list.hidden = false; input.setAttribute("aria-expanded", "true"); mark();
  };
  input.addEventListener("input", render);
  input.addEventListener("focus", () => { input.select(); if (input.value.trim()) render(); });
  input.addEventListener("blur", () => setTimeout(close, 120));
  input.addEventListener("keydown", e => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (list.hidden) render();
      if (items.length) { active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length; mark(); }
    } else if (e.key === "Enter") {
      if (!list.hidden && items[active]) { e.preventDefault(); pick(items[active]); input.blur(); }
    } else if (e.key === "Escape") {
      if (!list.hidden) { close(); e.stopPropagation(); } else { input.value = ""; }
    }
  });
  document.addEventListener("pointerdown", e => { if (!e.target.closest(".search")) close(); });
}
