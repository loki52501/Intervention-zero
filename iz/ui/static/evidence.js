"use strict";
const $ = (id) => document.getElementById(id);
function h(tag, cls, ...kids) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  for (const k of kids) {
    if (k == null || k === false) continue;
    e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return e;
}
const err = (s) => h("div", "err", "Unavailable: " + (s.error || "unknown"));
const short = (v, n = 90) => { v = v == null ? "" : String(v); return v.length > n ? v.slice(0, n) + "..." : v; };
const kv = (k, v) => h("div", "kv", h("span", null, k), h("b", null, v == null ? "-" : v));
function snip(pairs) {
  const d = h("div", "snip");
  const rows = pairs.filter(([, v]) => v != null && v !== "");
  if (!rows.length) { d.textContent = "no documents yet"; return d; }
  rows.forEach(([k, v], i) => { if (i) d.append(h("br")); d.append(h("b", null, k + ": "), short(v)); });
  return d;
}

function renderLoop(s) {
  const box = $("loop"); box.replaceChildren();
  if (!s.ok) return box.append(err(s));
  for (const c of s.cards) {
    const n = c.newest || {};
    const card = h("div", "card", h("h3", null, c.title),
      h("div", "big", c.count == null ? "v" + (c.active_version ?? "-") : c.count.toLocaleString()));
    if (c.key === "takeovers") card.append(snip([["tip", n.operator_tip], ["site", n.site], ["situation", n.situation_type]]));
    else if (c.key === "edits") card.append(snip([["kind", n.kind], ["id", n._id]]));
    else if (c.key === "decisions") {
      if (n.verdict) card.append(h("div", null, h("span", "verdict " + n.verdict, n.verdict)));
      card.append(snip([["reason", n.reason], ["v", n.parent_version + " -> " + (n.new_version ?? "-")]]));
      const v = Object.entries(c.verdicts || {}).map(([k, x]) => k + " " + x).join(" | ");
      if (v) card.append(h("div", "note", v));
      card.append(h("div", "note", "transactions committed: " + c.extra.transactions_committed));
    } else if (c.key === "genomes") {
      card.append(h("div", "snip", h("div", "chain", (c.lineage || []).map((x) => "v" + x).join(" → ") || "no lineage"),
        h("div", null, "active: v" + (c.active_version ?? "-"))));
    } else if (c.key === "active") {
      card.append(snip([["version", n.version], ["updated", n.updated_at]]));
      card.append(h("div", "note", "change stream reloads: " + c.extra.change_stream_reloads));
    }
    card.append(h("span", "pill", c.pill));
    box.append(card);
  }
}

function renderSafety(s) {
  const box = $("safety"); box.replaceChildren();
  if (!s.ok) return box.append(err(s));
  const p = s.probe || {};
  const wrap = h("div", "card safety");
  wrap.append(h("div", "stamp" + (p.refused ? "" : " none"), p.refused ? "DATABASE REFUSED" : "NOT REFUSED"));
  const d = h("div");
  if (p.refused) {
    d.append(h("div", null, "Live probe on ", h("code", null, p.db), ": insert genome with ", h("code", null, p.attempted)),
      h("div", null, "Server error code ", h("b", null, p.code)), h("pre", null, p.errmsg));
  } else d.append(h("div", "err", p.error || p.note || "probe unavailable"));
  d.append(h("div", "note", "validator_refused events logged by the app: " + s.refusal_events));
  if (s.event) d.append(h("pre", null, "latest: " + JSON.stringify(s.event.payload || s.event)));
  wrap.append(d); box.append(wrap);
}

function mcard(title, sub, count, rows) {
  const c = h("div", "card", h("h3", null, title), h("div", "big", count == null ? "-" : Number(count).toLocaleString()),
    h("div", "note", sub));
  if (rows) c.append(rows);
  return c;
}
function renderMemory(s) {
  const box = $("memory"), cb = $("card"); box.replaceChildren(); cb.replaceChildren();
  if (!s.ok) return box.append(err(s));
  const e = s.episodic, f = s.semantic, p = s.procedural, t = s.taxonomic;
  box.append(
    mcard("Episodic", e.collection, e.count, snip([["tip", (e.newest || {}).operator_tip], ["failure", (e.newest || {}).failure], ["venue", (e.newest || {}).venue]])),
    mcard("Semantic", f.collection, f.count, snip([["fact", (f.newest || {}).fact || (f.newest || {}).text], ["venue", (f.newest || {}).venue]])),
  );
  const nodes = (t.nodes || []).slice().sort((a, b) => b.count - a.count);
  const mx = Math.max(1, ...nodes.map((n) => n.count));
  const tree = h("div", "snip");
  nodes.slice(0, 7).forEach((n) => { const r = h("div", "bar", h("span", null, short(n.name, 20)), h("i")); r.lastChild.style.width = (n.count / mx * 90 + 3) + "px"; r.append(String(n.count)); tree.append(r); });
  box.append(mcard("Taxonomic", "nodes, shared across worlds", t.count, tree),
    mcard("Procedural", p.collection + " + genome rulebook", p.count,
      snip([["rulebook rules (genome v" + p.newest_genome + ")", p.rulebook_rules_in_newest_genome], ["newest", (p.newest || {}).text]])));
  const rc = s.robot_card;
  cb.append(h("div", "card", h("h3", null, "Robot card (" + s.robot_cards_count + " stored)"),
    rc ? h("div", null, rc.robot + " — sensors: " + (rc.sensors ?? "n/a") + ", actuators: " + (rc.actuators ?? "n/a")) : "no robot card yet"));
  cb.firstChild.style.marginTop = "16px";
}

function renderRecallMeta(s) {
  const m = $("recall");
  if (m.dataset.busy) return;
  if (!m.dataset.has) {
    m.replaceChildren(!s.ok ? err(s) : h("div", "note", s.takeovers_searchable + " takeovers searchable. " + s.hint +
      (s.last_path ? ". Last path used by the app: " + s.last_path : "")));
  }
}
async function doRecall(q) {
  const m = $("recall"); m.dataset.busy = "1"; m.dataset.has = "1"; m.replaceChildren(h("div", "note", "searching..."));
  try {
    const r = await (await fetch("/api/evidence/recall?q=" + encodeURIComponent(q))).json();
    m.replaceChildren();
    if (r.ok === false) m.append(err(r));
    m.append(h("div", "tags", h("span", null, "path: " + r.path), h("span", null, "rerank: " + (r.rerank ? "on" : "off")),
      h("span", null, "embedder: " + (r.embedder || "-"))));
    (r.hits || []).forEach((x) => m.append(h("div", "hit", h("b", null, (x.situation_type || "") + " @ " + (x.site || "")),
      x.score != null ? " (score " + Number(x.score).toFixed(3) + ")" : "", h("div", null, short(x.operator_tip, 200)))));
    if (!(r.hits || []).length) m.append(h("div", "note", "no hits"));
  } catch (e) { m.replaceChildren(err({ error: String(e) })); }
  delete m.dataset.busy;
}

function spark(series) {
  const ns = "http://www.w3.org/2000/svg", W = 400, H = 60;
  const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "spark");
  svg.setAttribute("preserveAspectRatio", "none");
  if (series.length < 2) return h("div", "note", series.length ? "1 bucket: " + series[0].count : "no takeover series yet");
  const mx = Math.max(...series.map((x) => x.count));
  const pts = series.map((x, i) => `${(i / (series.length - 1)) * W},${H - 4 - (x.count / mx) * (H - 8)}`).join(" ");
  const pl = document.createElementNS(ns, "polyline");
  pl.setAttribute("points", pts); pl.setAttribute("fill", "none"); pl.setAttribute("stroke", "#3b5bdb"); pl.setAttribute("stroke-width", "2.5");
  svg.append(pl); return svg;
}
function counts(title, obj) {
  const c = h("div", "card", h("h3", null, title));
  const e = Object.entries(obj || {});
  if (!e.length) c.append(h("div", "note", "none"));
  e.forEach(([k, v]) => c.append(kv(k, v)));
  return c;
}
function renderRaindrop(s) {
  const box = $("raindrop"); box.replaceChildren();
  if (!s.ok) return box.append(err(s));
  if (s.loading) box.append(h("div", "note", "querying Raindrop Query API..."));
  else if (s.degraded) {
    box.append(h("div", "warn", s.message || "Raindrop degraded - falling back to MongoDB"), h("div", "note", s.error));
    if (s.mongo_fallback) box.append(h("div", "note", "MongoDB takeovers: " + (s.mongo_fallback.takeovers ?? s.mongo_fallback.error)));
  } else {
    const g = h("div", "two", counts("Events by name (" + s.window + ", " + s.total_events + " total)", s.events), counts("Signals by name", s.signals));
    box.append(g, h("div", "card", h("h3", null, "'takeover' signals per hour"), spark(s.takeover_series || [])));
    box.lastChild.style.marginTop = "16px";
  }
  box.append(h("div", "note", "Sent this session: " + s.local.events_sent_this_session + " events (Raindrop write " + (s.local.enabled ? "enabled" : "disabled") + ")"));
  box.append(h("div", "note", s.fixer_reads));
}

let failures = 0;
async function tick() {
  try {
    const r = await fetch("/api/evidence/summary"); const s = await r.json();
    renderLoop(s.loop); renderSafety(s.safety); renderMemory(s.memory); renderRecallMeta(s.recall); renderRaindrop(s.raindrop);
    $("dot").className = "dot ok"; $("stamp").textContent = "live · " + new Date().toLocaleTimeString(); $("dbname").textContent = "db: " + s.db;
    failures = 0;
  } catch (e) { failures++; $("dot").className = "dot bad"; $("stamp").textContent = "disconnected (" + failures + ")"; }
}
$("rf").addEventListener("submit", (ev) => { ev.preventDefault(); const q = $("q").value.trim(); if (q) doRecall(q); });
tick(); setInterval(tick, 3000);
