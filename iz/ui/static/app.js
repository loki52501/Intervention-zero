/* Intervention Zero - live stage. Vanilla JS, no deps. Consumes /events SSE. */
(function () {
  'use strict';

  // ---------- helpers ----------
  const $ = (id) => document.getElementById(id);
  const setText = (id, t) => { const e = $(id); if (e) e.textContent = t; };
  const cssVar = (name, fb) => {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fb;
    } catch (e) { return fb; }
  };
  const C = () => ({
    zoneFill: cssVar('--zone-fill', '#ffffff'),
    zoneStroke: cssVar('--zone-stroke', '#475569'),
    corridor: cssVar('--corridor', '#d6dbe3'),
    door: cssVar('--door', '#1f2937'),
    humanOnly: cssVar('--human-only', '#dc2626'),
    ok: cssVar('--ok', '#16a34a'),
    bad: cssVar('--bad', '#dc2626'),
    warn: cssVar('--warn', '#f59e0b'),
    ink: cssVar('--ink', '#0f172a'),
    muted: cssVar('--muted', '#64748b'),
  });
  const robotColors = () => [
    cssVar('--robot-1', '#2563eb'), cssVar('--robot-2', '#9333ea'),
    cssVar('--robot-3', '#0d9488'), cssVar('--robot-4', '#ea580c'),
  ];
  const trunc = (s, n) => { s = String(s ?? ''); return s.length > n ? s.slice(0, n - 1) + '…' : s; };
  const num = (v, d) => (typeof v === 'number' && isFinite(v) ? v : (d ?? 0));
  const fix2 = (v) => num(v).toFixed(2);
  const basename = (p) => String(p || '').split(/[\\/]/).pop();
  const now = () => performance.now();
  const clear = (el) => { while (el && el.firstChild) el.removeChild(el.firstChild); };
  function mk(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function restartClass(el, cls) {
    if (!el) return;
    el.classList.remove(cls);
    void el.offsetWidth;
    el.classList.add(cls);
  }

  // ---------- floor state ----------
  const floor = { site: null, siteName: '', zones: {}, edges: [], robots: new Map(), effects: [], routes: new Map(), view: null };
  const canvas = $('floor');
  const ctx = canvas && canvas.getContext ? canvas.getContext('2d') : null;
  let cw = 0, ch = 0, dpr = 1;
  let hatch = null;

  function resize() {
    if (!canvas) return;
    const wrap = $('floor-wrap') || canvas.parentElement;
    const r = wrap.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    cw = Math.max(200, Math.floor(r.width));
    ch = Math.max(150, Math.floor(r.height || r.width * 0.6));
    canvas.width = Math.floor(cw * dpr);
    canvas.height = Math.floor(ch * dpr);
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';
    computeView();
  }
  if (typeof ResizeObserver !== 'undefined' && $('floor-wrap')) new ResizeObserver(resize).observe($('floor-wrap'));
  window.addEventListener('resize', resize);

  function makeHatch(color) {
    const p = document.createElement('canvas');
    p.width = p.height = 12;
    const g = p.getContext('2d');
    g.fillStyle = 'rgba(255,255,255,1)'; g.fillRect(0, 0, 12, 12);
    g.strokeStyle = color; g.globalAlpha = 0.55; g.lineWidth = 2;
    g.beginPath(); g.moveTo(-2, 14); g.lineTo(14, -2); g.moveTo(-2, 2); g.lineTo(2, -2); g.moveTo(10, 14); g.lineTo(14, 10); g.stroke();
    return ctx.createPattern(p, 'repeat');
  }

  function computeView() {
    const ids = Object.keys(floor.zones);
    if (!ids.length || !cw) { floor.view = null; return; }
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    ids.forEach((id) => {
      const [x, y] = floor.zones[id].xy;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    });
    const padX = 150, padY = 80; // px room for zone boxes
    const w = Math.max(x1 - x0, 1), h = Math.max(y1 - y0, 1);
    const s = Math.min((cw - 2 * padX) / w, (ch - 2 * padY) / h);
    const scale = Math.max(4, s);
    const ox = (cw - w * scale) / 2 - x0 * scale;
    const oy = (ch - h * scale) / 2 - y0 * scale;
    floor.view = { scale, ox, oy };
  }
  const toPx = (x, y) => {
    const v = floor.view || { scale: 20, ox: 0, oy: 0 };
    return [x * v.scale + v.ox, y * v.scale + v.oy];
  };

  function robotPx(r) {
    const [x, y] = toPx(r.x, r.y);
    return [x + (r.dx || 0), y + (r.dy || 0)];
  }
  function zoneXY(id) { const z = floor.zones[id]; return z ? z.xy : null; }
  function getRobot(id, xy) {
    let r = floor.robots.get(id);
    if (!r) {
      const cols = robotColors();
      const p = xy || [0, 0];
      r = { id, x: p[0], y: p[1], tx: p[0], ty: p[1], color: cols[floor.robots.size % cols.length], status: '' };
      floor.robots.set(id, r);
    }
    return r;
  }
  function addEffect(robotId, type, extra) {
    const dur = { warn: 3000, ok: 2000, pulse: 2200, tip: 4000 }[type] || 2000;
    floor.effects.push(Object.assign({ robotId, type, t0: now(), dur }, extra || {}));
  }

  // ---------- drawing ----------
  function rrect(g, x, y, w, h, r) {
    g.beginPath();
    g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }

  function drawFloor(t) {
    if (!ctx) return;
    const c = C();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    if (!floor.view) {
      ctx.fillStyle = c.muted; ctx.font = '16px system-ui, sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('waiting for site…', cw / 2, ch / 2);
      return;
    }
    if (!hatch) hatch = makeHatch(c.humanOnly);
    const sc = Math.min(1.4, Math.max(0.7, floor.view.scale / 40));
    const bz = Math.min(1.25, sc + 0.1), bw = 110 * bz, bh = 56 * bz;

    // corridors
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    floor.edges.forEach((e) => {
      const a = zoneXY(e.a), b = zoneXY(e.b);
      if (!a || !b) return;
      const [ax, ay] = toPx(a[0], a[1]), [bx, by] = toPx(b[0], b[1]);
      ctx.strokeStyle = c.corridor; ctx.lineWidth = 18 * sc;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    });
    // routes (faint dashed)
    floor.routes.forEach((route, rid) => {
      const r = floor.robots.get(rid);
      ctx.strokeStyle = r ? r.color : c.muted; ctx.globalAlpha = 0.35; ctx.lineWidth = 2; ctx.setLineDash([6, 6]);
      ctx.beginPath(); let started = false;
      route.forEach((zid) => {
        const p = zoneXY(zid); if (!p) return;
        const [x, y] = toPx(p[0], p[1]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    });
    // zones
    Object.keys(floor.zones).forEach((id) => {
      const z = floor.zones[id];
      const [x, y] = toPx(z.xy[0], z.xy[1]);
      const human = z.kind === 'human_only';
      rrect(ctx, x - bw / 2, y - bh / 2, bw, bh, 10);
      ctx.fillStyle = human && hatch ? hatch : c.zoneFill; ctx.fill();
      ctx.lineWidth = human ? 2.5 : 1.8;
      ctx.strokeStyle = human ? c.humanOnly : c.zoneStroke; ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = human ? c.humanOnly : c.ink;
      ctx.font = 'bold ' + Math.round(14 * Math.max(0.9, sc)) + 'px system-ui, sans-serif';
      ctx.fillText(id, x, y - 6);
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillStyle = human ? c.humanOnly : c.muted;
      ctx.fillText(human ? 'HUMAN ONLY' : (z.kind || ''), x, y + 12);
      if (z.kind === 'elevator') {
        ctx.fillStyle = c.zoneStroke; ctx.fillRect(x + bw / 2 - 34, y - bh / 2 - 8, 34, 14);
        ctx.fillStyle = '#fff'; ctx.font = 'bold 9px system-ui, sans-serif'; ctx.fillText('ELEV', x + bw / 2 - 17, y - bh / 2 - 1);
      }
    });
    // doors
    floor.edges.forEach((e) => {
      if (!e.door_id) return;
      const a = zoneXY(e.a), b = zoneXY(e.b);
      if (!a || !b) return;
      const [ax, ay] = toPx(a[0], a[1]), [bx, by] = toPx(b[0], b[1]);
      const mx = (ax + bx) / 2, my = (ay + by) / 2;
      const ang = Math.atan2(by - ay, bx - ax) + Math.PI / 2;
      const fire = /fire/i.test(e.door_type || '') || /fire/i.test(e.door_id);
      const col = fire ? c.bad : c.door;
      ctx.strokeStyle = col; ctx.lineWidth = 5; ctx.lineCap = 'butt';
      ctx.beginPath();
      ctx.moveTo(mx - Math.cos(ang) * 13, my - Math.sin(ang) * 13);
      ctx.lineTo(mx + Math.cos(ang) * 13, my + Math.sin(ang) * 13); ctx.stroke();
      ctx.fillStyle = col; ctx.font = 'bold 11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(e.door_id, mx, my - 14);
      ctx.lineCap = 'round';
    });
    // robots
    // robots sit on the top edge of their zone box (staggered) so zone labels stay readable
    let ri = 0;
    floor.robots.forEach((r) => { r.dx = ((ri++ % 3) - 1) * 30; r.dy = -bh / 2 - 8; });
    floor.robots.forEach((r) => {
      r.x += (r.tx - r.x) * 0.08; r.y += (r.ty - r.y) * 0.08;
      const [x, y] = robotPx(r);
      ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI * 2);
      ctx.fillStyle = r.color; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = '#fff'; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 9px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(r.id, x, y + 0.5);
    });
    // effects
    const tn = now();
    floor.effects = floor.effects.filter((f) => tn - f.t0 < f.dur);
    floor.effects.forEach((f) => {
      const r = floor.robots.get(f.robotId); if (!r) return;
      const [x, y] = robotPx(r);
      const p = (tn - f.t0) / f.dur;
      ctx.globalAlpha = Math.max(0, 1 - Math.pow(p, 3));
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      if (f.type === 'warn') {
        badge(x, y - 34, c.warn, '!', f.text, c);
      } else if (f.type === 'ok') {
        badge(x, y - 34 - p * 6, c.ok, '✓', '', c);
      } else if (f.type === 'pulse') {
        for (let i = 0; i < 3; i++) {
          const q = p * 3 - i; if (q < 0 || q > 1) continue;
          ctx.globalAlpha = 1 - q; ctx.strokeStyle = c.bad; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.arc(x, y, 14 + q * 38, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.fillStyle = c.bad; rrect(ctx, x - 40, y - 52, 80, 20, 6); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = 'bold 12px system-ui, sans-serif'; ctx.fillText('TAKEOVER', x, y - 42);
      } else if (f.type === 'tip') {
        ctx.font = '12px system-ui, sans-serif';
        const tw = Math.min(260, ctx.measureText(f.text).width + 16);
        ctx.fillStyle = '#fff'; ctx.strokeStyle = c.bad; ctx.lineWidth = 1.5;
        const bx = Math.max(4, Math.min(cw - tw - 4, x - tw / 2));
        rrect(ctx, bx, y + 36, tw, 24, 8); ctx.fill(); ctx.stroke();
        ctx.fillStyle = c.ink; ctx.textAlign = 'left'; ctx.fillText(f.text, bx + 8, y + 49);
      }
      ctx.globalAlpha = 1;
    });
  }
  function badge(x, y, col, glyph, text, c) {
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 14px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(glyph, x, y + 1);
    if (text) {
      ctx.font = '11px system-ui, sans-serif';
      const w = ctx.measureText(text).width + 12;
      ctx.fillStyle = '#fff'; ctx.strokeStyle = col; ctx.lineWidth = 1.2;
      rrect(ctx, x + 14, y - 9, w, 18, 6); ctx.fill(); ctx.stroke();
      ctx.fillStyle = c.ink; ctx.textAlign = 'left'; ctx.fillText(text, x + 20, y + 1);
    }
  }

  function loop(t) {
    try { drawFloor(t); } catch (e) { console.warn('draw', e); }
    requestAnimationFrame(loop);
  }

  // ---------- ticker ----------
  const ATLAS = ['vector_search', 'atlas_search', 'rank_fusion', 'rerank', 'transaction', 'change_stream', 'validator', 'time_series'];
  const lastTicker = {};
  function familyOf(f) {
    if (ATLAS.indexOf(f) >= 0) return 'atlas';
    if (/^raindrop_/.test(f)) return 'raindrop';
    if (f === 'mujoco') return 'physics';
    return 'other';
  }
  function addPill(family, feature, text) {
    const track = $('ticker-track'); if (!track) return;
    const pill = mk('span', 'pill ' + family + ' enter');
    pill.appendChild(mk('b', 'pill-feature', String(feature).toUpperCase().replace(/_/g, ' ')));
    pill.appendChild(mk('span', 'pill-text', trunc(text, 60)));
    track.insertBefore(pill, track.firstChild);
    while (track.children.length > 30) track.removeChild(track.lastChild);
    lastTicker[feature] = now();
  }
  const recentTicker = (f, ms) => lastTicker[f] != null && now() - lastTicker[f] < ms;

  // ---------- video ----------
  let clipTimer = null;
  function hideClip() {
    const ov = $('clip-overlay'), v = $('clip-video');
    if (clipTimer) { clearTimeout(clipTimer); clipTimer = null; }
    if (ov) ov.classList.add('hidden');
    if (v) { try { v.pause(); } catch (e) { /* ignore */ } }
  }
  function showClip(clip, label) {
    const ov = $('clip-overlay'), v = $('clip-video');
    if (!ov || !v) return;
    setText('clip-caption', label);
    v.src = '/clips/' + basename(clip);
    ov.classList.remove('hidden');
    const pr = v.play(); if (pr && pr.catch) pr.catch(() => {});
    if (clipTimer) clearTimeout(clipTimer);
    clipTimer = setTimeout(hideClip, 12000);
  }
  (function () {
    const v = $('clip-video');
    if (v) { v.addEventListener('ended', hideClip); v.addEventListener('error', hideClip); }
  })();

  // ---------- gate ----------
  const BAR_IDS = ['bar-train', 'bar-heldout', 'bar-regression', 'bar-safety'];
  let gateTimers = [];
  function clearGateTimers() { gateTimers.forEach(clearTimeout); gateTimers = []; }
  function setBar(id, before, after, nums, bad) {
    const row = $(id); if (!row) return;
    const b = row.querySelector('.bar-fill.before'), a = row.querySelector('.bar-fill.after'), n = row.querySelector('.bar-nums');
    const pct = (x) => Math.max(0, Math.min(100, x * 100)) + '%';
    if (b) b.style.width = pct(before);
    if (a) a.style.width = pct(after);
    if (n) n.textContent = nums;
    row.classList.toggle('bad', !!bad);
  }
  function resetGate(candidate) {
    clearGateTimers();
    BAR_IDS.forEach((id) => setBar(id, 0, 0, '', false));
    const st = $('gate-stamp');
    if (st) { st.classList.remove('show', 'promote', 'reject', 'quarantine', 'refused'); st.textContent = ''; }
    setText('gate-candidate', candidate || '');
    setText('gate-reason', '');
  }
  function frac(a, n) { return n > 0 ? num(a) / n : 0; }
  function stamp(text, cls) {
    const st = $('gate-stamp'); if (!st) return;
    st.classList.remove('promote', 'reject', 'quarantine', 'refused', 'show');
    void st.offsetWidth;
    st.textContent = text;
    st.classList.add(cls);
    void st.offsetWidth;
    st.classList.add('show');
  }

  // ---------- spark ----------
  const spark = [];
  function drawSpark() {
    const cv = $('tfmr-spark'); if (!cv || !cv.getContext) return;
    const g = cv.getContext('2d');
    const d = window.devicePixelRatio || 1;
    const w = cv.clientWidth || cv.width, h = cv.clientHeight || cv.height;
    if (cv.width !== Math.floor(w * d)) { cv.width = Math.floor(w * d); cv.height = Math.floor(h * d); }
    g.setTransform(d, 0, 0, d, 0, 0);
    g.clearRect(0, 0, w, h);
    if (!spark.length) return;
    const c = C();
    const max = Math.max(0.05, Math.max.apply(null, spark)), pad = 5;
    const X = (i) => pad + (spark.length > 1 ? (i / (spark.length - 1)) * (w - 2 * pad) : (w - 2 * pad) / 2);
    const Y = (v) => h - pad - (v / max) * (h - 2 * pad);
    g.beginPath();
    spark.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
    g.lineTo(X(spark.length - 1), h - pad); g.lineTo(X(0), h - pad); g.closePath();
    g.fillStyle = 'rgba(37,99,235,0.12)'; g.fill();
    g.beginPath();
    spark.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
    g.strokeStyle = cssVar('--robot-1', '#2563eb'); g.lineWidth = 2; g.lineJoin = 'round'; g.stroke();
    const li = spark.length - 1;
    g.beginPath(); g.arc(X(li), Y(spark[li]), 3.5, 0, Math.PI * 2); g.fillStyle = c.ink; g.fill();
  }

  // ---------- banner ----------
  let driftTimer = null;
  function hideDriftSoon(ms) {
    if (driftTimer) clearTimeout(driftTimer);
    driftTimer = setTimeout(() => { const b = $('drift-banner'); if (b) b.classList.add('hidden'); }, ms);
  }

  // ---------- handlers ----------
  const seenDrift = { on: false };
  const handlers = {
    site_loaded(p) {
      floor.site = p.site;
      const s = String(p.site ?? '');
      floor.siteName = s === 'A_reno' ? 'Site A (renovated)' : 'Site ' + s;
      setText('site-name', floor.siteName);
      floor.zones = {};
      (p.zones || []).forEach((z) => { if (z && z.id != null && Array.isArray(z.xy)) floor.zones[z.id] = { id: z.id, kind: z.kind, xy: z.xy }; });
      floor.edges = (p.edges || []).filter((e) => e);
      floor.robots.forEach((r) => { r.tx = r.x; r.ty = r.y; });
      floor.routes.clear();
      computeView();
    },
    mission_start(p) {
      const first = zoneXY((p.route || [])[0]);
      const r = getRobot(p.robot_id, first);
      if (first && r.x === 0 && r.y === 0 && r.tx === 0 && r.ty === 0) { r.x = r.tx = first[0]; r.y = r.ty = first[1]; }
      floor.routes.set(p.robot_id, p.route || []);
    },
    robot_move(p) {
      const xy = Array.isArray(p.xy) ? p.xy : zoneXY(p.zone);
      if (!xy) return;
      const fresh = !floor.robots.has(p.robot_id);
      const r = getRobot(p.robot_id, xy);
      r.tx = xy[0]; r.ty = xy[1];
      if (fresh) { r.x = xy[0]; r.y = xy[1]; }
    },
    situation(p) {
      getRobot(p.robot_id, zoneXY(p.zone));
      addEffect(p.robot_id, 'warn', { text: trunc(p.type, 28) });
    },
    decision(p) {
      const r = getRobot(p.robot_id);
      r.status = 'decided';
    },
    outcome(p) {
      const s = p.status;
      getRobot(p.robot_id);
      if (s === 'success' || s === 'escalated_ok') addEffect(p.robot_id, 'ok');
      else if (s === 'takeover' || s === 'safety_violation' || s === 'escalation_missed') addEffect(p.robot_id, 'pulse');
      if (p.clip) showClip(p.clip, (p.judged_by === 'mujoco' ? 'MuJoCo judge' : 'Rule judge') + ': ' + (p.reason ?? ''));
      if (p.judged_by === 'mujoco') addPill('physics', 'mujoco', p.reason ?? '');
    },
    takeover(p) {
      getRobot(p.robot_id);
      addEffect(p.robot_id, 'pulse');
      if (p.operator_tip) addEffect(p.robot_id, 'tip', { text: trunc(p.operator_tip, 80) });
    },
    fix_proposed(p) {
      const empty = $('fix-empty'); if (empty) empty.classList.add('hidden');
      const meta = $('fix-meta');
      if (meta) {
        clear(meta);
        meta.appendChild(document.createTextNode(String(p.candidate_id ?? '') + ' '));
        const au = String(p.author ?? 'fixer');
        meta.appendChild(mk('span', 'author-' + au, au));
      }
      const box = $('fix-edits');
      if (box) {
        clear(box);
        (p.edits || []).forEach((e) => {
          const d = mk('div', 'edit');
          const t = e.text_or_change;
          d.appendChild(mk('div', 'diff-line add', '+ ' + (typeof t === 'string' ? t : JSON.stringify(t ?? ''))));
          const sc = e.scope || {};
          const parts = [];
          if (sc.site != null) parts.push('site=' + sc.site);
          if (sc.situation_type != null) parts.push('situation=' + sc.situation_type);
          if (sc.zone_class != null) parts.push('zone=' + sc.zone_class);
          if (sc.door_id != null) parts.push('door=' + sc.door_id);
          d.appendChild(mk('div', 'diff-meta', (e.kind ?? 'edit') + ' · ' + (parts.length ? parts.join(' ') : 'global')));
          if (e.rationale) d.appendChild(mk('div', 'rationale', e.rationale));
          const pr = e.prediction || {};
          d.appendChild(mk('div', 'prediction', 'predicts: fixes ' + (pr.train_fixed ?? '?') + ' train / ' + (pr.heldout_fixed ?? '?') + ' held-out'));
          box.appendChild(d);
        });
      }
      resetGate(p.candidate_id);
    },
    validator_refused(p) {
      resetGate(p.candidate_id);
      stamp('DATABASE REFUSED', 'refused');
      setText('gate-reason', p.message ?? '');
      if (!recentTicker('validator', 1500)) addPill('atlas', 'validator', 'refused ' + (p.candidate_id ?? ''));
    },
    gate_result(p) {
      clearGateTimers();
      setText('gate-candidate', p.candidate_id ?? '');
      BAR_IDS.forEach((id) => setBar(id, 0, 0, '', false));
      const tr = p.train || {}, ho = p.heldout || {}, rg = p.regression || {};
      const sv = num(p.safety_violations);
      const steps = [
        () => setBar('bar-train', frac(tr.before, tr.n), frac(tr.after, tr.n), num(tr.before) + ' → ' + num(tr.after) + ' / ' + num(tr.n), false),
        () => setBar('bar-heldout', frac(ho.before, ho.n), frac(ho.after, ho.n), num(ho.before) + ' → ' + num(ho.after) + ' / ' + num(ho.n), false),
        () => setBar('bar-regression', 0, frac(rg.passed, rg.n), num(rg.passed) + ' / ' + num(rg.n) + ' passed', num(rg.passed) < num(rg.n)),
        () => setBar('bar-safety', 0, 1, sv + (sv === 1 ? ' violation' : ' violations'), sv > 0),
      ];
      steps.forEach((fn, i) => gateTimers.push(setTimeout(() => { try { fn(); } catch (e) { console.warn(e); } }, 60 + i * 250)));
      gateTimers.push(setTimeout(() => {
        const v = String(p.verdict ?? '').toUpperCase();
        if (v === 'PROMOTE') stamp('PROMOTED', 'promote');
        else if (v === 'QUARANTINE') stamp('QUARANTINE', 'quarantine');
        else stamp('REJECTED', 'reject');
        setText('gate-reason', p.reason ?? '');
      }, 1200));
    },
    promoted(p) {
      const el = $('policy-version');
      if (el) { el.textContent = 'policy v' + p.version; restartClass(el, 'bump'); setTimeout(() => el.classList.remove('bump'), 1200); }
      if (!recentTicker('transaction', 1500)) addPill('atlas', 'transaction', 'promoted v' + p.version + ' (txn)');
      if (seenDrift.on) hideDriftSoon(4000);
    },
    reload(p) {
      addPill('atlas', 'change_stream', 'hot-reloaded ' + (p.robots ?? '?') + ' robots → v' + (p.version ?? '?'));
    },
    drift_alarm(p) {
      const b = $('drift-banner'); if (!b) return;
      seenDrift.on = true;
      b.classList.remove('hidden');
      restartClass(b, 'flash');
      setText('drift-title', 'DRIFT DETECTED — Site ' + (p.site ?? '') + ' renovated');
      setText('drift-detail', 'window TFMR ' + fix2(p.window_tfmr) + ' vs baseline ' + fix2(p.baseline_tfmr) +
        ' · p = ' + (p.p_value ?? '?') + ' · ' + (p.missions_since_change ?? '?') + ' missions since change');
      hideDriftSoon(20000);
    },
    coach(p) {
      setText('coach-version', 'playbook v' + (p.parent ?? '?') + ' → v' + (p.playbook_version ?? '?'));
      const v = $('coach-verdict');
      if (v) {
        v.classList.remove('promote', 'reject', 'quarantine');
        v.textContent = String(p.verdict ?? '');
        const k = String(p.verdict ?? '').toLowerCase();
        if (k) v.classList.add(k);
      }
      const ul = $('coach-changes');
      if (ul) { clear(ul); (p.changes || []).forEach((c) => ul.appendChild(mk('li', null, String(c)))); }
      setText('coach-reason', p.reason ?? '');
    },
    metrics(p) {
      const t = p.tfmr;
      if (typeof t === 'number') setText('tfmr-value', t.toFixed(2));
      setText('day-shift', 'day ' + (p.day ?? '?') + (p.shift ? ' · ' + p.shift + ' shift' : ''));
      if (p.policy_version != null) setText('policy-version', 'policy v' + p.policy_version);
      const v = p.window_tfmr ?? p.tfmr;
      if (typeof v === 'number') { spark.push(v); if (spark.length > 120) spark.shift(); drawSpark(); }
    },
    ticker(p) {
      const f = String(p.feature ?? 'other');
      addPill(familyOf(f), f, p.text ?? '');
    },
  };

  // ---------- SSE ----------
  const seen = new Set();
  function onMessage(e) {
    let ev;
    try { ev = JSON.parse(e.data); } catch (err) { return; }
    if (!ev || typeof ev !== 'object') return;
    if (ev.id != null) { if (seen.has(ev.id)) return; seen.add(ev.id); }
    const p = ev.payload || {};
    if (p.synthetic === true) { const b = $('synthetic-badge'); if (b) b.classList.remove('hidden'); }
    if (ev.run_id) setText('run-id', ev.run_id);
    const h = handlers[ev.kind];
    if (!h) return;
    try { h(p, ev); } catch (err) { console.warn('handler ' + ev.kind, err); }
  }

  function connect() {
    if (typeof EventSource === 'undefined') return;
    const es = new EventSource('/events' + location.search);
    const cs = $('conn-status');
    es.onopen = () => { if (cs) { cs.classList.add('live'); cs.classList.remove('down'); } };
    es.onerror = () => { if (cs) { cs.classList.add('down'); cs.classList.remove('live'); } };
    es.onmessage = onMessage;
  }

  // ---------- boot ----------
  resize();
  window.addEventListener('resize', drawSpark);
  requestAnimationFrame(loop);
  connect();
})();
