(function () {
  'use strict';
  var EXAMPLES = [
    'a pharmacy with a wet floor near aisle 2, a heavy cart and rush-hour shoppers',
    'a small grocery store with two narrow aisles and a spill by the checkout',
    'a warehouse aisle with a heavy cart and fast workers'
  ];
  var $ = function (id) { return document.getElementById(id); };
  var slug = null, es = null, gens = {}, fake = /[?&]fake=1/.test(location.search);

  function esc(s) { var d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; }
  function ticker(msg, busy) { $('ticker').textContent = msg; document.querySelector('footer').classList.toggle('busy', !!busy); }
  function status(msg, err) { var s = $('status'); s.textContent = msg || ''; s.className = 'status' + (err ? ' err' : ''); }

  EXAMPLES.forEach(function (t) {
    var b = document.createElement('button');
    b.className = 'chip'; b.type = 'button'; b.textContent = t;
    b.onclick = function () { $('prompt').value = t; };
    $('examples').appendChild(b);
  });

  function wilson(k, n) {
    if (!n) return [0, 0];
    var z = 1.96, p = k / n, d = 1 + z * z / n;
    var c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.max(0, c - h), Math.min(1, c + h)];
  }

  function renderSpec(spec, repairs) {
    var sh = spec.shoppers || {}, cart = spec.cart || {};
    var kv = [
      ['venue', spec.venue], ['size', (spec.size_m || []).join(' x ') + ' m'], ['aisles', spec.aisles],
      ['cart mass', (cart.mass_kg != null ? cart.mass_kg : '?') + ' kg'],
      ['shoppers', (sh.count != null ? sh.count : 0) + ' (' + (sh.density || '-') + ')'],
      ['spills', (spec.spills || []).length], ['target product', spec.target_product || '-']
    ];
    $('spec').className = 'kv';
    $('spec').innerHTML = kv.map(function (p) { return '<div class="k"><b>' + esc(p[0]) + '</b>' + esc(p[1]) + '</div>'; }).join('');
    $('repairs').innerHTML = (repairs && repairs.length ? repairs : ['no repairs needed']).map(function (r) { return '<li>' + esc(r) + '</li>'; }).join('');
  }

  function renderWorld(d) {
    renderSpec(d.spec, d.repairs);
    var c = d.counts || {};
    $('counts').textContent = 'train ' + c.train + ' / held-out ' + c.heldout + ' / safety ' + c.safety + ' valid worlds';
    if (d.preview_url) {
      $('preview').src = d.preview_url + '?t=' + Date.now(); $('preview').hidden = false; $('previewEmpty').hidden = true;
    } else { $('preview').hidden = true; $('previewEmpty').hidden = false; }
  }

  function renderRows() {
    var keys = Object.keys(gens).map(Number).sort(function (a, b) { return a - b; });
    if (!keys.length) return;
    $('rows').innerHTML = keys.map(function (k) {
      var g = gens[k], ci = wilson(g.heldout_success, g.heldout_n);
      var ed = Object.keys(g.edits || {}).map(function (e) { return e + ' = ' + g.edits[e]; }).join(', ') || (k === 0 ? 'naive defaults' : 'none');
      return '<tr><td><b>' + k + '</b></td><td class="edits">' + esc(ed) + '</td><td><span class="stamp ' + esc(g.verdict) + '">' + esc(g.verdict) + '</span></td>' +
        '<td><b>' + g.heldout_success + '/' + g.heldout_n + '</b> <span class="ci">[' + Math.round(ci[0] * 100) + '-' + Math.round(ci[1] * 100) + '%]</span></td>' +
        '<td>' + (g.safety_violations || 0) + '</td></tr>';
    }).join('');
    $('chart').innerHTML = keys.map(function (k) {
      var g = gens[k], pct = g.heldout_n ? g.heldout_success / g.heldout_n : 0;
      return '<div class="bar"><span>' + Math.round(pct * 100) + '%</span><i style="height:' + Math.max(2, pct * 100) + '%"></i><span>g' + k + '</span></div>';
    }).join('');
    var latest = gens[keys[keys.length - 1]];
    if (latest && latest.rationale) {
      $('rationale').className = ''; $('rationale').textContent = latest.rationale; $('ratGen').textContent = '(generation ' + latest.gen + ')';
    }
  }

  function loadClips() {
    if (!slug) return;
    fetch('/api/studio/clips/' + slug).then(function (r) { return r.json(); }).then(function (d) {
      var urls = (d.clips || []).map(function (c) { return c.url; });
      if (!urls.length) return;
      $('videoNote').textContent = 'Playing ' + (d.clips.map(function (c) { return c.name; }).join(' then '));
      var v = $('video'), i = 0;
      v.src = urls[0];
      v.onended = function () { i++; if (i < urls.length) { v.src = urls[i]; v.play(); } };
    }).catch(function () {});
  }

  function listen() {
    if (es) es.close();
    es = new EventSource('/api/studio/events/' + slug);
    function on(kind, fn) { es.addEventListener(kind, function (m) { try { fn(JSON.parse(m.data).payload || {}); } catch (e) {} }); }
    on('ticker', function (p) { ticker(p.msg, true); });
    on('metrics', function (p) {
      if (p.gen === 0 && !gens[0]) { /* baseline row filled by the metrics event */
        gens[0] = { gen: 0, verdict: 'BASELINE', edits: {}, heldout_success: Math.round((p.heldout_rate || 0) * (window._hn || 0)), heldout_n: window._hn || 0, safety_violations: 0 };
        renderRows();
      }
    });
    on('generation', function (p) { gens[p.gen] = p; renderRows(); ticker('generation ' + p.gen + ' ' + p.verdict, true); });
    on('done', function () { es.close(); ticker('Learning finished.', false); $('btnLearn').disabled = false; status('Learning finished.'); loadClips(); });
    on('error', function (p) { es.close(); status(p.message || 'error', true); ticker('Error.', false); $('btnLearn').disabled = false; });
  }

  $('btnWorld').onclick = function () {
    var prompt = $('prompt').value.trim();
    if (!prompt) { status('Type or pick a prompt first.', true); return; }
    $('btnWorld').disabled = true; $('btnLearn').disabled = true; status('LLM is writing the world spec...'); ticker('LLM spec -> validator -> world build', true);
    fetch('/api/studio/world' + (fake ? '?fake=1' : ''), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt: prompt }) })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.detail || r.status); return j; }); })
      .then(function (d) {
        slug = d.slug; window._hn = d.counts.heldout; gens = {};
        renderWorld(d); status('World ready. Press Learn.'); ticker('World built and validated.', false);
        $('btnLearn').disabled = false;
        $('rows').innerHTML = '<tr class="empty-row"><td colspan="5">Press Learn.</td></tr>'; $('chart').innerHTML = '';
        loadClips();
      })
      .catch(function (e) { status('Failed: ' + e.message, true); ticker('Error.', false); })
      .then(function () { $('btnWorld').disabled = false; });
  };

  $('btnLearn').onclick = function () {
    if (!slug) return;
    $('btnLearn').disabled = true; gens = {}; status('Learning...');
    listen();
    fetch('/api/studio/learn', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ slug: slug, generations: +$('gens').value || 3, fake: fake }) })
      .then(function (r) { return r.json(); })
      .then(function (d) { if (!d.started) status(d.reason || 'not started', true); })
      .catch(function (e) { status('Failed: ' + e.message, true); $('btnLearn').disabled = false; });
  };
})();
