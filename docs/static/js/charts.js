/*
 * Resonant Brane Splatting project page: results (3/3).
 * The "u explorer", the cost chart, the quality-vs-cost scatter, the alternative-primitives chart and the Set5 table.
 * Every number comes from window.RBS_DATA, which was generated from the paper's LaTeX tables.
 */
(function () {
  'use strict';
  var U = window.RBSUI, R = window.RBS, D = window.RBS_DATA;
  if (!U || !D) return;
  var $ = U.$, $$ = U.$$, svgEl = U.svgEl, bindSeg = U.bindSeg, fmt = U.fmt;
  var TAU = Math.PI * 2;

  var C = { pink: '#FF4D9D', violet: '#C77DFF', amber: '#FFC247', cyan: '#2FE1FF', lime: '#7CF29A', mute: '#8E8CC8', gray: '#6E6D95', ink: '#ECEBFA', ink2: '#A5A4C4' };
  var STYLE = {
    'RBS u=2': { color: C.pink, w: 3.8, label: 'RBS u=2', z: 7 },
    'RBS u=4': { color: C.violet, w: 2.8, label: 'RBS u=4', z: 6 },
    'RBS u=1': { color: C.amber, w: 2.8, label: 'RBS u=1', z: 5 },
    'GSASR': { color: C.cyan, w: 3.2, label: 'GSASR', z: 4 },
    'GRAPE': { color: C.lime, w: 2.2, label: 'GRAPE', z: 3 },
    'GaussianSR': { color: C.mute, w: 1.8, label: 'GaussianSR', z: 2 },
    'ContinuousSR*': { color: C.mute, w: 1.8, dash: '5 4', label: 'ContinuousSR*', z: 2 },
    'CiaoSR': { color: C.gray, w: 1.5, label: 'CiaoSR', z: 1 }
  };
  function styleOf(name) { return STYLE[name] || { color: C.gray, w: 1.3, label: null, z: 0 }; }
  var NAMES = Object.keys(D.cost).sort(function (a, b) { return styleOf(a).z - styleOf(b).z; });

  /* ---------- shared chart helpers ---------- */
  function chartWidth(host) { return Math.max(320, Math.min(900, Math.round(host.clientWidth || 880))); }
  function redrawOnResize(host, draw) {
    var last = chartWidth(host), t = 0;
    function check() { var w = chartWidth(host); if (Math.abs(w - last) > 8) { last = w; draw(); } }
    if ('ResizeObserver' in window) new ResizeObserver(function () { clearTimeout(t); t = setTimeout(check, 120); }).observe(host);
  }
  function tooltip(host) {
    var t = document.createElement('div'); t.className = 'tooltip'; t.hidden = true; host.appendChild(t);
    return {
      show: function (html, ev) { var r = host.getBoundingClientRect(); t.innerHTML = html; t.hidden = false; t.style.left = Math.max(70, Math.min(r.width - 70, ev.clientX - r.left)) + 'px'; t.style.top = (ev.clientY - r.top) + 'px'; },
      hide: function () { t.hidden = true; }
    };
  }
  function spread(items, minGap, lo, hi) { // push label y-positions apart
    items.sort(function (a, b) { return a.y - b.y; });
    for (var i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < minGap) items[i].y = items[i - 1].y + minGap;
    var over = items.length ? items[items.length - 1].y - hi : 0;
    if (over > 0) for (var j = items.length - 1; j >= 0; j--) { items[j].y -= over; if (j > 0 && items[j].y - items[j - 1].y < minGap) { items[j - 1].y = items[j].y - minGap; over = 0; } }
    items.forEach(function (it) { if (it.y < lo) it.y = lo; });
  }
  function niceStep(raw) { var p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; }
  function logTicks(lo, hi) { var out = []; for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) [1, 2, 5].forEach(function (k) { var v = k * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }); return out; }
  function fmtTick(v) { return v >= 1000 ? (v / 1000) + 'k' : String(v); }
  function fmtMem(m) { return m >= 1000 ? (m / 1000).toFixed(1) + ' GB' : Math.round(m) + ' MB'; }

  /* ======================================================================
   * u explorer: a smaller u means fewer primitives, so each Brane gets more complex as the scale grows.
   * Schematic: the degrees reached at x12 (4, 7 and 12 for u = 4, 2, 1) set the trend; in between they follow log(s) / log(12).
   * Every cell holds one Brane drawn alone in its cell; its footprint is kept inside the cell, so what changes is the content.
   * ==================================================================== */
  var NMAX_AT_12 = { 1: 12, 2: 7, 4: 4 }, USCALES = [1, 2, 3, 4, 6, 8, 12], UCOL = { 1: C.amber, 2: C.pink, 4: C.violet };
  function degreeAt(u, s) { return Math.round(NMAX_AT_12[u] * Math.log(s) / Math.log(12)); }

  function initUExp() {
    var cv = $('#uCanvas'); if (!cv) return;
    var ctx = cv.getContext('2d'), S = cv.width, P = 4, T = U.tex, st = { u: 2, si: 0, playing: false, timer: 0 };
    var img = new Float32Array(S * S * 3), cache = {}, idata = ctx.createImageData(S, S);
    var BASES = [[0.95, 0.42, 0.6], [0.55, 0.5, 0.97], [0.28, 0.85, 0.95], [0.98, 0.78, 0.34], [0.5, 0.95, 0.62], [0.9, 0.6, 0.95]];
    var GROUND = [0.024, 0.024, 0.05];
    var trim = function (v) { return parseFloat(v < 10 ? v.toFixed(2) : v.toFixed(0)); };

    function build(u, D) { // pixels of the window (four by four input positions, u x u Branes each), at degree D
      var cells = P * u, cp = S / cells, tile = new Float32Array(cp * cp * 3), sg = 0.46 / Math.max(R.rmaxOf(D), 2.8), i, j, p, q, c;
      for (p = 0; p < img.length; p += 3) { img[p] = GROUND[0]; img[p + 1] = GROUND[1]; img[p + 2] = GROUND[2]; }
      for (j = 0; j < cells; j++) for (i = 0; i < cells; i++) {
        var seed = 1000 * u + j * cells + i, rng = R.makeRng(seed * 17 + 3);
        var br = { cx: 0, cy: 0, sx: sg * rng.uniform(0.9, 1.05), sy: sg * rng.uniform(0.9, 1.05), rho: rng.uniform(-0.6, 0.6), alpha: 1, D: D, coef: R.modeColors(D, seed, { base: BASES[Math.floor(rng.next() * BASES.length) % BASES.length] }) };
        R.renderBranes([br], cp, [-0.5, 0.5, -0.5, 0.5], { out: tile });
        for (var y = 0; y < cp; y++) for (var x = 0; x < cp; x++) {
          p = ((j * cp + y) * S + (i * cp + x)) * 3; q = (y * cp + x) * 3;
          for (c = 0; c < 3; c++) { var v = img[p + c] + tile[q + c]; img[p + c] = v > 1 ? 1 : v; }
        }
      }
      R.toRGBA(img, S, null, idata.data);
      return new ImageData(new Uint8ClampedArray(idata.data), S, S);
    }

    function draw() {
      var u = st.u, s = USCALES[st.si], D = degreeAt(u, s), key = u + ':' + D;
      if (!cache[key]) cache[key] = build(u, D);
      ctx.putImageData(cache[key], 0, 0);
      var cells = P * u, cp = S / cells; ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(190,190,255,.2)'; ctx.beginPath();
      for (var i = 1; i < cells; i++) { ctx.moveTo(i * cp + 0.5, 0); ctx.lineTo(i * cp + 0.5, S); ctx.moveTo(0, i * cp + 0.5); ctx.lineTo(S, i * cp + 0.5); }
      ctx.stroke();
      $('#uChip').textContent = 'Each cell is one Brane, degree ' + D + (D === 0 ? ' (a Gaussian)' : '');
      stats(u, s, D); curve(u, s, D); cards(u, s);
      $$('#uScale button').forEach(function (b) { b.setAttribute('aria-pressed', +b.dataset.s === s ? 'true' : 'false'); });
    }

    function stat(title, big, sm, hot) { return '<div class="ustat' + (hot ? ' hot' : '') + '"><h5>' + title + '</h5><span class="big">' + big + '</span><span class="sm">' + sm + '</span></div>'; }
    function stats(u, s, D) {
      var L = (D + 1) * (D + 1), ppb = (s / u) * (s / u), n = P * P * u * u, g = P * P * 16;
      $('#uStats').innerHTML =
        stat('Degree', T('{N=M=' + D + '}'), L + (L === 1 ? ' mode' : ' modes') + ' per Brane, ' + (6 + 3 * L) + ' numbers', true) +
        stat('Output pixels per Brane', String(trim(ppb)), T('(s/u)^2=(' + s + '/' + u + ')^2')) +
        stat('Branes in this window', String(n), (n < g ? (g / n) + '&times; fewer than GSASR (' + g + ')' : 'as many as GSASR (' + g + ')'));
    }

    function curve(u, s, D) {
      var host = $('#uCurve'), W = 520, H = 250, M = { l: 34, r: 74, t: 16, b: 38 }, svg;
      host.innerHTML = '';
      svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Brane degree against scale for u = 1, 2 and 4: at x12 the degrees are 12, 7 and 4' }, host);
      var X = function (v) { return M.l + (W - M.l - M.r) * Math.log(v) / Math.log(12); }, Y = function (n) { return M.t + (H - M.t - M.b) * (1 - n / 12); };
      [0, 4, 8, 12].forEach(function (n) {
        svgEl('line', { x1: M.l, x2: W - M.r, y1: Y(n), y2: Y(n), 'class': 'grid' }, svg);
        svgEl('text', { x: M.l - 8, y: Y(n) + 4, 'text-anchor': 'end' }, svg).textContent = n;
      });
      USCALES.forEach(function (v) { svgEl('text', { x: X(v), y: H - M.b + 20, 'text-anchor': 'middle' }, svg).textContent = '×' + v; });
      svgEl('text', { x: M.l - 8, y: 8, 'text-anchor': 'end', 'class': 'axis-title' }, svg).textContent = 'N = M';
      svgEl('text', { x: (M.l + W - M.r) / 2, y: H - 4, 'text-anchor': 'middle', 'class': 'axis-title' }, svg).textContent = 'scale';
      svgEl('line', { x1: X(s), x2: X(s), y1: M.t, y2: H - M.b, 'class': 'cur-line' }, svg);
      [1, 2, 4].forEach(function (uu) {
        var d = USCALES.map(function (v, i) { return (i ? 'L' : 'M') + X(v).toFixed(1) + ',' + Y(degreeAt(uu, v)).toFixed(1); }).join(' ');
        svgEl('path', { d: d, 'class': 'curve' + (uu === u ? '' : ' off'), stroke: UCOL[uu], 'stroke-width': uu === u ? 4 : 2.4 }, svg);
        var yy = Y(degreeAt(uu, s));
        svgEl('circle', { cx: X(s), cy: yy, r: uu === u ? 6.5 : 4, fill: UCOL[uu], stroke: '#0A0A14', 'stroke-width': 2, opacity: uu === u ? 1 : 0.55 }, svg);
        var t = svgEl('text', { x: W - M.r + 9, y: Y(NMAX_AT_12[uu]) + 4, 'class': 'lbl', fill: UCOL[uu], opacity: uu === u ? 1 : 0.6 }, svg); t.textContent = 'u=' + uu + ': ' + NMAX_AT_12[uu];
        svgEl('circle', { cx: X(12), cy: Y(NMAX_AT_12[uu]), r: 3, fill: UCOL[uu], opacity: uu === u ? 1 : 0.5 }, svg);
      });
      var hit = svgEl('rect', { x: M.l - 6, y: M.t, width: W - M.l - M.r + 12, height: H - M.t - M.b, fill: 'transparent', style: 'cursor:pointer' }, svg);
      hit.addEventListener('click', function (ev) {
        var r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) * (W / r.width), best = 0, bd = 1e9;
        USCALES.forEach(function (v, i) { var d = Math.abs(X(v) - x); if (d < bd) { bd = d; best = i; } });
        stop(); st.si = best; draw();
      });
    }

    function card(title, big, small, vs) { return '<div class="ucard"><h5>' + title + '</h5><span class="big">' + big + (small ? ' <small>' + small + '</small>' : '') + '</span><p class="vs">' + vs + '</p></div>'; }
    function cards(u, s) {
      var host = $('#uCards');
      if (s < 2) { host.innerHTML = '<p class="none">The paper reports quality and cost from &times;2 upward. Raise the scale to see the numbers.</p>'; return; }
      var key = 'RBS u=' + u, q = D.quality[key].psnr[s], qg = D.quality.GSASR.psnr[s];
      var t = D.cost[key].time[s], tg = D.cost.GSASR.time[s], m = D.cost[key].mem[s], mg = D.cost.GSASR.mem[s];
      var dq = q - qg, speed = tg / t, dm = (m / mg - 1) * 100;
      var memBig = m >= 1000 ? (m / 1000).toFixed(1) : Math.round(m), memUnit = m >= 1000 ? 'GB' : 'MB';
      host.innerHTML =
        card('PSNR, Urban100 &times;' + s, q.toFixed(2), 'dB', 'GSASR ' + qg.toFixed(2) + ' &middot; <span class="' + (dq >= 0 ? 'good' : 'bad') + '">' + (dq >= 0 ? '+' : '−') + Math.abs(dq).toFixed(2) + ' dB</span>') +
        card('Time per image', fmt(t), 'ms', 'GSASR ' + fmt(tg) + ' ms &middot; <span class="' + (speed >= 1 ? 'good' : 'bad') + '">' + (speed >= 1 ? speed.toFixed(1) + '&times; faster' : (1 / speed).toFixed(1) + '&times; slower') + '</span>') +
        card('GPU memory', memBig, memUnit, 'GSASR ' + fmtMem(mg) + ' &middot; <span class="' + (dm <= 0 ? 'good' : 'bad') + '">' + (dm <= 0 ? '−' : '+') + Math.abs(dm).toFixed(0) + '%</span>');
    }

    function stop() { st.playing = false; clearInterval(st.timer); $('#uPlay span').textContent = 'Raise the scale'; }
    function play() {
      stop(); st.playing = true; $('#uPlay span').textContent = 'Pause';
      if (st.si >= USCALES.length - 1) { st.si = 0; draw(); }
      st.timer = setInterval(function () { if (st.si >= USCALES.length - 1) { stop(); return; } st.si++; draw(); }, 900);
    }
    $('#uPlay').addEventListener('click', function () {
      if (st.playing) { stop(); return; }
      if (U.reduceMotion) { st.si = USCALES.length - 1; draw(); return; }
      play();
    });
    bindSeg($('#uSeg'), 'u', function (v) { st.u = +v; draw(); });
    bindSeg($('#uScale'), 's', function (v) { stop(); st.si = USCALES.indexOf(+v); draw(); });
    if (U.reduceMotion) st.si = USCALES.length - 1;
    draw();
    if (!U.reduceMotion) U.onVisible(cv, function (v) { if (v && !st.started) { st.started = true; play(); } }, 0.45);
  }

  /* ======================================================================
   * Cost across scales
   * ==================================================================== */
  function initCostChart() {
    var host = $('#costChart'); if (!host) return;
    var metric = 'time', focus = null, scales = D.scales, tip = tooltip(host);
    var W = 880, H = 440, M = { l: 62, r: 150, t: 22, b: 44 };
    function layout() { W = chartWidth(host); var n = W < 600; H = n ? 340 : 440; M = { l: n ? 44 : 62, r: n ? 98 : 150, t: 22, b: 44 }; }
    function applyFocus(svg) { $$('.line, .lbl', svg).forEach(function (e) { e.classList.toggle('dim', !!focus && e.getAttribute('data-m') !== focus); }); }

    function draw() {
      layout();
      var old = $('svg', host); if (old) old.remove();
      var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Line chart of ' + (metric === 'time' ? 'inference time' : 'GPU memory') + ' versus upscaling factor for each method' }, host);
      var vals = []; NAMES.forEach(function (n) { scales.forEach(function (s) { vals.push(D.cost[n][metric][s]); }); });
      var lo = Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, vals) / 1.2))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, vals) * 1.2))), unit = metric === 'time' ? 'ms' : 'MB';
      var ly = function (v) { return M.t + (H - M.t - M.b) * (1 - (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))); };
      var lx = function (i) { return M.l + i * (W - M.l - M.r) / (scales.length - 1); };
      logTicks(lo, hi).forEach(function (v) {
        svgEl('line', { x1: M.l, x2: W - M.r, y1: ly(v), y2: ly(v), 'class': 'grid' }, svg);
        svgEl('text', { x: M.l - 10, y: ly(v) + 4, 'text-anchor': 'end' }, svg).textContent = fmtTick(v);
      });
      svgEl('text', { x: M.l - 10, y: 10, 'text-anchor': 'end', 'class': 'axis-title' }, svg).textContent = unit;
      scales.forEach(function (s, i) { svgEl('text', { x: lx(i), y: H - M.b + 24, 'text-anchor': 'middle' }, svg).textContent = '×' + s; });
      svgEl('text', { x: (M.l + W - M.r) / 2, y: H - 4, 'text-anchor': 'middle', 'class': 'axis-title' }, svg).textContent = 'upscaling factor (log scale on the vertical axis)';

      var labels = [];
      NAMES.forEach(function (name) {
        var st = styleOf(name), v = D.cost[name][metric];
        var d = scales.map(function (s, i) { return (i ? 'L' : 'M') + lx(i).toFixed(1) + ',' + ly(v[s]).toFixed(1); }).join(' ');
        var p = svgEl('path', { d: d, stroke: st.color, 'stroke-width': st.w, 'class': 'line', 'data-m': name }, svg);
        if (st.dash) p.setAttribute('stroke-dasharray', st.dash);
        var hit = svgEl('path', { d: d, 'class': 'hit', fill: 'none', stroke: 'transparent', 'stroke-width': 16, 'pointer-events': 'stroke' }, svg);
        hit.addEventListener('pointerenter', function () { focus = name; applyFocus(svg); });
        hit.addEventListener('pointerleave', function () { focus = null; applyFocus(svg); tip.hide(); });
        hit.addEventListener('pointermove', function (ev) {
          var r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) * (W / r.width);
          var i = Math.max(0, Math.min(scales.length - 1, Math.round((x - M.l) / ((W - M.l - M.r) / (scales.length - 1))))), s = scales[i];
          tip.show('<b>' + name + '</b><small>×' + s + ': ' + fmt(v[s]) + ' ' + unit + '</small>', ev);
        });
        if (st.label) labels.push({ name: name, y: ly(v[scales[scales.length - 1]]), color: st.color, text: st.label });
      });
      spread(labels, 15, M.t + 6, H - M.b);
      labels.forEach(function (l) { var t = svgEl('text', { x: W - M.r + 10, y: l.y + 4, 'class': 'lbl', 'data-m': l.name }, svg); t.textContent = l.text; t.style.fill = l.color; });
      applyFocus(svg);
    }
    bindSeg($('#costMetric'), 'metric', function (m) { metric = m; draw(); });
    draw(); redrawOnResize(host, draw);
  }

  /* ======================================================================
   * Quality against cost (Urban100)
   * ==================================================================== */
  var METRIC_INFO = {
    psnr: { label: 'PSNR (dB), higher is better', dec: 2, better: 1, tick: 1 },
    ssim: { label: 'SSIM, higher is better', dec: 4, better: 1, tick: 3 },
    lpips: { label: 'LPIPS, lower is better (axis flipped)', dec: 4, better: -1, tick: 3 },
    dists: { label: 'DISTS, lower is better (axis flipped)', dec: 4, better: -1, tick: 3 }
  };
  function initScatter() {
    var host = $('#scatter'); if (!host) return;
    var metric = 'psnr', scale = '4', W = 880, H = 500, M = { l: 66, r: 28, t: 18, b: 56 }, tip = tooltip(host);
    var PRIORITY = ['RBS u=2', 'RBS u=4', 'RBS u=1', 'GSASR', 'GRAPE', 'ContinuousSR*', 'GaussianSR', 'CiaoSR', 'LMF', 'SRNO', 'Meta-SR', 'LIIF', 'LTE', 'LINF'];
    function layout() { W = chartWidth(host); var n = W < 600; H = n ? 400 : 500; M = { l: n ? 52 : 66, r: n ? 14 : 28, t: 18, b: 56 }; }

    function draw() {
      layout();
      var old = $('svg', host); if (old) old.remove();
      var info = METRIC_INFO[metric], better = info.better;
      var pts = Object.keys(D.quality).map(function (n) { return { name: n, t: D.cost[n].time[scale], mem: D.cost[n].mem[scale], v: D.quality[n][metric][scale] }; }).filter(function (p) { return p.v != null; });
      var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Scatter plot of ' + metric.toUpperCase() + ' against inference time at x' + scale + ' on Urban100' }, host);
      var tmin = Math.min.apply(null, pts.map(function (p) { return p.t; })), tmax = Math.max.apply(null, pts.map(function (p) { return p.t; }));
      var x0 = Math.pow(10, Math.floor(Math.log10(tmin * 0.8))), x1 = Math.pow(10, Math.ceil(Math.log10(tmax * 1.25)));
      var vs = pts.map(function (p) { return p.v; }), lo = Math.min.apply(null, vs), hi = Math.max.apply(null, vs), pad = (hi - lo) * 0.14 || 0.01; lo -= pad; hi += pad;
      var X = function (t) { return M.l + (Math.log10(t) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * (W - M.l - M.r); };
      var Y = function (v) { var u = (v - lo) / (hi - lo); if (better < 0) u = 1 - u; return M.t + (H - M.t - M.b) * (1 - u); };

      var step = niceStep((hi - lo) / 5), start = Math.ceil(lo / step) * step;
      for (var v = start; v <= hi; v += step) {
        svgEl('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), 'class': 'grid' }, svg);
        svgEl('text', { x: M.l - 10, y: Y(v) + 4, 'text-anchor': 'end' }, svg).textContent = v.toFixed(info.tick);
      }
      for (var e = Math.log10(x0); e <= Math.log10(x1) + 1e-9; e++) {
        var t = Math.pow(10, e);
        svgEl('line', { x1: X(t), x2: X(t), y1: M.t, y2: H - M.b, 'class': 'grid' }, svg);
        svgEl('text', { x: X(t), y: H - M.b + 20, 'text-anchor': 'middle' }, svg).textContent = t >= 1000 ? (t / 1000) + ' s' : t + ' ms';
      }
      svgEl('text', { x: (M.l + W - M.r) / 2, y: H - 8, 'text-anchor': 'middle', 'class': 'axis-title' }, svg).textContent = 'inference time (log scale), faster to the left';
      svgEl('text', { x: 0, y: 0, 'class': 'axis-title', transform: 'translate(16,' + (M.t + (H - M.t - M.b) / 2) + ') rotate(-90)', 'text-anchor': 'middle' }, svg).textContent = info.label;

      var sorted = pts.slice().sort(function (a, b) { return a.t - b.t; }), best = -Infinity, front = [];
      sorted.forEach(function (p) { if (p.v * better > best) { front.push(p); best = p.v * better; } });
      svgEl('polyline', { points: front.map(function (p) { return X(p.t).toFixed(1) + ',' + Y(p.v).toFixed(1); }).join(' '), fill: 'none', stroke: '#ECEBFA', 'stroke-width': 1.5, 'stroke-dasharray': '5 5', opacity: 0.45 }, svg);

      var memMax = Math.max.apply(null, pts.map(function (p) { return p.mem; })), rad = function (p) { return 5 + 11 * Math.sqrt(p.mem / memMax); };
      pts.slice().sort(function (a, b) { return styleOf(a.name).z - styleOf(b.name).z; }).forEach(function (p) {
        var st = styleOf(p.name), c = svgEl('circle', { cx: X(p.t), cy: Y(p.v), r: rad(p), fill: st.color, 'fill-opacity': st.z >= 4 ? 0.9 : 0.5, stroke: st.z >= 4 ? '#0A0A14' : st.color, 'stroke-width': st.z >= 4 ? 2 : 1, 'class': 'dot' }, svg);
        c.addEventListener('pointermove', function (ev) { tip.show('<b>' + p.name + '</b><small>' + p.t + ' ms &middot; ' + fmt(p.mem) + ' MB &middot; ' + metric.toUpperCase() + ' ' + p.v.toFixed(info.dec) + '</small>', ev); });
        c.addEventListener('pointerleave', function () { tip.hide(); });
      });
      var placed = [];
      PRIORITY.forEach(function (name) {
        var p = pts.filter(function (q) { return q.name === name; })[0]; if (!p) return;
        var st = styleOf(name), r = rad(p), text = st.label || name, w = text.length * 6.6 + 8, cx = X(p.t), cy = Y(p.v);
        var cands = [[cx + r + 5, cy - 7], [cx + r + 5, cy + 7], [cx - r - 5 - w, cy - 7], [cx - r - 5 - w, cy + 7], [cx - w / 2, cy - r - 20], [cx - w / 2, cy + r + 4]], chosen = null;
        for (var k = 0; k < cands.length && !chosen; k++) {
          var box = { x: cands[k][0], y: cands[k][1], w: w, h: 15 };
          if (box.x + w > W - 4 || box.x < M.l - 4) continue;
          var clash = placed.some(function (o) { return !(box.x + box.w < o.x || o.x + o.w < box.x || box.y + box.h < o.y || o.y + o.h < box.y); });
          if (!clash) chosen = box;
        }
        if (!chosen) { if (st.z < 4) return; chosen = { x: cands[0][0], y: cands[0][1], w: w, h: 15 }; }
        placed.push(chosen);
        var tx = svgEl('text', { x: chosen.x, y: chosen.y + 11, 'text-anchor': 'start', 'class': 'lbl' }, svg);
        tx.textContent = text; tx.style.fill = st.z >= 4 ? st.color : C.ink2;
      });
    }

    bindSeg($('#scatterMetric'), 'm', function (v) { metric = v; draw(); });
    bindSeg($('#scatterScale'), 's', function (v) { scale = v; draw(); });
    draw(); redrawOnResize(host, draw);
  }

  /* ======================================================================
   * Alternative primitives (Urban100, EDSR): RBS vs Gaussian Billboard vs 2DGH
   * ==================================================================== */
  function initAlt() {
    var host = $('#altChart'); if (!host) return;
    var metric = 'psnr', A = D.alt, scales = A.scales, W = 880, H = 400, M = { l: 62, r: 170, t: 28, b: 44 }, tip = tooltip(host);
    var UNITS = { psnr: 'dB', ssim: '', time: 'ms' }, DEC = { psnr: 2, ssim: 4, time: 0 };
    function layout() { W = chartWidth(host); var n = W < 600; H = n ? 340 : 400; M = { l: n ? 46 : 62, r: n ? 108 : 170, t: 28, b: 44 }; }

    function draw() {
      layout();
      var old = $('svg', host); if (old) old.remove();
      var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Line chart of ' + metric + ' for RBS, Gaussian Billboards and 2DGH at three scales' }, host);
      var series = metric === 'time'
        ? [{ name: 'RBS', key: 'RBS', color: C.pink }, { name: 'GB and 2DGH', key: 'GB', color: C.cyan }]
        : [{ name: 'RBS', key: 'RBS', color: C.pink }, { name: 'Gaussian Billboard', key: 'GB', color: C.cyan }, { name: '2DGH', key: '2DGH', color: C.amber }];
      var vals = []; series.forEach(function (s) { A[s.key][metric].forEach(function (v) { vals.push(v); }); });
      var log = metric === 'time', lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), pad = (hi - lo) * 0.12;
      if (log) { lo = lo / 1.3; hi = hi * 1.3; } else { lo -= pad; hi += pad; }
      var Y = function (v) { var u = log ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo); return M.t + (H - M.t - M.b) * (1 - u); };
      var X = function (i) { return M.l + 40 + i * (W - M.l - M.r - 80) / (scales.length - 1); };
      var ticks = [];
      if (log) ticks = logTicks(lo, hi); else { var step = niceStep((hi - lo) / 5); for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v); }
      ticks.forEach(function (v) {
        svgEl('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), 'class': 'grid' }, svg);
        svgEl('text', { x: M.l - 10, y: Y(v) + 4, 'text-anchor': 'end' }, svg).textContent = log ? fmtTick(v) : (metric === 'ssim' ? v.toFixed(2) : v.toFixed(1));
      });
      svgEl('text', { x: M.l - 10, y: 12, 'text-anchor': 'end', 'class': 'axis-title' }, svg).textContent = UNITS[metric] || '';
      scales.forEach(function (s, i) { svgEl('text', { x: X(i), y: H - M.b + 24, 'text-anchor': 'middle' }, svg).textContent = '×' + s; });
      if (log) svgEl('text', { x: (M.l + W - M.r) / 2, y: H - 4, 'text-anchor': 'middle', 'class': 'axis-title' }, svg).textContent = 'upscaling factor (log scale on the vertical axis)';

      var labels = [];
      series.forEach(function (s) {
        var d = A[s.key][metric], path = d.map(function (v, i) { return (i ? 'L' : 'M') + X(i).toFixed(1) + ',' + Y(v).toFixed(1); }).join(' ');
        svgEl('path', { d: path, stroke: s.color, 'stroke-width': s.name === 'RBS' ? 3.6 : 2.4, 'class': 'line' }, svg);
        d.forEach(function (v, i) {
          var c = svgEl('circle', { cx: X(i), cy: Y(v), r: s.name === 'RBS' ? 6 : 5, fill: s.color, stroke: '#0A0A14', 'stroke-width': 2, 'class': 'dot' }, svg);
          c.addEventListener('pointermove', function (ev) { tip.show('<b>' + s.name + '</b><small>×' + scales[i] + ': ' + v.toFixed(DEC[metric]) + ' ' + UNITS[metric] + (metric === 'time' ? ' (approx.)' : '') + '</small>', ev); });
          c.addEventListener('pointerleave', function () { tip.hide(); });
          var vt = svgEl('text', { x: X(i), y: Y(v) + (s.name === 'RBS' ? -12 : s.key === 'GB' ? 21 : 35), 'text-anchor': 'middle', 'class': 'lbl' }, svg);
          vt.textContent = v.toFixed(DEC[metric]); vt.style.fill = s.color; vt.style.fontSize = '11px';
        });
        labels.push({ y: Y(d[d.length - 1]), color: s.color, text: s.name });
      });
      spread(labels, 16, M.t + 6, H - M.b);
      labels.forEach(function (l) { var t = svgEl('text', { x: W - M.r + 10, y: l.y + 4, 'class': 'lbl' }, svg); t.textContent = l.text; t.style.fill = l.color; });
    }
    bindSeg($('#altMetric'), 'm', function (m) { metric = m; draw(); });
    draw(); redrawOnResize(host, draw);
  }

  /* ======================================================================
   * Set5 worst case: GSASR vs RBS (u = 4)
   * ==================================================================== */
  function initSet5() {
    var host = $('#set5Table'); if (!host) return;
    var S5 = D.set5, metric = 'psnr', DEC = { psnr: 2, ssim: 4, lpips: 4, dists: 4 }, HIGHER = { psnr: true, ssim: true, lpips: false, dists: false };
    function draw() {
      var rows = S5.scales.map(function (s, i) {
        var g = S5.GSASR[metric][i], r = S5['RBS u=4'][metric][i], d = r - g, dec = DEC[metric], tie = Math.abs(d) < Math.pow(10, -dec) / 2;
        var rWins = !tie && (HIGHER[metric] ? r > g : r < g), gWins = !tie && !rWins;
        return '<tr><td>×' + s + '</td><td class="' + (gWins ? 'win' : '') + '">' + g.toFixed(dec) + '</td><td class="' + (rWins ? 'win' : '') + '">' + r.toFixed(dec) + '</td><td class="' + (tie ? '' : rWins ? 'd-pos' : 'd-neg') + '">' + (tie ? '0' : (d > 0 ? '+' : '−') + Math.abs(d).toFixed(dec)) + '</td></tr>';
      });
      host.innerHTML = '<table><thead><tr><th>Scale</th><th>GSASR</th><th>RBS u=4</th><th>Difference</th></tr></thead><tbody>' + rows.join('') + '</tbody></table>';
    }
    bindSeg($('#set5Metric'), 'm', function (m) { metric = m; draw(); });
    draw();
  }

  U.inits.push(initUExp, initCostChart, initScatter, initAlt, initSet5);
})();
