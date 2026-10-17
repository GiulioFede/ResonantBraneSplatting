/*
 * Resonant Brane Splatting project page: interactive parts (1/3).
 * Hero field, Brane Builder, mode zoo, Brane Lab, rasterizer/culling demo, small helpers.
 * Plain JavaScript, no dependencies. Needs window.RBS (core.js). fit.js and charts.js register themselves in RBSUI.inits.
 * Nothing here is the trained network: the demos re-implement the paper's equations on a single primitive.
 */
(function () {
  'use strict';

  var R = window.RBS;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var fmt = function (n) { return Math.round(n).toLocaleString('en-US'); };
  var clamp = R.clamp;
  var reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var TAU = Math.PI * 2;

  var UI = window.RBSUI = { $: $, $$: $$, fmt: fmt, reduceMotion: reduceMotion, inits: [] };

  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  UI.svgEl = svgEl;

  /** Segmented control: one pressed button at a time. */
  function bindSeg(container, key, cb) {
    if (!container) return;
    container.addEventListener('click', function (ev) {
      var b = ev.target.closest('button');
      if (!b || b.dataset[key] === undefined) return;
      $$('button', container).forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
      cb(b.dataset[key]);
    });
  }
  UI.bindSeg = bindSeg;

  /** Calls cb(true/false) when the element enters/leaves the viewport (true immediately if unsupported). */
  function onVisible(el, cb, threshold) {
    if (!('IntersectionObserver' in window)) { cb(true); return; }
    new IntersectionObserver(function (es) { cb(es[es.length - 1].isIntersecting); }, { threshold: threshold || 0.15 }).observe(el);
  }
  UI.onVisible = onVisible;

  /** A canvas that shows a small float RGB image scaled up. Drawn additively over a dark ground. */
  function makeBlitter(canvas, size, smooth) {
    var off = document.createElement('canvas'); off.width = off.height = size;
    var octx = off.getContext('2d'), idata = octx.createImageData(size, size), ctx = canvas.getContext('2d');
    return function (img, ground) {
      R.toRGBA(img, size, null, idata.data); octx.putImageData(idata, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = ground || '#06060D'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.globalCompositeOperation = 'lighter';
      ctx.imageSmoothingEnabled = smooth !== false; ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
      ctx.globalCompositeOperation = 'source-over';
    };
  }
  UI.makeBlitter = makeBlitter;

  /** x' axis direction (cos rho, sin rho); returns canvas-space point of canonical coordinates (xp, yp). */
  function toCanvas(br, xp, yp, W) {
    var co = Math.cos(br.rho), si = Math.sin(br.rho);
    return [(br.cx + co * br.sx * xp - si * br.sy * yp) * W, (br.cy + si * br.sx * xp + co * br.sy * yp) * W];
  }
  function strokeEllipse(ctx, br, k, W) { // ellipse of canonical radius k
    ctx.beginPath(); ctx.ellipse(br.cx * W, br.cy * W, k * br.sx * W, k * br.sy * W, br.rho, 0, TAU);
  }
  function drawNodal(ctx, br, n, m, Rr, W) { // nodal lines of mode (n, m): x' = roots of H_n, y' = roots of H_m
    ctx.beginPath();
    R.hermiteRoots(n).forEach(function (r) { if (Math.abs(r) > Rr) return; var h = Math.sqrt(Math.max(0, Rr * Rr - r * r)), a = toCanvas(br, r, -h, W), b = toCanvas(br, r, h, W); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); });
    R.hermiteRoots(m).forEach(function (r) { if (Math.abs(r) > Rr) return; var h = Math.sqrt(Math.max(0, Rr * Rr - r * r)), a = toCanvas(br, -h, r, W), b = toCanvas(br, h, r, W); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); });
    ctx.stroke();
  }
  function cross(ctx, x, y, r, color) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke(); ctx.restore();
  }

  /* ======================================================================
   * Hero background: a slowly resonating superposition of Hermite modes.
   * The time evolution is the real one of the oscillator: mode (n, m) rotates with frequency (n + m + 1).
   * ==================================================================== */
  function initHeroField() {
    var cv = $('#heroField'); if (!cv) return;
    var S = 160, ctx = cv.getContext('2d'), id = ctx.createImageData(S, S), px = id.data, DEG = 4, L1 = DEG + 1, ext = 4.1;
    var gx = [], c = [], i, j, n, m;
    var hb = new Float64Array(L1);
    for (n = 0; n < L1; n++) gx.push(new Float64Array(S));
    for (i = 0; i < S; i++) {
      var z = -ext + (i + 0.5) / S * 2 * ext, e = Math.exp(-0.5 * z * z); R.hermite(z, DEG, hb);
      for (n = 0; n < L1; n++) gx[n][i] = hb[n] * e;
    }
    var rng = R.makeRng(11);
    for (n = 0; n < L1; n++) for (m = 0; m < L1; m++) c.push({ n: n, m: m, a: rng.uniform(-1, 1) * 1.2 / (1 + 0.7 * (n + m)), ph: rng.uniform(0, TAU) });
    var T = [], row = new Float64Array(L1);
    for (n = 0; n < L1; n++) T.push(new Float64Array(S));
    var BG = [10, 10, 20], POS = [255, 77, 157], NEG = [47, 225, 255];

    function draw(t) {
      var k;
      for (n = 0; n < L1; n++) T[n].fill(0);
      for (k = 0; k < c.length; k++) {
        var q = c[k], amp = q.a * Math.cos((q.n + q.m + 1) * t * 0.5 + q.ph), g = gx[q.m];
        if (Math.abs(amp) < 1e-4) continue;
        var Tn = T[q.n]; for (j = 0; j < S; j++) Tn[j] += amp * g[j];
      }
      for (j = 0; j < S; j++) {
        for (n = 0; n < L1; n++) row[n] = T[n][j];
        for (i = 0; i < S; i++) {
          var v = 0; for (n = 0; n < L1; n++) v += gx[n][i] * row[n];
          var a = 1 - Math.exp(-Math.abs(v) * 1.5), col = v >= 0 ? POS : NEG, p = (j * S + i) * 4;
          px[p] = BG[0] + (col[0] - BG[0]) * a; px[p + 1] = BG[1] + (col[1] - BG[1]) * a; px[p + 2] = BG[2] + (col[2] - BG[2]) * a; px[p + 3] = 255;
        }
      }
      ctx.putImageData(id, 0, 0);
    }
    draw(1.3);
    if (reduceMotion) return;
    var vis = true, last = 0, t0 = performance.now();
    onVisible($('.hero'), function (v) { vis = v; if (v) requestAnimationFrame(loop); }, 0.02);
    function loop(ts) {
      if (!vis || document.hidden) return;
      if (ts - last > 45) { last = ts; draw(1.3 + (ts - t0) / 1000); }
      requestAnimationFrame(loop);
    }
    document.addEventListener('visibilitychange', function () { if (!document.hidden && vis) requestAnimationFrame(loop); });
    requestAnimationFrame(loop);
  }

  /* ======================================================================
   * Hero demo: Brane Builder (one primitive, degree N = M)
   * ==================================================================== */
  var BUILD_BRANE = { cx: 0.5, cy: 0.5, sx: 0.13, sy: 0.13, rho: 0.26, alpha: 1 };
  var BASE = [0.7, 0.68, 0.76];

  function initBuilder() {
    var cv = $('#builderCanvas'); if (!cv) return;
    var ctx = cv.getContext('2d'), S = 256, W = cv.width, blit = makeBlitter(cv, S, true);
    var st = { deg: reduceMotion ? 5 : 0, view: 'brane', seed: 3, playing: false, timer: 0 };
    var out = new Float32Array(S * S * 3);

    function brane(D) { return { cx: BUILD_BRANE.cx, cy: BUILD_BRANE.cy, sx: BUILD_BRANE.sx, sy: BUILD_BRANE.sy, rho: BUILD_BRANE.rho, alpha: 1, D: D, coef: R.modeColors(D, st.seed, { base: BASE }) }; }

    function drawModes(D) { // grid of the weighted modes c_nm * Psi_nm * E, hue-normalized so that every tile is readable
      var L1 = D + 1, tp = Math.floor(S / L1), Rt = Math.min(Math.sqrt(2 * D + 1), 4.6) + 0.3, coef = R.modeColors(D, st.seed, { base: BASE });
      out.fill(0.02);
      var hx = new Float64Array(L1), hy = new Float64Array(L1);
      for (var m = 0; m < L1; m++) for (var n = 0; n < L1; n++) {
        var l = (n * L1 + m) * 3, mx = Math.max(Math.abs(coef[l]), Math.abs(coef[l + 1]), Math.abs(coef[l + 2]), 1e-6), u = [coef[l] / mx, coef[l + 1] / mx, coef[l + 2] / mx];
        for (var b = 1; b < tp - 1; b++) {
          R.hermite(((b + 0.5) / tp * 2 - 1) * Rt, D, hy);
          var ey = Math.exp(-0.5 * Math.pow(((b + 0.5) / tp * 2 - 1) * Rt, 2));
          for (var a = 1; a < tp - 1; a++) {
            var xx = ((a + 0.5) / tp * 2 - 1) * Rt; R.hermite(xx, D, hx);
            var v = hx[n] * hy[m] * Math.exp(-0.5 * xx * xx) * ey * 1.15, p = ((m * tp + b) * S + (n * tp + a)) * 3;
            out[p] = clamp(0.5 + 0.5 * u[0] * v, 0, 1); out[p + 1] = clamp(0.5 + 0.5 * u[1] * v, 0, 1); out[p + 2] = clamp(0.5 + 0.5 * u[2] * v, 0, 1);
          }
        }
      }
      blit(out, '#06060D');
    }

    function render() {
      var D = st.deg, br = brane(D), Rr = R.rmaxOf(D);
      if (st.view === 'modes') drawModes(D);
      else {
        R.renderBranes([br], S, [0, 1, 0, 1], { out: out });
        if (st.view === 'lines') for (var k = 0; k < out.length; k++) out[k] *= 0.55;
        blit(out);
        if (st.view === 'lines' && D > 0) {
          ctx.save(); ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,.85)'; drawNodal(ctx, br, D, D, Rr, W);
          ctx.setLineDash([10, 8]); ctx.strokeStyle = '#FFC247'; strokeEllipse(ctx, br, Rr, W); ctx.stroke(); ctx.restore();
        }
      }
      $('#statModes').textContent = (D + 1) * (D + 1);
      $('#statParams').textContent = 6 + 3 * (D + 1) * (D + 1);
      $('#statDeg').textContent = 'N = M = ' + D;
      $('#statR').innerHTML = 'R<sub>max</sub> = ' + Rr.toFixed(2);
      $('#bDegree').value = D; $('#bDegreeOut').textContent = D;
      $('#builderChip').textContent = st.view === 'brane' ? (D === 0 ? 'A Gaussian splat: one mode, one color' : 'One primitive: the sum of ' + (D + 1) * (D + 1) + ' colored modes')
        : st.view === 'modes' ? 'Each tile: color c' + String.fromCharCode(8344, 8345) + ' times mode Ψ' + String.fromCharCode(8344, 8345) + ', column n, row m'
        : D === 0 ? 'Degree 0 has no nodal lines' : 'Nodal lines of the highest mode Ψ' + D + ',' + D + ', dashed: turning point';
    }

    function setDeg(d) { st.deg = clamp(Math.round(d), 0, 12); render(); }
    function stop() { st.playing = false; clearInterval(st.timer); $('#bPlay span').textContent = 'Build it up'; }
    function play(from, to) {
      stop(); st.playing = true; $('#bPlay span').textContent = 'Pause';
      if (from != null) setDeg(from);
      st.timer = setInterval(function () { if (st.deg >= to) { stop(); return; } setDeg(st.deg + 1); }, 520);
    }
    $('#bPlay').addEventListener('click', function () {
      if (st.playing) { stop(); return; }
      if (reduceMotion) { setDeg(12); return; }
      play(st.deg >= 12 ? 0 : null, 12);
    });
    $('#bDegree').addEventListener('input', function (e) { stop(); setDeg(+e.target.value); });
    $('#bShuffle').addEventListener('click', function () { st.seed++; render(); });
    bindSeg($('#bView'), 'v', function (v) { st.view = v; render(); });

    // the same primitive with only the zero-order mode: a plain Gaussian splat
    var gi = $('#gaussInset');
    if (gi) {
      var gs = 64, gimg = R.renderBranes([{ cx: 0.5, cy: 0.5, sx: BUILD_BRANE.sx, sy: BUILD_BRANE.sy, rho: BUILD_BRANE.rho, alpha: 1, D: 0, coef: new Float64Array(BASE) }], gs, [0, 1, 0, 1]).img;
      makeBlitter(gi, gs, true)(gimg);
    }

    render();
    if (!reduceMotion) onVisible(cv, function (v) { if (v && !st.started) { st.started = true; play(0, 5); } }, 0.3);
  }

  /* ======================================================================
   * Mode zoo: Psi_{n,m} for n, m <= 5
   * ==================================================================== */
  function initZoo() {
    var cv = $('#zooCanvas'); if (!cv) return;
    var ctx = cv.getContext('2d'), N = 6, TP = 64, SZ = N * TP, W = cv.width, TW = W / N, Rt = 3.9, rimR = Math.sqrt(2 * 5 + 1);
    var off = document.createElement('canvas'); off.width = off.height = SZ;
    var octx = off.getContext('2d'), id = octx.createImageData(SZ, SZ), px = id.data, hx = new Float64Array(N), hy = new Float64Array(N);
    var BG = [10, 10, 20], POS = [255, 77, 157], NEG = [47, 225, 255];
    for (var m = 0; m < N; m++) for (var n = 0; n < N; n++) for (var b = 0; b < TP; b++) {
      var y = ((b + 0.5) / TP * 2 - 1) * Rt; R.hermite(y, N - 1, hy); var ey = Math.exp(-0.5 * y * y);
      for (var a = 0; a < TP; a++) {
        var x = ((a + 0.5) / TP * 2 - 1) * Rt; R.hermite(x, N - 1, hx);
        var v = hx[n] * hy[m] * Math.exp(-0.5 * x * x) * ey, al = Math.tanh(Math.abs(v) * 1.5), col = v >= 0 ? POS : NEG, p = ((m * TP + b) * SZ + (n * TP + a)) * 4;
        px[p] = BG[0] + (col[0] - BG[0]) * al; px[p + 1] = BG[1] + (col[1] - BG[1]) * al; px[p + 2] = BG[2] + (col[2] - BG[2]) * al; px[p + 3] = 255;
      }
    }
    octx.putImageData(id, 0, 0);
    var sel = [2, 1];

    function draw() {
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(off, 0, 0, W, W);
      ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(160,160,230,.22)'; ctx.beginPath();
      for (var i = 1; i < N; i++) { ctx.moveTo(i * TW, 0); ctx.lineTo(i * TW, W); ctx.moveTo(0, i * TW); ctx.lineTo(W, i * TW); }
      ctx.stroke();
      ctx.setLineDash([5, 5]); ctx.strokeStyle = 'rgba(255,194,71,.55)'; ctx.lineWidth = 1.2;
      for (var mm = 0; mm < N; mm++) for (var nn = 0; nn < N; nn++) { ctx.beginPath(); ctx.arc((nn + 0.5) * TW, (mm + 0.5) * TW, rimR / Rt * TW / 2, 0, TAU); ctx.stroke(); }
      ctx.setLineDash([]);
      ctx.font = '500 ' + Math.round(TW * 0.12) + 'px "JetBrains Mono", monospace'; ctx.fillStyle = 'rgba(236,235,250,.7)'; ctx.textBaseline = 'top';
      for (mm = 0; mm < N; mm++) for (nn = 0; nn < N; nn++) ctx.fillText(nn + ',' + mm, nn * TW + 7, mm * TW + 6);
      ctx.lineWidth = 3; ctx.strokeStyle = '#fff'; ctx.strokeRect(sel[0] * TW + 1.5, sel[1] * TW + 1.5, TW - 3, TW - 3);
    }
    function describe() {
      var n = sel[0], m = sel[1];
      $('#zooName').innerHTML = '&Psi;<sub>' + n + ',' + m + '</sub>';
      $('#zooDesc').innerHTML = (n === 0 && m === 0)
        ? 'The zero-order mode is the Gaussian itself: one lobe, no nodal lines. A Brane with only this mode is a standard Gaussian splat.'
        : n + ' nodal line' + (n === 1 ? '' : 's') + ' across <i>x</i>&prime; and ' + m + ' across <i>y</i>&prime; make <b>' + ((n + 1) * (m + 1)) + ' lobes</b> of alternating sign. Along <i>x</i>&prime; the mode lives up to &radic;(2&middot;' + n + '+1) = ' + Math.sqrt(2 * n + 1).toFixed(2) + ', along <i>y</i>&prime; up to ' + Math.sqrt(2 * m + 1).toFixed(2) + '.';
    }
    function pick(ev) {
      var r = cv.getBoundingClientRect(), n = clamp(Math.floor((ev.clientX - r.left) / r.width * N), 0, N - 1), m = clamp(Math.floor((ev.clientY - r.top) / r.height * N), 0, N - 1);
      if (n === sel[0] && m === sel[1]) return;
      sel = [n, m]; draw(); describe();
    }
    cv.addEventListener('pointermove', pick); cv.addEventListener('pointerdown', pick);
    draw(); describe();
  }

  /* ======================================================================
   * Brane Lab: edit one Brane
   * ==================================================================== */
  function initLab() {
    var cv = $('#labCanvas'); if (!cv) return;
    var ctx = cv.getContext('2d'), S = 256, W = cv.width, blit = makeBlitter(cv, S, true), out = new Float32Array(S * S * 3);
    var MAXD = 6, DEFAULT_C00 = [0.85, 0.35, 0.35];
    var st = { cx: 0.5, cy: 0.5, sx: 0.14, sy: 0.14, rho: 0, alpha: 1, D: 3, sel: [0, 0], c: {}, over: { foot: true, rmax: true, lines: false } };
    function key(n, m) { return n + ',' + m; }
    function clearCoefs() { st.c = {}; st.c[key(0, 0)] = DEFAULT_C00.slice(); }
    function getC(n, m) { return st.c[key(n, m)] || [0, 0, 0]; }
    clearCoefs();

    function brane() {
      var D = st.D, L1 = D + 1, coef = new Float64Array(L1 * L1 * 3);
      for (var n = 0; n <= D; n++) for (var m = 0; m <= D; m++) { var c = getC(n, m), l = (n * L1 + m) * 3; coef[l] = c[0]; coef[l + 1] = c[1]; coef[l + 2] = c[2]; }
      return { cx: st.cx, cy: st.cy, sx: st.sx, sy: st.sy, rho: st.rho * Math.PI / 180, alpha: st.alpha, D: D, coef: coef };
    }

    function render() {
      var br = brane(), Rr = R.rmaxOf(st.D);
      R.renderBranes([br], S, [0, 1, 0, 1], { out: out }); blit(out);
      ctx.save();
      if (st.over.foot) { ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,.8)'; strokeEllipse(ctx, br, 1, W); ctx.stroke(); ctx.setLineDash([4, 6]); ctx.strokeStyle = 'rgba(255,255,255,.4)'; strokeEllipse(ctx, br, 2, W); ctx.stroke(); ctx.setLineDash([]); }
      if (st.over.rmax) { ctx.lineWidth = 2.5; ctx.setLineDash([12, 8]); ctx.strokeStyle = '#FFC247'; strokeEllipse(ctx, br, Rr, W); ctx.stroke(); ctx.setLineDash([]); }
      if (st.over.lines) { ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,.9)'; drawNodal(ctx, br, st.sel[0], st.sel[1], Rr, W); }
      cross(ctx, st.cx * W, st.cy * W, 7, 'rgba(255,255,255,.9)');
      ctx.restore();
    }

    function rgb(c, signed) { var f = signed ? function (v) { return Math.round(clamp(0.5 + 0.5 * v, 0, 1) * 255); } : function (v) { return Math.round(clamp(v, 0, 1) * 255); }; return 'rgb(' + f(c[0]) + ',' + f(c[1]) + ',' + f(c[2]) + ')'; }

    function buildPicker() {
      var host = $('#modePick'), L1 = st.D + 1; host.style.setProperty('--cols', L1); host.innerHTML = '';
      if (st.sel[0] > st.D || st.sel[1] > st.D) st.sel = [0, 0];
      for (var m = 0; m < L1; m++) for (var n = 0; n < L1; n++) {
        var c = getC(n, m), b = document.createElement('button');
        b.type = 'button'; b.dataset.n = n; b.dataset.m = m; b.textContent = n + ',' + m;
        b.setAttribute('aria-label', 'Mode ' + n + ',' + m); b.title = 'Mode (' + n + ', ' + m + ')';
        b.style.background = (n === 0 && m === 0) ? rgb(c, false) : rgb(c, true);
        if (n !== 0 || m !== 0) { if (Math.abs(c[0]) + Math.abs(c[1]) + Math.abs(c[2]) < 0.015) b.classList.add('off'); }
        b.setAttribute('aria-pressed', n === st.sel[0] && m === st.sel[1] ? 'true' : 'false');
        host.appendChild(b);
      }
      syncColorSliders();
    }
    function refreshSwatch() {
      var n = st.sel[0], m = st.sel[1], b = $('#modePick button[data-n="' + n + '"][data-m="' + m + '"]'); if (!b) return;
      var c = getC(n, m); b.style.background = (n === 0 && m === 0) ? rgb(c, false) : rgb(c, true);
      b.classList.toggle('off', !(n === 0 && m === 0) && Math.abs(c[0]) + Math.abs(c[1]) + Math.abs(c[2]) < 0.015);
    }
    function syncColorSliders() {
      var n = st.sel[0], m = st.sel[1], base = n === 0 && m === 0, c = getC(n, m);
      $('#modeTitle').innerHTML = 'Mode (' + n + ', ' + m + ') &middot; <b>' + (base ? 'base color (sigmoid, 0 to 1)' : 'signed residual (tanh, &minus;1 to 1)') + '</b>';
      $$('#labColor input').forEach(function (inp) {
        var ch = +inp.dataset.ch; inp.min = base ? 0 : -1; inp.max = 1; inp.step = 0.01; inp.value = c[ch]; inp.nextElementSibling.textContent = c[ch].toFixed(2);
      });
    }
    function syncFootSliders() {
      $$('#labFoot input').forEach(function (inp) {
        var p = inp.dataset.p, v = st[p]; inp.value = v; inp.nextElementSibling.textContent = p === 'rho' ? Math.round(v) + '°' : p === 'sx' || p === 'sy' ? v.toFixed(3) : v.toFixed(2);
      });
      $('#labDeg').value = st.D; $('#labDegOut').textContent = st.D;
    }
    function all() { syncFootSliders(); buildPicker(); render(); }

    $('#labFoot').addEventListener('input', function (e) {
      var inp = e.target; if (!inp.dataset.p) return; st[inp.dataset.p] = +inp.value;
      var p = inp.dataset.p; inp.nextElementSibling.textContent = p === 'rho' ? Math.round(st[p]) + '°' : p === 'sx' || p === 'sy' ? st[p].toFixed(3) : st[p].toFixed(2);
      render();
    });
    $('#labDeg').addEventListener('input', function (e) { st.D = +e.target.value; $('#labDegOut').textContent = st.D; buildPicker(); render(); });
    $('#modePick').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return; st.sel = [+b.dataset.n, +b.dataset.m];
      $$('#modePick button').forEach(function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); }); syncColorSliders(); render();
    });
    $('#labColor').addEventListener('input', function (e) {
      var inp = e.target; if (inp.dataset.ch === undefined) return;
      var k = key(st.sel[0], st.sel[1]), c = (st.c[k] || [0, 0, 0]).slice(); c[+inp.dataset.ch] = +inp.value; st.c[k] = c;
      inp.nextElementSibling.textContent = (+inp.value).toFixed(2); refreshSwatch(); render();
    });
    $('#labOver').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return; var o = b.dataset.o, on = b.getAttribute('aria-pressed') !== 'true';
      b.setAttribute('aria-pressed', on ? 'true' : 'false'); st.over[o] = on; render();
    });

    // drag to move the Brane
    var dragging = false;
    function move(ev) {
      var r = cv.getBoundingClientRect(); st.cx = clamp((ev.clientX - r.left) / r.width, 0.2, 0.8); st.cy = clamp((ev.clientY - r.top) / r.height, 0.2, 0.8); syncFootSliders(); render();
    }
    cv.addEventListener('pointerdown', function (ev) { dragging = true; cv.setPointerCapture(ev.pointerId); move(ev); });
    cv.addEventListener('pointermove', function (ev) { if (dragging) move(ev); });
    cv.addEventListener('pointerup', function () { dragging = false; });
    cv.addEventListener('pointercancel', function () { dragging = false; });
    cv.style.cursor = 'grab'; cv.style.touchAction = 'pan-y';

    var PRESETS = {
      gauss: function () { clearCoefs(); st.c[key(0, 0)] = [0.9, 0.55, 0.3]; Object.assign(st, { D: 0, sx: 0.14, sy: 0.14, rho: 0, alpha: 1, cx: 0.5, cy: 0.5 }); },
      edge: function () { clearCoefs(); st.c[key(0, 0)] = [0.42, 0.5, 0.72]; st.c[key(1, 0)] = [0.55, -0.1, -0.5]; Object.assign(st, { D: 2, sx: 0.11, sy: 0.17, rho: 20, alpha: 1, cx: 0.5, cy: 0.5 }); },
      checker: function () { clearCoefs(); st.c[key(0, 0)] = [0.5, 0.42, 0.34]; st.c[key(1, 1)] = [0.55, 0.15, -0.5]; Object.assign(st, { D: 2, sx: 0.13, sy: 0.13, rho: 0, alpha: 1, cx: 0.5, cy: 0.5 }); },
      figure: function () { // the colors of "Brane A" in the paper's figure script
        clearCoefs(); st.c[key(0, 0)] = [0.85, 0.35, 0.35]; st.c[key(1, 0)] = [0.7, -0.7, 0]; st.c[key(0, 1)] = [0, 0.6, -0.6]; st.c[key(3, 3)] = [0, 0.9, -0.9];
        Object.assign(st, { D: 3, sx: 0.12, sy: 0.12, rho: 0, alpha: 1, cx: 0.5, cy: 0.5 });
      },
      random: function () {
        var rng = R.makeRng((Math.random() * 1e9) | 0); clearCoefs(); st.D = 4;
        st.c[key(0, 0)] = [rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9)];
        for (var n = 0; n <= 4; n++) for (var m = 0; m <= 4; m++) if (n + m > 0 && rng.next() < 0.55) { var a = 0.9 / (1 + 0.6 * (n + m)); st.c[key(n, m)] = [rng.uniform(-a, a), rng.uniform(-a, a), rng.uniform(-a, a)]; }
        Object.assign(st, { sx: rng.uniform(0.09, 0.17), sy: rng.uniform(0.09, 0.17), rho: Math.round(rng.uniform(-90, 90)), alpha: 1, cx: 0.5, cy: 0.5 });
      }
    };
    $('.presets').addEventListener('click', function (e) {
      var b = e.target.closest('button[data-preset]'); if (!b) return; PRESETS[b.dataset.preset](); st.sel = [0, 0]; all();
    });
    PRESETS.figure(); all();
  }

  /* ======================================================================
   * Rasterizer: d_max box, then the quantum turning point R_max
   * Three Branes of the paper's figure, on a 12 x 12 pixel grid (image coordinates, y down; the colors of odd m flip sign
   * because the vertical axis is flipped with respect to the script).
   * ==================================================================== */
  function initRast() {
    var cv = $('#rastCanvas'); if (!cv) return;
    var ctx = cv.getContext('2d'), G = 12, W = cv.width, PW = W / G, dom = [-0.5, G - 0.5, -0.5, G - 0.5];
    var CF = { // script colors (n, m): [r, g, b]
      A: { '0,0': [0.85, 0.35, 0.35], '1,0': [0.7, -0.7, 0], '0,1': [0, 0.6, -0.6], '3,3': [0, 0.9, -0.9] },
      B: { '0,0': [0.35, 0.65, 0.9], '1,0': [-0.7, 0.7, 0], '0,1': [0.6, 0, 0.6] },
      C: { '0,0': [0.85, 0.35, 0.35], '1,0': [0.7, -0.7, 0], '0,1': [0, 0.6, -0.6] }
    };
    var SPEC = [
      { name: 'A', cx: 5, cy: 6, s: 0.7, rho: 0, cf: CF.A, color: '#FFC247' },
      { name: 'B', cx: 8.5, cy: 3.5, s: 0.6, rho: -Math.PI / 4, cf: CF.B, color: '#FF7B54' },
      { name: 'C', cx: 1.5, cy: 1.5, s: 0.5, rho: 15 * Math.PI / 180, cf: CF.C, color: '#A6E36E' }
    ];
    var st = { D: 3, dmax: 3.5, cull: 'both', px: 6, py: 6 };
    var tmp = document.createElement('canvas'); tmp.width = tmp.height = G;
    var tctx = tmp.getContext('2d'), tdata = tctx.createImageData(G, G);

    function branes(D) {
      return SPEC.map(function (s) {
        var L1 = D + 1, coef = new Float64Array(L1 * L1 * 3);
        Object.keys(s.cf).forEach(function (k) {
          var p = k.split(','), n = +p[0], m = +p[1]; if (n > D || m > D) return;
          var sg = m % 2 ? -1 : 1, l = (n * L1 + m) * 3; coef[l] = s.cf[k][0] * sg; coef[l + 1] = s.cf[k][1] * sg; coef[l + 2] = s.cf[k][2] * sg;
        });
        return { cx: s.cx, cy: s.cy, sx: s.s, sy: s.s, rho: s.rho, alpha: 1, D: D, coef: coef };
      });
    }
    function pairInfo(br, X, Y, D) { // canonical radius^2 and box test of brane br for the pixel centre (X, Y)
      var dx = X - br.cx, dy = Y - br.cy, co = Math.cos(br.rho), si = Math.sin(br.rho), xp = (co * dx + si * dy) / br.sx, yp = (-si * dx + co * dy) / br.sy;
      return { inBox: Math.abs(dx) < st.dmax && Math.abs(dy) < st.dmax, r2: xp * xp + yp * yp, cheb: Math.max(Math.abs(dx), Math.abs(dy)) };
    }

    function draw() {
      var D = st.D, br = branes(D), full = R.renderBranes(br, G, dom, { cull: 'none' });
      var cur = st.cull === 'both' ? R.renderBranes(br, G, dom, { cull: 'rmax', dmax: st.dmax }) : full;
      R.toRGBA(cur.img, G, null, tdata.data); tctx.putImageData(tdata, 0, 0);
      ctx.fillStyle = '#06060D'; ctx.fillRect(0, 0, W, W);
      ctx.globalCompositeOperation = 'lighter'; ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, W, W); ctx.globalCompositeOperation = 'source-over';
      ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,.14)'; ctx.beginPath();
      for (var i = 1; i < G; i++) { ctx.moveTo(i * PW, 0); ctx.lineTo(i * PW, W); ctx.moveTo(0, i * PW); ctx.lineTo(W, i * PW); }
      ctx.stroke();
      var U = function (v) { return (v + 0.5) * PW; };

      // R_max ellipses and centres
      var Rr = R.rmaxOf(D);
      br.forEach(function (b, k) {
        ctx.lineWidth = 2.5; ctx.setLineDash([11, 7]); ctx.strokeStyle = SPEC[k].color; ctx.beginPath();
        ctx.ellipse(U(b.cx), U(b.cy), Rr * b.sx * PW, Rr * b.sy * PW, b.rho, 0, TAU); ctx.stroke(); ctx.setLineDash([]);
        cross(ctx, U(b.cx), U(b.cy), 6, '#fff');
        ctx.font = '600 15px "Space Grotesk", sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        var lbl = 'Brane ' + SPEC[k].name, tw = ctx.measureText(lbl).width + 12, ly = U(b.cy) - Rr * Math.max(b.sx, b.sy) * PW - 14;
        ctx.fillStyle = SPEC[k].color; ctx.beginPath(); ctx.roundRect ? ctx.roundRect(U(b.cx) - tw / 2, ly - 11, tw, 22, 7) : ctx.rect(U(b.cx) - tw / 2, ly - 11, tw, 22); ctx.fill();
        ctx.fillStyle = '#0A0A14'; ctx.fillText(lbl, U(b.cx), ly + 1);
      });
      // d_max box and target pixel
      ctx.lineWidth = 3; ctx.setLineDash([14, 6, 3, 6]); ctx.strokeStyle = '#2FE1FF';
      ctx.strokeRect(U(st.px - st.dmax), U(st.py - st.dmax), 2 * st.dmax * PW, 2 * st.dmax * PW); ctx.setLineDash([]);
      ctx.lineWidth = 3.5; ctx.strokeStyle = '#FF4D9D'; ctx.strokeRect(st.px * PW + 1.5, st.py * PW + 1.5, PW - 3, PW - 3);

      // counts over the whole 12 x 12 output
      var c1 = 0, c2 = 0, total = br.length * G * G;
      br.forEach(function (b) { for (var j = 0; j < G; j++) for (var i2 = 0; i2 < G; i2++) { var q = pairInfo(b, i2, j, D); if (q.inBox) { c1++; if (q.r2 <= Rr * Rr) c2++; } } });
      var mse = R.mse(full.img, cur.img), mx = 0; for (var t = 0; t < full.img.length; t++) mx = Math.max(mx, Math.abs(full.img[t] - cur.img[t]));
      var bar = function (label, n, color) { return '<div class="fn-row"><span>' + label + '</span><div class="fn-bar"><i style="width:' + (100 * n / total).toFixed(1) + '%;background:' + color + '"></i></div><b>' + n + '</b></div>'; };
      $('#funnel').innerHTML = bar('All Brane&ndash;pixel pairs', total, '#7C7B9C') + bar('Inside the <i>d</i><sub>max</sub> box', c1, '#2FE1FF') + bar('Inside <i>R</i><sub>max</sub>', c2, '#FF4D9D') +
        '<div class="fn-row err"><span>' + (st.cull === 'both'
          ? 'Picture with culling against the full evaluation: <b>' + (mse < 1e-12 ? 'identical' : R.psnr(mse).toFixed(1) + ' dB, largest change ' + (mx * 255).toFixed(1) + ' / 255') + '</b>'
          : 'No culling: all ' + total + ' pairs are evaluated. This is the reference picture.') + '</span></div>';

      // the selected pixel
      var rows = br.map(function (b, k) {
        var q = pairInfo(b, st.px, st.py, D), dot = '<i style="background:' + SPEC[k].color + '"></i>', name = '<b>Brane ' + SPEC[k].name + '</b> ';
        if (!q.inBox) return '<div class="pc-row">' + dot + '<div>' + name + '<span class="out">skipped by <i>d</i><sub>max</sub></span>, its center is ' + q.cheb.toFixed(1) + ' pixels away (box half-side ' + st.dmax + ')</div></div>';
        var r = Math.sqrt(q.r2);
        if (q.r2 > Rr * Rr) return '<div class="pc-row">' + dot + '<div>' + name + '<span class="cut">skipped by <i>R</i><sub>max</sub></span>, canonical radius ' + r.toFixed(2) + ' &gt; ' + Rr.toFixed(2) + '</div></div>';
        return '<div class="pc-row">' + dot + '<div>' + name + '<span class="ok">evaluated</span>, canonical radius ' + r.toFixed(2) + ' &le; ' + Rr.toFixed(2) + '</div></div>';
      });
      $('#pixelCard').innerHTML = '<h5>Target pixel (' + st.px + ', ' + st.py + ')</h5>' + rows.join('');
    }

    $('#rastD').addEventListener('input', function (e) { st.dmax = +e.target.value; $('#rastDOut').textContent = st.dmax; draw(); });
    $('#rastN').addEventListener('input', function (e) { st.D = +e.target.value; $('#rastNOut').textContent = st.D; draw(); });
    bindSeg($('#rastCull'), 'c', function (v) { st.cull = v; draw(); });
    var down = false;
    function pick(ev) {
      var r = cv.getBoundingClientRect(), x = clamp(Math.floor((ev.clientX - r.left) / r.width * G), 0, G - 1), y = clamp(Math.floor((ev.clientY - r.top) / r.height * G), 0, G - 1);
      if (x === st.px && y === st.py) return; st.px = x; st.py = y; draw();
    }
    cv.addEventListener('pointerdown', function (ev) { down = true; cv.setPointerCapture(ev.pointerId); pick(ev); });
    cv.addEventListener('pointermove', function (ev) { if (down) pick(ev); });
    cv.addEventListener('pointerup', function () { down = false; }); cv.addEventListener('pointercancel', function () { down = false; });
    draw();
  }

  /* ======================================================================
   * Small things: copy buttons, lightbox, missing figures, scroll spy
   * ==================================================================== */
  function initMisc() {
    $$('.copy').forEach(function (b) {
      b.addEventListener('click', function () {
        var text = $(b.dataset.copy).textContent;
        var done = function () { var o = b.textContent; b.textContent = 'Copied'; b.classList.add('done'); setTimeout(function () { b.textContent = o; b.classList.remove('done'); }, 1400); };
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback); else fallback();
        function fallback() { var ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ } ta.remove(); }
      });
    });

    // A figure that is not in static/images yet shows a clear placeholder instead of a broken icon.
    document.addEventListener('error', function (e) {
      var im = e.target;
      if (!im || im.tagName !== 'IMG' || !im.classList.contains('paper-fig')) return;
      var d = document.createElement('div'); d.className = 'fig-missing';
      d.innerHTML = '<strong>Figure not added yet</strong><span>' + im.getAttribute('src') + '</span>';
      im.replaceWith(d);
    }, true);

    var lb = $('#lightbox'), lbImg = $('img', lb);
    document.addEventListener('click', function (e) {
      var im = e.target.closest && e.target.closest('img.paper-fig');
      if (im && lb.showModal) { lbImg.src = im.currentSrc || im.src; lbImg.alt = im.alt; lb.showModal(); }
    });
    lb.addEventListener('click', function () { lb.close(); });

    if ('IntersectionObserver' in window) {
      var links = {}; $$('.topbar-nav a').forEach(function (a) { links[a.getAttribute('href').slice(1)] = a; });
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (!en.isIntersecting) return;
          var id = en.target.id, key = links[id] ? id : null;
          if (!key) return;
          Object.keys(links).forEach(function (k) { links[k].removeAttribute('aria-current'); });
          links[key].setAttribute('aria-current', 'true');
        });
      }, { rootMargin: '-35% 0px -55% 0px' });
      $$('main section[id]').forEach(function (s) { io.observe(s); });
    }
  }

  /* ---------- boot ---------- */
  function boot() {
    [initHeroField, initBuilder, initZoo, initLab, initRast, initMisc].concat(UI.inits).forEach(function (f) {
      try { f(); } catch (err) { if (window.console) console.error(f.name || 'init', err); }
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
