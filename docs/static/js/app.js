/*
 * Resonant Brane Splatting project page: interactive parts (1/3).
 * Hero field, Brane Builder, mode zoo, rasterizer/culling demo, video, small helpers.
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

  /** LaTeX to HTML with the bundled KaTeX. Falls back to the raw TeX if KaTeX did not load. */
  UI.tex = function (src, display) {
    try { return window.katex.renderToString(src, { throwOnError: false, displayMode: !!display, strict: 'ignore' }); } catch (e) { return src; }
  };
  /** Typesets every \( ... \) and \[ ... \] found in the static HTML of `root`. */
  UI.typeset = function (root) {
    if (typeof window.renderMathInElement !== 'function') return;
    try {
      window.renderMathInElement(root || document.body, {
        delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
        throwOnError: false, strict: 'ignore', ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      });
    } catch (e) { if (window.console) console.error('math', e); }
  };

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
    var S = 256, blit = makeBlitter(cv, S, true);
    var st = { deg: reduceMotion ? 5 : 0, seed: 3, playing: false, timer: 0 };
    var out = new Float32Array(S * S * 3);

    function brane(D) { return { cx: BUILD_BRANE.cx, cy: BUILD_BRANE.cy, sx: BUILD_BRANE.sx, sy: BUILD_BRANE.sy, rho: BUILD_BRANE.rho, alpha: 1, D: D, coef: R.modeColors(D, st.seed, { base: BASE }) }; }

    function render() {
      var D = st.deg, Rr = R.rmaxOf(D);
      R.renderBranes([brane(D)], S, [0, 1, 0, 1], { out: out });
      blit(out);
      $('#statModes').textContent = (D + 1) * (D + 1);
      $('#statParams').textContent = 6 + 3 * (D + 1) * (D + 1);
      $('#statDeg').innerHTML = UI.tex('N=M=' + D);
      $('#statR').innerHTML = UI.tex('R_{\\max}=\\sqrt{2N+1}=' + Rr.toFixed(2));
      $('#bDegree').value = D; $('#bDegreeOut').textContent = D;
      $('#builderChip').textContent = D === 0 ? 'A Gaussian splat: one mode, one color' : 'One primitive: the sum of ' + (D + 1) * (D + 1) + ' colored modes';
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
    var ctx = cv.getContext('2d'), N = 6, TP = 64, SZ = N * TP, W = cv.width, TW = W / N, Rt = 3.9;
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
      // every tile has the same canonical window, so the turning point of the mode, sqrt(2 max(n, m) + 1), visibly grows with the order
      ctx.setLineDash([5, 5]); ctx.strokeStyle = 'rgba(255,194,71,.8)'; ctx.lineWidth = 1.4;
      for (var mm = 0; mm < N; mm++) for (var nn = 0; nn < N; nn++) { ctx.beginPath(); ctx.arc((nn + 0.5) * TW, (mm + 0.5) * TW, R.rmaxOf(Math.max(nn, mm)) / Rt * TW / 2, 0, TAU); ctx.stroke(); }
      ctx.setLineDash([]);
      var fs = Math.round(TW * 0.12); ctx.font = '500 ' + fs + 'px "JetBrains Mono", monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
      for (mm = 0; mm < N; mm++) for (nn = 0; nn < N; nn++) {
        ctx.fillStyle = 'rgba(236,235,250,.72)'; ctx.fillText(nn + ',' + mm, nn * TW + 7, mm * TW + 6);
        ctx.fillStyle = 'rgba(255,194,71,.95)'; ctx.textAlign = 'right'; ctx.fillText(R.rmaxOf(Math.max(nn, mm)).toFixed(2), (nn + 1) * TW - 7, (mm + 1) * TW - fs - 6); ctx.textAlign = 'left';
      }
      ctx.lineWidth = 3; ctx.strokeStyle = '#fff'; ctx.strokeRect(sel[0] * TW + 1.5, sel[1] * TW + 1.5, TW - 3, TW - 3);
    }
    function describe() {
      var n = sel[0], m = sel[1], k = Math.max(n, m), r = R.rmaxOf(k), T = UI.tex;
      $('#zooName').innerHTML = T('\\Psi_{' + n + ',' + m + '}');
      $('#zooDesc').innerHTML = (n === 0 && m === 0)
        ? 'The zero-order mode is the Gaussian itself: one lobe, no nodal lines. Its turning point is ' + T('R_{\\max}=\\sqrt{2\\cdot0+1}=1') + '. A Brane with only this mode is a standard Gaussian splat.'
        : n + ' nodal line' + (n === 1 ? '' : 's') + ' across ' + T("x'") + ' and ' + m + ' across ' + T("y'") + ' make <b>' + ((n + 1) * (m + 1)) + ' lobes</b> of alternating sign. The mode oscillates out to its quantum turning point ' + T('R_{\\max}=\\sqrt{2\\max(' + n + ',' + m + ')+1}=\\sqrt{' + (2 * k + 1) + '}\\approx' + r.toFixed(2)) + ', and decays exponentially beyond it. The higher the order, the farther out it reaches.';
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
   * Rasterizer: d_max box, then the quantum turning point R_max
   * Three Branes of the paper's figure on a 12 x 12 pixel grid (image coordinates, y down). Nothing is rendered here: the grid is
   * discrete and the Brane field is continuous, so the demo only shows where each Brane stops mattering (its turning point).
   * ==================================================================== */
  function initRast() {
    var cv = $('#rastCanvas'); if (!cv) return;
    var ctx = cv.getContext('2d'), G = 12, W = cv.width, PW = W / G, T = UI.tex;
    var SPEC = [
      { name: 'A', cx: 5, cy: 6, s: 0.7, rho: 0, color: '#FFC247' },
      { name: 'B', cx: 8.5, cy: 3.5, s: 0.6, rho: -Math.PI / 4, color: '#FF7B54' },
      { name: 'C', cx: 1.5, cy: 1.5, s: 0.5, rho: 15 * Math.PI / 180, color: '#A6E36E' }
    ];
    var st = { D: 3, dmax: 3.5, px: 6, py: 6 };
    var RM = T('R_{\\max}'), DM = T('d_{\\max}');

    function pairInfo(sp, X, Y) { // canonical radius^2 and box test of Brane sp for the pixel centre (X, Y)
      var dx = X - sp.cx, dy = Y - sp.cy, co = Math.cos(sp.rho), si = Math.sin(sp.rho), xp = (co * dx + si * dy) / sp.s, yp = (-si * dx + co * dy) / sp.s;
      return { inBox: Math.abs(dx) < st.dmax && Math.abs(dy) < st.dmax, r2: xp * xp + yp * yp, cheb: Math.max(Math.abs(dx), Math.abs(dy)) };
    }

    function draw() {
      var D = st.D, Rr = R.rmaxOf(D), U = function (v) { return (v + 0.5) * PW; };
      ctx.fillStyle = '#06060D'; ctx.fillRect(0, 0, W, W);
      ctx.fillStyle = 'rgba(143,123,255,.05)';
      for (var j = 0; j < G; j++) for (var i = 0; i < G; i++) if ((i + j) % 2 === 0) ctx.fillRect(i * PW, j * PW, PW, PW);
      ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,.13)'; ctx.beginPath();
      for (i = 1; i < G; i++) { ctx.moveTo(i * PW, 0); ctx.lineTo(i * PW, W); ctx.moveTo(0, i * PW); ctx.lineTo(W, i * PW); }
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,.26)'; // pixel centres, where the field is sampled
      for (j = 0; j < G; j++) for (i = 0; i < G; i++) { ctx.beginPath(); ctx.arc(U(i), U(j), 2.2, 0, TAU); ctx.fill(); }

      // d_max box and target pixel
      ctx.lineWidth = 3; ctx.setLineDash([14, 6, 3, 6]); ctx.strokeStyle = '#2FE1FF';
      ctx.strokeRect(U(st.px - st.dmax), U(st.py - st.dmax), 2 * st.dmax * PW, 2 * st.dmax * PW); ctx.setLineDash([]);

      // the turning point of every Brane: solid when it counts for the selected pixel, dashed and dim when it does not
      SPEC.forEach(function (sp) {
        var q = pairInfo(sp, st.px, st.py), live = q.inBox && q.r2 <= Rr * Rr;
        ctx.lineWidth = live ? 4 : 2.5; ctx.strokeStyle = sp.color; ctx.globalAlpha = live ? 1 : 0.5; ctx.setLineDash(live ? [] : [11, 8]); ctx.beginPath();
        ctx.ellipse(U(sp.cx), U(sp.cy), Rr * sp.s * PW, Rr * sp.s * PW, sp.rho, 0, TAU); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
        cross(ctx, U(sp.cx), U(sp.cy), 6, '#fff');
        ctx.font = '600 15px "Space Grotesk", sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        var lbl = 'Brane ' + sp.name, tw = ctx.measureText(lbl).width + 12, ly = clamp(U(sp.cy) - Rr * sp.s * PW - 14, 14, W - 14);
        ctx.fillStyle = sp.color; ctx.globalAlpha = live ? 1 : 0.7; ctx.beginPath(); ctx.roundRect ? ctx.roundRect(U(sp.cx) - tw / 2, ly - 11, tw, 22, 7) : ctx.rect(U(sp.cx) - tw / 2, ly - 11, tw, 22); ctx.fill(); ctx.globalAlpha = 1;
        ctx.fillStyle = '#0A0A14'; ctx.fillText(lbl, U(sp.cx), ly + 1);
      });
      ctx.lineWidth = 3.5; ctx.strokeStyle = '#FF4D9D'; ctx.strokeRect(st.px * PW + 1.5, st.py * PW + 1.5, PW - 3, PW - 3);

      // counts over the whole 12 x 12 output
      var c1 = 0, c2 = 0, total = SPEC.length * G * G;
      SPEC.forEach(function (sp) { for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) { var q = pairInfo(sp, x, y); if (q.inBox) { c1++; if (q.r2 <= Rr * Rr) c2++; } } });
      var bar = function (label, n, color) { return '<div class="fn-row"><span>' + label + '</span><div class="fn-bar"><i style="width:' + (100 * n / total).toFixed(1) + '%;background:' + color + '"></i></div><b>' + n + '</b></div>'; };
      $('#funnel').innerHTML = bar('All Brane&ndash;pixel pairs', total, '#7C7B9C') + bar('Inside the ' + DM + ' box', c1, '#2FE1FF') + bar('Inside ' + RM, c2, '#FF4D9D');

      // the selected pixel
      var rows = SPEC.map(function (sp) {
        var q = pairInfo(sp, st.px, st.py), dot = '<i style="background:' + sp.color + '"></i>', name = '<b>Brane ' + sp.name + '</b> ';
        if (!q.inBox) return '<div class="pc-row">' + dot + '<div>' + name + '<span class="out">skipped by ' + DM + '</span>, its center is ' + q.cheb.toFixed(1) + ' pixels away (box half-side ' + st.dmax + ')</div></div>';
        var r = Math.sqrt(q.r2);
        if (q.r2 > Rr * Rr) return '<div class="pc-row">' + dot + '<div>' + name + '<span class="cut">skipped by ' + RM + '</span>, canonical radius ' + r.toFixed(2) + ' &gt; ' + Rr.toFixed(2) + '</div></div>';
        return '<div class="pc-row">' + dot + '<div>' + name + '<span class="ok">evaluated</span>, canonical radius ' + r.toFixed(2) + ' &le; ' + Rr.toFixed(2) + '</div></div>';
      });
      $('#pixelCard').innerHTML = '<h5>Target pixel (' + st.px + ', ' + st.py + ')</h5>' + rows.join('') + '<p class="pc-key">' + T('R_{\\max}=\\sqrt{2\\cdot' + D + '+1}=' + Rr.toFixed(2)) + ' at degree ' + D + '</p>';
    }

    $('#rastD').addEventListener('input', function (e) { st.dmax = +e.target.value; $('#rastDOut').textContent = st.dmax; draw(); });
    $('#rastN').addEventListener('input', function (e) { st.D = +e.target.value; $('#rastNOut').textContent = st.D; draw(); });
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
  function initVideo() {
    var box = $('#videoBox'), btn = $('#videoPlay'); if (!box || !btn) return;
    btn.addEventListener('click', function () {
      var id = box.dataset.yt, start = parseInt(box.dataset.start, 10) || 0, f = document.createElement('iframe');
      f.src = 'https://www.youtube-nocookie.com/embed/' + id + '?autoplay=1&rel=0&playsinline=1' + (start ? '&start=' + start : '');
      f.title = 'Resonant Brane Splatting, WACV presentation'; f.allowFullscreen = true; f.referrerPolicy = 'strict-origin-when-cross-origin';
      f.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
      btn.replaceWith(f);
    });
  }

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
    UI.typeset(document.body);
    [initHeroField, initBuilder, initZoo, initRast, initVideo, initMisc].concat(UI.inits).forEach(function (f) {
      try { f(); } catch (err) { if (window.console) console.error(f.name || 'init', err); }
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
