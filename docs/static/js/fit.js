/*
 * Resonant Brane Splatting project page: live fitting playground (2/3).
 * Fits a small patch with one primitive of each kind, in the browser, with analytic gradients and AdamW
 * (the re-implementation in core.js, mirroring the toy experiments of the paper). It is not the trained network.
 */
(function () {
  'use strict';
  var U = window.RBSUI, R = window.RBS;
  if (!U || !R) return;
  var $ = U.$;
  var TAU = Math.PI * 2, PS = 32, VS = 192, MAXSTEPS = 1000, BUDGET_MS = 13;

  var MODELS = {
    gs: [
      { name: '1 Gaussian', sub: 'one flat color', steps: 500, cfg: { method: 'GS', K: 1 } },
      { name: '12 Gaussians', sub: 'flat colors, overlapping', steps: 500, cfg: { method: 'GS', K: 12 } },
      { name: '1 Brane', sub: 'N = M = 12, 169 modes', steps: 1000, cfg: { method: 'RBS', K: 1, D: 12 } }
    ],
    alt: [
      { name: '2DGH', sub: 'Hermite opacity, one color, N = M = 12', steps: 1000, cfg: { method: '2DGH', K: 1, D: 12 } },
      { name: 'Gaussian Billboard', sub: '13 × 13 color texture', steps: 1000, cfg: { method: 'GB', K: 1, gbRes: 13, gbSigma: 0.5 } },
      { name: 'RBS Brane', sub: 'N = M = 12, a color per mode', steps: 1000, cfg: { method: 'RBS', K: 1, D: 12 } }
    ]
  };
  var NOTES = {
    gs: 'Same optimizer for everyone: AdamW, learning rate 0.08, mean squared error, a random start. As in the paper&rsquo;s script, Gaussians train for 500 steps (they have converged by then) and the Brane for 1,000. A Brane carries 6 + 3\u00D7169 = 513 numbers and a Gaussian 9, so this shows what one primitive can express, not an equal-parameter match; the outcome depends on the patch and on the random start, so try a few. In RBS the network predicts those numbers from features, and the point is that a richer primitive needs far fewer overlapping splats per pixel.',
    alt: 'One primitive each (K = 1), same optimizer and 1,000 steps. 2DGH uses its Hermite modes only to reshape the opacity and keeps a single color. A Gaussian Billboard paints a 13\u00D713 color texture inside the footprint and samples it bilinearly. A Brane gives every mode its own color. Parameters: 177, 513 and 513. Outcomes depend on the patch and the random start.'
  };

  function initFit() {
    var root = $('#fit'); if (!root) return;
    var picker = $('#fitPicker'), pctx = picker.getContext('2d'), panelsEl = $('#fitPanels'), runBtn = $('#fitRun'), bar = $('#fitBar'), stepEl = $('#fitStep'), note = $('#fitNote'), footChk = $('#fitFoot'), hint = $('#fitPickHint');
    var src = document.createElement('canvas'), tmp = document.createElement('canvas'); tmp.width = tmp.height = PS;
    var tctx = tmp.getContext('2d'), tdata = tctx.createImageData(PS, PS);
    var st = { mode: 'gs', box: { x: 198, y: 129, s: 96 }, seed: 40, running: false, userPaused: false, visible: false, step: 0, perFrame: 2, fitters: [], target: null, panels: [], raf: 0, custom: false, dragging: false, ready: false };

    function setSource(img, w, h) {
      var k = Math.min(1, 480 / Math.max(w, h)), W = Math.max(8, Math.round(w * k)), H = Math.max(8, Math.round(h * k));
      src.width = W; src.height = H; src.getContext('2d').drawImage(img, 0, 0, W, H);
      picker.width = W; picker.height = H; picker.style.aspectRatio = W + ' / ' + H;
      var s = Math.round(0.2 * Math.min(W, H));
      st.box = st.custom ? { x: Math.round((W - s) / 2), y: Math.round((H - s) / 2), s: s } : { x: 198, y: 129, s: 96 };
      st.ready = true;
    }
    function clampBox() { st.box.x = Math.max(0, Math.min(src.width - st.box.s, Math.round(st.box.x))); st.box.y = Math.max(0, Math.min(src.height - st.box.s, Math.round(st.box.y))); }

    function drawPicker() {
      var b = st.box, W = picker.width, H = picker.height;
      pctx.drawImage(src, 0, 0);
      pctx.fillStyle = 'rgba(5,5,12,.55)';
      pctx.fillRect(0, 0, W, b.y); pctx.fillRect(0, b.y + b.s, W, H - b.y - b.s); pctx.fillRect(0, b.y, b.x, b.s); pctx.fillRect(b.x + b.s, b.y, W - b.x - b.s, b.s);
      pctx.lineWidth = 4; pctx.strokeStyle = 'rgba(0,0,0,.6)'; pctx.strokeRect(b.x, b.y, b.s, b.s);
      pctx.lineWidth = 2.5; pctx.strokeStyle = '#fff'; pctx.strokeRect(b.x, b.y, b.s, b.s);
    }
    function extract() {
      var c = document.createElement('canvas'); c.width = c.height = PS; var x = c.getContext('2d'); x.imageSmoothingEnabled = true; x.imageSmoothingQuality = 'high';
      x.drawImage(src, st.box.x, st.box.y, st.box.s, st.box.s, 0, 0, PS, PS);
      var d = x.getImageData(0, 0, PS, PS).data, t = new Float32Array(PS * PS * 3);
      for (var p = 0; p < PS * PS; p++) { t[p * 3] = d[p * 4] / 255; t[p * 3 + 1] = d[p * 4 + 1] / 255; t[p * 3 + 2] = d[p * 4 + 2] / 255; }
      return t;
    }

    function buildPanels() {
      var defs = [{ name: 'Target', sub: PS + ' × ' + PS + ' pixel patch' }].concat(MODELS[st.mode]);
      panelsEl.innerHTML = ''; st.panels = [];
      defs.forEach(function (d, i) {
        var el = document.createElement('div'); el.className = 'fp';
        el.innerHTML = '<canvas width="' + VS + '" height="' + VS + '" aria-label="' + d.name + '"></canvas><p class="fp-name">' + d.name + '<small>' + d.sub + '</small></p><p class="fp-stat">&nbsp;</p>';
        panelsEl.appendChild(el);
        st.panels.push({ el: el, cv: el.querySelector('canvas'), stat: el.querySelector('.fp-stat'), idx: i });
      });
    }
    function paint(panel, img, prims) {
      R.toRGBA(img, PS, null, tdata.data); tctx.putImageData(tdata, 0, 0);
      var c = panel.cv.getContext('2d'); c.imageSmoothingEnabled = false; c.drawImage(tmp, 0, 0, VS, VS);
      if (prims && footChk.checked) {
        c.lineWidth = 1.6; c.strokeStyle = 'rgba(255,255,255,.9)';
        prims.forEach(function (p) { c.beginPath(); c.ellipse(p.cx * VS, p.cy * VS, p.sx * VS, p.sy * VS, p.rho, 0, TAU); c.stroke(); });
      }
    }
    function redraw() {
      if (!st.target) return;
      paint(st.panels[0], st.target, null);
      var best = -1, bi = -1, vals = [];
      st.fitters.forEach(function (f, i) {
        var done = f.steps >= MODELS[st.mode][i].steps, img = (done && st.cache[i]) ? st.cache[i] : f.image(), p = (done && st.cacheP[i] != null) ? st.cacheP[i] : R.psnr(R.mse(img, st.target)); vals.push(p);
        if (done && !st.cache[i]) { st.cache[i] = Float32Array.from(img); st.cacheP[i] = p; }
        if (p > best) { best = p; bi = i; }
        paint(st.panels[i + 1], img, f.prims());
        st.panels[i + 1].stat.innerHTML = '<b>' + p.toFixed(1) + ' dB</b> &middot; ' + f.paramCount() + ' numbers';
      });
      st.panels.forEach(function (p, i) { p.el.classList.toggle('best', i > 0 && i - 1 === bi && st.step >= 60); p.el.classList.toggle('final', st.step >= MAXSTEPS); });
      bar.style.width = (100 * st.step / MAXSTEPS).toFixed(1) + '%';
      stepEl.textContent = 'step ' + st.step + ' / ' + MAXSTEPS;
    }

    function cancel() { st.running = false; cancelAnimationFrame(st.raf); }
    function reset() {
      cancel(); st.step = 0; st.cache = []; st.cacheP = []; st.target = extract();
      if (!st.panels.length || st.panels[0].mode !== st.mode) { buildPanels(); st.panels.forEach(function (p) { p.mode = st.mode; }); }
      st.fitters = MODELS[st.mode].map(function (m) { return R.createFitter(Object.assign({ size: PS, target: st.target, seed: st.seed, maxSteps: MAXSTEPS }, m.cfg)); });
      runBtn.textContent = 'Run'; redraw();
    }
    function frame() {
      if (!st.running) return;
      var t0 = performance.now(), n = Math.min(st.perFrame, MAXSTEPS - st.step);
      st.fitters.forEach(function (f, i) { var left = MODELS[st.mode][i].steps - f.steps; if (left > 0) f.step(Math.min(n, left)); });
      st.step += n; redraw();
      var dt = performance.now() - t0; st.perFrame = Math.max(1, Math.min(30, Math.round(0.5 * st.perFrame + 0.5 * n * BUDGET_MS / Math.max(dt, 0.5))));
      if (st.step >= MAXSTEPS) { st.running = false; runBtn.textContent = 'Run again'; return; }
      st.raf = requestAnimationFrame(frame);
    }
    function start() { if (!st.ready || st.running || st.dragging) return; st.running = true; runBtn.textContent = 'Pause'; st.raf = requestAnimationFrame(frame); }
    function pause(user) { if (!st.running) return; cancel(); runBtn.textContent = 'Resume'; if (user) st.userPaused = true; }
    function autoStart() { if (st.visible && !st.userPaused && !U.reduceMotion && st.step < MAXSTEPS) start(); }

    runBtn.addEventListener('click', function () {
      if (st.running) { pause(true); return; }
      st.userPaused = false; if (st.step >= MAXSTEPS) reset(); start();
    });
    $('#fitReset').addEventListener('click', function () { st.seed += 1; st.userPaused = false; reset(); start(); });
    footChk.addEventListener('change', redraw);
    U.bindSeg($('#fitMode'), 'm', function (v) { st.mode = v; note.innerHTML = NOTES[v]; reset(); autoStart(); });

    /* ---- picking a patch: drag the square, arrow keys, presets, own photo ---- */
    var resetTimer = 0;
    function boxChanged(restart) {
      clampBox(); drawPicker(); st.target = extract(); paint(st.panels[0], st.target, null);
      clearTimeout(resetTimer);
      if (restart) resetTimer = setTimeout(function () { reset(); autoStart(); }, 220);
    }
    function pointerBox(ev) {
      var r = picker.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * picker.width, y = (ev.clientY - r.top) / r.height * picker.height;
      st.box.x = x - st.box.s / 2; st.box.y = y - st.box.s / 2; boxChanged(false);
    }
    picker.addEventListener('pointerdown', function (ev) { if (!st.ready) return; st.dragging = true; pause(false); picker.classList.add('drag'); picker.setPointerCapture(ev.pointerId); pointerBox(ev); });
    picker.addEventListener('pointermove', function (ev) { if (st.dragging) pointerBox(ev); });
    function endDrag() { if (!st.dragging) return; st.dragging = false; picker.classList.remove('drag'); reset(); autoStart(); }
    picker.addEventListener('pointerup', endDrag); picker.addEventListener('pointercancel', endDrag);
    picker.tabIndex = 0;
    picker.addEventListener('keydown', function (ev) {
      var d = { ArrowLeft: [-12, 0], ArrowRight: [12, 0], ArrowUp: [0, -12], ArrowDown: [0, 12] }[ev.key]; if (!d) return;
      ev.preventDefault(); pause(false); st.box.x += d[0]; st.box.y += d[1]; boxChanged(true);
    });
    $('#fitPresets').addEventListener('click', function (ev) {
      var b = ev.target.closest('button[data-box]'); if (!b || !st.ready) return;
      var p = b.dataset.box.split(','); st.box.x = +p[0]; st.box.y = +p[1]; pause(false); boxChanged(false); reset(); autoStart();
    });
    $('#fitUpload').addEventListener('click', function () { $('#fitFile').click(); });
    $('#fitFile').addEventListener('change', function (ev) {
      var f = ev.target.files && ev.target.files[0]; if (!f || !/^image\//.test(f.type)) return;
      var url = URL.createObjectURL(f), im = new Image();
      im.onload = function () {
        st.custom = true; setSource(im, im.naturalWidth, im.naturalHeight); URL.revokeObjectURL(url);
        $('#fitPresets').hidden = true; hint.textContent = 'Your photo stays in your browser. Drag the square to choose a patch.';
        drawPicker(); st.userPaused = false; reset(); autoStart();
      };
      im.onerror = function () { URL.revokeObjectURL(url); hint.textContent = 'That file could not be read as an image.'; };
      im.src = url; ev.target.value = '';
    });

    note.innerHTML = NOTES.gs;
    var img = new Image();
    img.onload = function () { setSource(img, img.naturalWidth, img.naturalHeight); drawPicker(); reset(); U.onVisible(root, function (v) { st.visible = v; if (v) autoStart(); else pause(false); }, 0.35); };
    img.onerror = function () { hint.textContent = 'The sample photo could not be loaded.'; };
    img.src = window.RBS_PARROT || '';
  }

  U.inits.push(initFit);
})();
