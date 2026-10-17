/*
 * Resonant Brane Splatting project page: the maths behind the demos.
 * Plain JavaScript, no dependencies, works in the browser (window.RBS) and in Node (module.exports).
 *
 *   - normalized Gaussian-Hermite modes and their derivatives
 *   - the Brane primitive and an additive splatting renderer, with d_max / R_max culling
 *   - a small differentiable fitter (analytic gradients, AdamW) for four primitives:
 *       GS (flat Gaussian), RBS (Brane), 2DGH (Hermite opacity), GB (Gaussian Billboards)
 *     mirroring the toy experiments of the paper (one primitive fits a patch).
 *
 * Nothing here is the trained network: these are illustrative re-implementations of the paper's equations.
 */
(function (root) {
  'use strict';

  /* ------------------------------------------------------------------ small helpers */
  var sigmoid = function (x) { return 1 / (1 + Math.exp(-x)); };
  var softplus = function (x) { return x > 20 ? x : Math.log1p(Math.exp(x)); };
  var clamp = function (x, a, b) { return x < a ? a : x > b ? b : x; };

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRng(seed) {
    var u = mulberry32(seed >>> 0), spare = null;
    return {
      uniform: function (a, b) { return a + (b - a) * u(); },
      normal: function (mean, std) { // Box-Muller
        if (spare !== null) { var s = spare; spare = null; return mean + std * s; }
        var a = 1 - u(), b = u(), r = Math.sqrt(-2 * Math.log(a));
        spare = r * Math.sin(2 * Math.PI * b);
        return mean + std * r * Math.cos(2 * Math.PI * b);
      },
      next: u
    };
  }

  /* ------------------------------------------------------------------ Gaussian-Hermite modes
   * Normalized physicist's Hermite polynomials  Hbar_n(z) = H_n(z) / sqrt(2^n n!).
   * Stable recurrence:  Hbar_n = sqrt(2/n) z Hbar_{n-1} - sqrt((n-1)/n) Hbar_{n-2}.
   * Derivative:         Hbar_n' = sqrt(2n) Hbar_{n-1}.
   * (Equivalent to Eq. 2 of the paper; the recurrence is just written for the normalized polynomials.) */
  var SQ_2_OVER = [0], SQ_PREV = [0], SQ_2N = [0];
  for (var i = 1; i <= 40; i++) { SQ_2_OVER[i] = Math.sqrt(2 / i); SQ_PREV[i] = Math.sqrt((i - 1) / i); SQ_2N[i] = Math.sqrt(2 * i); }

  function hermite(z, D, out) {
    out[0] = 1;
    if (D >= 1) out[1] = Math.SQRT2 * z;
    for (var n = 2; n <= D; n++) out[n] = SQ_2_OVER[n] * z * out[n - 1] - SQ_PREV[n] * out[n - 2];
    return out;
  }

  /** Roots of Hbar_n on [-R, R] (the nodal lines of a mode are the lines x' = root, y' = root). */
  var rootCache = {};
  function hermiteRoots(n) {
    if (n === 0) return [];
    if (rootCache[n]) return rootCache[n];
    var R = Math.sqrt(2 * n + 1) + 1, steps = 4000, buf = new Float64Array(n + 1), roots = [], prev = null, pz = -R;
    hermite(-R, n, buf); prev = buf[n];
    for (var s = 1; s <= steps; s++) {
      var z = -R + 2 * R * s / steps; hermite(z, n, buf); var v = buf[n];
      if (prev === 0 || (prev < 0) !== (v < 0)) {
        var a = pz, b = z, fa = prev;
        for (var it = 0; it < 50; it++) { var m = (a + b) / 2; hermite(m, n, buf); if ((buf[n] < 0) === (fa < 0)) { a = m; fa = buf[n]; } else b = m; }
        roots.push((a + b) / 2);
      }
      prev = v; pz = z;
    }
    rootCache[n] = roots;
    return roots;
  }

  /* ------------------------------------------------------------------ colors */
  /** Google's Turbo colormap (Apache-2.0), 33 samples of the original table, linearly interpolated. */
  var TURBO_LUT = [[0.1900,0.0718,0.2322],[0.2250,0.1635,0.4510],[0.2511,0.2524,0.6337],[0.2682,0.3382,0.7805],[0.2763,0.4212,0.8912],[0.2754,0.5011,0.9659],[0.2586,0.5796,0.9988],[0.2138,0.6589,0.9796],[0.1584,0.7355,0.9231],[0.1117,0.8057,0.8452],[0.0927,0.8655,0.7623],[0.1201,0.9119,0.6866],[0.1966,0.9490,0.5947],[0.3051,0.9770,0.4899],[0.4278,0.9942,0.3857],[0.5466,0.9991,0.2958],[0.6436,0.9900,0.2336],[0.7260,0.9647,0.2064],[0.8047,0.9245,0.2046],[0.8753,0.8727,0.2155],[0.9330,0.8124,0.2267],[0.9732,0.7468,0.2254],[0.9931,0.6741,0.2035],[0.9959,0.5870,0.1690],[0.9836,0.4929,0.1285],[0.9580,0.3996,0.0883],[0.9211,0.3149,0.0548],[0.8742,0.2453,0.0330],[0.8161,0.1846,0.0181],[0.7462,0.1310,0.0085],[0.6645,0.0844,0.0042],[0.5710,0.0447,0.0053],[0.4796,0.0158,0.0106]];
  function turbo(x) {
    x = clamp(x, 0, 1) * 32; var i = Math.min(31, Math.floor(x)), f = x - i, a = TURBO_LUT[i], b = TURBO_LUT[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }

  /** Illustrative mode colors (the demos do not run the trained network): every mode gets a seeded random hue and sign,
   *  and an amplitude A / (1 + s (n + m)) that shrinks with the mode's order, so that adding modes refines the picture
   *  instead of saturating it. A mode's color does not depend on D, so raising the degree only adds modes. */
  function modeColors(D, seed, opts) {
    opts = opts || {};
    var A = opts.A == null ? 1.0 : opts.A, s = opts.s == null ? 1.3 : opts.s, base = opts.base || [0.7, 0.68, 0.76];
    var L = (D + 1) * (D + 1), out = new Float64Array(L * 3);
    for (var n = 0; n <= D; n++) for (var m = 0; m <= D; m++) {
      var l = n * (D + 1) + m;
      if (n === 0 && m === 0) { out[0] = base[0]; out[1] = base[1]; out[2] = base[2]; continue; }
      var rng = makeRng((n * 100 + m) * 7919 + seed * 31 + 1), amp = A / (1 + s * (n + m)), t = turbo(rng.next()), sg = rng.next() < 0.5 ? -1 : 1;
      for (var c = 0; c < 3; c++) out[l * 3 + c] = sg * amp * (t[c] * 2 - 0.7);
    }
    return out;
  }

  /* ------------------------------------------------------------------ the Brane primitive and its renderer
   * brane = { cx, cy, sx, sy, rho, alpha, D, coef }       (N = M = D, L = (D+1)^2 modes, coef[l*3 + c], l = n*(D+1) + m)
   * Canonical coordinates (Eq. 11):  x' = ( cos(rho) dx + sin(rho) dy) / sx,  y' = (-sin(rho) dx + cos(rho) dy) / sy
   * Emission (Eq. 4 / 12):           alpha * exp(-(x'^2 + y'^2)/2) * sum_{n,m} c_{n,m} Hbar_n(x') Hbar_m(y')           */
  function rmaxOf(D) { return Math.sqrt(2 * D + 1); }

  /**
   * Splat branes additively on a size x size grid covering domain [x0,x1] x [y0,y1].
   * opts.cull: 'none' | 'rmax' (skip pixels whose canonical radius exceeds sqrt(2 D + 1))
   * opts.dmax: skip a brane for a pixel when |dx| or |dy| >= dmax (in domain units); Infinity by default.
   * opts.out : optional Float32Array(size*size*3) to reuse. Returns { img, evals } where evals counts brane-pixel evaluations.
   */
  function renderBranes(branes, size, dom, opts) {
    opts = opts || {};
    var x0 = dom[0], x1 = dom[1], y0 = dom[2], y1 = dom[3];
    var img = opts.out || new Float32Array(size * size * 3); img.fill(0);
    var cull = opts.cull || 'none', dmax = opts.dmax == null ? Infinity : opts.dmax, evals = 0;
    var sx_ = (x1 - x0) / size, sy_ = (y1 - y0) / size;
    for (var b = 0; b < branes.length; b++) {
      var br = branes[b], D = br.D, L1 = D + 1, co = Math.cos(br.rho), si = Math.sin(br.rho), coef = br.coef;
      var hx = new Float64Array(L1), hy = new Float64Array(L1), T = new Float64Array(L1 * 3), r2max = 2 * D + 1;
      var ixmin = 0, ixmax = size - 1, iymin = 0, iymax = size - 1;
      if (isFinite(dmax)) { // bounding box in pixel indices
        ixmin = Math.max(0, Math.ceil((br.cx - dmax - x0) / sx_ - 0.5)); ixmax = Math.min(size - 1, Math.floor((br.cx + dmax - x0) / sx_ - 0.5));
        iymin = Math.max(0, Math.ceil((br.cy - dmax - y0) / sy_ - 0.5)); iymax = Math.min(size - 1, Math.floor((br.cy + dmax - y0) / sy_ - 0.5));
      }
      for (var j = iymin; j <= iymax; j++) {
        var Y = y0 + (j + 0.5) * sy_, dy = Y - br.cy;
        for (var i = ixmin; i <= ixmax; i++) {
          var X = x0 + (i + 0.5) * sx_, dx = X - br.cx;
          if (isFinite(dmax) && (Math.abs(dx) >= dmax || Math.abs(dy) >= dmax)) continue;
          var xp = (co * dx + si * dy) / br.sx, yp = (-si * dx + co * dy) / br.sy, r2 = xp * xp + yp * yp;
          if (cull === 'rmax' && r2 > r2max) continue;
          if (r2 > 60) continue; // exp(-30) ~ 1e-13: below float precision of the picture
          evals++;
          var E = br.alpha * Math.exp(-0.5 * r2);
          hermite(xp, D, hx); hermite(yp, D, hy);
          var s0 = 0, s1 = 0, s2 = 0, p = (j * size + i) * 3;
          for (var n = 0; n <= D; n++) {
            var t0 = 0, t1 = 0, t2 = 0, base = n * L1 * 3;
            for (var m = 0; m <= D; m++) { var h = hy[m], q = base + m * 3; t0 += coef[q] * h; t1 += coef[q + 1] * h; t2 += coef[q + 2] * h; }
            var hn = hx[n]; s0 += hn * t0; s1 += hn * t1; s2 += hn * t2;
          }
          img[p] += E * s0; img[p + 1] += E * s1; img[p + 2] += E * s2;
        }
      }
    }
    for (var k = 0; k < img.length; k++) img[k] = img[k] < 0 ? 0 : img[k] > 1 ? 1 : img[k];
    return { img: img, evals: evals };
  }

  /** Float RGB (0..1) -> RGBA bytes (opaque), optionally over a background color. */
  function toRGBA(img, size, bg, outBuf) {
    var out = outBuf || new Uint8ClampedArray(size * size * 4), bgr = bg || [0, 0, 0];
    for (var p = 0, q = 0; p < size * size; p++, q += 4) {
      out[q] = img[p * 3] * 255; out[q + 1] = img[p * 3 + 1] * 255; out[q + 2] = img[p * 3 + 2] * 255; out[q + 3] = 255;
    }
    return out;
  }

  /* ------------------------------------------------------------------ differentiable fitter
   * Same parameterization and initialization as the toy experiments of the paper:
   *   center = sigmoid(c_raw) in (0,1)   scale = softplus(s_raw) + 0.015   rho = pi tanh(a_raw)   alpha = sigmoid(al_raw)
   *   GS : color = sigmoid(raw)                       image = clamp( sum_k alpha E color )
   *   RBS: c_00 = sigmoid(raw), c_nm = tanh(raw)       image = clamp( sum_k alpha E sum c_nm Hbar_n Hbar_m )
   *   2DGH: f = E sum a_nm Hbar_n Hbar_m, opacity alpha (1 - exp(-5 f^2)), one color per primitive
   *   GB : alpha E * bilinear color texture (gbRes x gbRes) sampled at (x'/gbSigma, y'/gbSigma), border clamped
   * Loss: MSE against the target.  Optimizer: AdamW (lr 0.08, weight decay 0.01).
   */
  var GEO = 6; // raw geometry parameters per primitive

  function createFitter(cfg) {
    var method = cfg.method, K = cfg.K || 1, D = method === 'GS' ? 0 : (cfg.D == null ? 0 : cfg.D), size = cfg.size, target = cfg.target;
    var lr = cfg.lr == null ? 0.08 : cfg.lr, wd = 0.01, GR = cfg.gbRes || 13, GSIG = cfg.gbSigma == null ? 0.5 : cfg.gbSigma;
    var L = (D + 1) * (D + 1), L1 = D + 1, rng = makeRng(cfg.seed == null ? 1 : cfg.seed);
    var NP = size * size;

    // per-primitive block layout
    var extra = method === 'GS' ? 3 : method === 'RBS' ? 3 * L : method === '2DGH' ? 3 + L : 3 * GR * GR;
    var stride = GEO + extra, P = K * stride;
    var theta = new Float64Array(P), grad = new Float64Array(P), m1 = new Float64Array(P), m2 = new Float64Array(P);

    function logit(p) { return Math.log(p / (1 - p)); }
    for (var k = 0; k < K; k++) {
      var o = k * stride;
      theta[o] = logit(clamp(rng.uniform(0, 1), 0.2, 0.8)); theta[o + 1] = logit(clamp(rng.uniform(0, 1), 0.2, 0.8));
      theta[o + 2] = rng.uniform(-2.4, -0.8); theta[o + 3] = rng.uniform(-2.4, -0.8);
      theta[o + 4] = rng.uniform(-1.5, 1.5);
      theta[o + 5] = rng.uniform(0, 1);
      var e = o + GEO;
      if (method === 'GS') { for (var c = 0; c < 3; c++) theta[e + c] = rng.normal(0, 0.8); }
      else if (method === 'RBS') {
        for (var l = 0; l < L; l++) for (c = 0; c < 3; c++) theta[e + l * 3 + c] = rng.normal(0, l === 0 ? 0.8 : 0.08);
      } else if (method === '2DGH') {
        var a0 = Math.max(0.01, 1 / (K * 2)); theta[o + 5] = logit(a0);
        for (c = 0; c < 3; c++) theta[e + c] = rng.normal(0, 0.8);
        for (l = 0; l < L; l++) theta[e + 3 + l] = l === 0 ? 1 : rng.normal(0, 0.05);
      } else { // GB
        var a1 = Math.max(0.01, 1 / (K * 2)); theta[o + 5] = logit(a1);
        for (var q = 0; q < 3 * GR * GR; q++) theta[e + q] = rng.normal(0, 0.3);
      }
    }

    // scratch
    var hx = new Float64Array(L1), hy = new Float64Array(L1);
    var T = new Float64Array(L1 * 3), U = new Float64Array(L1 * 3), Ts = new Float64Array(L1), Us = new Float64Array(L1);
    var coefAct = new Float64Array(L * 3), pred = new Float32Array(NP * 3), raw = new Float64Array(NP * 3), gimg = new Float64Array(NP * 3);
    var step = 0, history = new Float32Array(cfg.maxSteps || 4000), loss = NaN;

    function geometry(o) {
      var cx = sigmoid(theta[o]), cy = sigmoid(theta[o + 1]), sx = softplus(theta[o + 2]) + 0.015, sy = softplus(theta[o + 3]) + 0.015;
      var th = Math.tanh(theta[o + 4]), rho = Math.PI * th, al = sigmoid(theta[o + 5]);
      return { cx: cx, cy: cy, sx: sx, sy: sy, rho: rho, th: th, al: al, co: Math.cos(rho), si: Math.sin(rho) };
    }
    function activateCoef(e) { // RBS: base sigmoid, residual tanh
      for (var l = 0; l < L; l++) for (var c = 0; c < 3; c++) { var v = theta[e + l * 3 + c]; coefAct[l * 3 + c] = l === 0 ? sigmoid(v) : Math.tanh(v); }
    }

    /** One primitive's forward value at pixel position, plus (optionally) everything needed for the backward pass. */
    function forward() {
      raw.fill(0);
      for (var k = 0; k < K; k++) {
        var o = k * stride, e = o + GEO, g = geometry(o), colr = 0, colg = 0, colb = 0;
        if (method === 'RBS' || method === 'GS') activateCoef(e);
        if (method === '2DGH') { colr = sigmoid(theta[e]); colg = sigmoid(theta[e + 1]); colb = sigmoid(theta[e + 2]); }
        for (var j = 0; j < size; j++) {
          var Y = (j + 0.5) / size, dy = Y - g.cy;
          for (var i = 0; i < size; i++) {
            var X = (i + 0.5) / size, dx = X - g.cx;
            var xp = (g.co * dx + g.si * dy) / g.sx, yp = (-g.si * dx + g.co * dy) / g.sy, r2 = xp * xp + yp * yp;
            if (r2 > 60) continue;
            var E = Math.exp(-0.5 * r2), p = (j * size + i) * 3;
            if (method === 'RBS' || method === 'GS') {
              hermite(xp, D, hx); hermite(yp, D, hy);
              var s0 = 0, s1 = 0, s2 = 0;
              for (var n = 0; n <= D; n++) {
                var t0 = 0, t1 = 0, t2 = 0, base = n * L1 * 3;
                for (var m = 0; m <= D; m++) { var h = hy[m], q = base + m * 3; t0 += coefAct[q] * h; t1 += coefAct[q + 1] * h; t2 += coefAct[q + 2] * h; }
                var hn = hx[n]; s0 += hn * t0; s1 += hn * t1; s2 += hn * t2;
              }
              var a = g.al * E; raw[p] += a * s0; raw[p + 1] += a * s1; raw[p + 2] += a * s2;
            } else if (method === '2DGH') {
              hermite(xp, D, hx); hermite(yp, D, hy);
              var P_ = 0;
              for (n = 0; n <= D; n++) { var tt = 0; for (m = 0; m <= D; m++) tt += theta[e + 3 + n * L1 + m] * hy[m]; P_ += hx[n] * tt; }
              var f = E * P_, G = 1 - Math.exp(-5 * f * f), w = g.al * G;
              raw[p] += w * colr; raw[p + 1] += w * colg; raw[p + 2] += w * colb;
            } else { // GB
              var u = clamp(xp / GSIG, -1, 1), v = clamp(yp / GSIG, -1, 1), up = (u + 1) * 0.5 * (GR - 1), vp = (v + 1) * 0.5 * (GR - 1);
              var iu = Math.min(Math.floor(up), GR - 1), iv = Math.min(Math.floor(vp), GR - 1), iu1 = Math.min(iu + 1, GR - 1), iv1 = Math.min(iv + 1, GR - 1), fu = up - iu, fv = vp - iv;
              var w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv), w01 = (1 - fu) * fv, w11 = fu * fv, a2 = g.al * E;
              for (var c = 0; c < 3; c++) {
                var cb = e + c * GR * GR;
                var cv = w00 * sigmoid(theta[cb + iv * GR + iu]) + w10 * sigmoid(theta[cb + iv * GR + iu1]) + w01 * sigmoid(theta[cb + iv1 * GR + iu]) + w11 * sigmoid(theta[cb + iv1 * GR + iu1]);
                raw[p + c] += a2 * cv;
              }
            }
          }
        }
      }
    }

    function lossOnly() {
      forward();
      var s = 0;
      for (var p = 0; p < NP * 3; p++) { var v = raw[p] < 0 ? 0 : raw[p] > 1 ? 1 : raw[p]; var d = v - target[p]; s += d * d; }
      return s / (NP * 3);
    }

    function lossAndGrad() {
      forward();
      var s = 0, scale = 2 / (NP * 3);
      for (var p = 0; p < NP * 3; p++) {
        var r = raw[p], v = r < 0 ? 0 : r > 1 ? 1 : r, d = v - target[p];
        s += d * d; pred[p] = v; gimg[p] = (r >= 0 && r <= 1) ? scale * d : 0;
      }
      grad.fill(0);
      for (var k = 0; k < K; k++) {
        var o = k * stride, e = o + GEO, g = geometry(o);
        var dcx = 0, dcy = 0, dsx = 0, dsy = 0, drho = 0, dal = 0;
        var colr = 0, colg = 0, colb = 0;
        if (method === 'RBS' || method === 'GS') activateCoef(e);
        if (method === '2DGH') { colr = sigmoid(theta[e]); colg = sigmoid(theta[e + 1]); colb = sigmoid(theta[e + 2]); }
        for (var j = 0; j < size; j++) {
          var Y = (j + 0.5) / size, dy = Y - g.cy;
          for (var i = 0; i < size; i++) {
            var X = (i + 0.5) / size, dx = X - g.cx;
            var xp = (g.co * dx + g.si * dy) / g.sx, yp = (-g.si * dx + g.co * dy) / g.sy, r2 = xp * xp + yp * yp;
            if (r2 > 60) continue;
            var p3 = (j * size + i) * 3, g0 = gimg[p3], g1 = gimg[p3 + 1], g2 = gimg[p3 + 2];
            if (g0 === 0 && g1 === 0 && g2 === 0) continue;
            var E = Math.exp(-0.5 * r2), Gx = 0, Gy = 0, n, m, c;

            if (method === 'RBS' || method === 'GS') {
              hermite(xp, D, hx); hermite(yp, D, hy);
              var a = g.al * E, S0 = 0, S1 = 0, S2 = 0, Sx0 = 0, Sx1 = 0, Sx2 = 0, Sy0 = 0, Sy1 = 0, Sy2 = 0;
              for (n = 0; n <= D; n++) {
                var t0 = 0, t1 = 0, t2 = 0, base = n * L1 * 3;
                for (m = 0; m <= D; m++) { var h = hy[m], q = base + m * 3; t0 += coefAct[q] * h; t1 += coefAct[q + 1] * h; t2 += coefAct[q + 2] * h; }
                T[n * 3] = t0; T[n * 3 + 1] = t1; T[n * 3 + 2] = t2;
                var hn = hx[n]; S0 += hn * t0; S1 += hn * t1; S2 += hn * t2;
                if (n > 0) { var dh = SQ_2N[n] * hx[n - 1]; Sx0 += dh * t0; Sx1 += dh * t1; Sx2 += dh * t2; }
              }
              if (D > 0) {
                for (m = 0; m <= D; m++) {
                  var u0 = 0, u1 = 0, u2 = 0;
                  for (n = 0; n <= D; n++) { var hh = hx[n], q2 = (n * L1 + m) * 3; u0 += coefAct[q2] * hh; u1 += coefAct[q2 + 1] * hh; u2 += coefAct[q2 + 2] * hh; }
                  if (m > 0) { var dh2 = SQ_2N[m] * hy[m - 1]; Sy0 += dh2 * u0; Sy1 += dh2 * u1; Sy2 += dh2 * u2; }
                }
              }
              // coefficient gradients
              var ga0 = g0 * a, ga1 = g1 * a, ga2 = g2 * a;
              for (n = 0; n <= D; n++) for (m = 0; m <= D; m++) {
                var bb = hx[n] * hy[m], l = n * L1 + m, ce = e + l * 3, ac = coefAct[l * 3];
                // chain through activation: l = 0 -> sigmoid, else tanh
                if (l === 0) {
                  grad[ce] += ga0 * bb * ac * (1 - ac);
                  grad[ce + 1] += ga1 * bb * coefAct[1] * (1 - coefAct[1]);
                  grad[ce + 2] += ga2 * bb * coefAct[2] * (1 - coefAct[2]);
                } else {
                  var c0 = coefAct[l * 3], c1 = coefAct[l * 3 + 1], c2 = coefAct[l * 3 + 2];
                  grad[ce] += ga0 * bb * (1 - c0 * c0); grad[ce + 1] += ga1 * bb * (1 - c1 * c1); grad[ce + 2] += ga2 * bb * (1 - c2 * c2);
                }
              }
              dal += E * (g0 * S0 + g1 * S1 + g2 * S2);
              Gx = a * (g0 * (-xp * S0 + Sx0) + g1 * (-xp * S1 + Sx1) + g2 * (-xp * S2 + Sx2));
              Gy = a * (g0 * (-yp * S0 + Sy0) + g1 * (-yp * S1 + Sy1) + g2 * (-yp * S2 + Sy2));
            } else if (method === '2DGH') {
              hermite(xp, D, hx); hermite(yp, D, hy);
              var Pv = 0, Px = 0, Py = 0;
              for (n = 0; n <= D; n++) {
                var tt = 0; for (m = 0; m <= D; m++) tt += theta[e + 3 + n * L1 + m] * hy[m];
                Ts[n] = tt; Pv += hx[n] * tt; if (n > 0) Px += SQ_2N[n] * hx[n - 1] * tt;
              }
              for (m = 1; m <= D; m++) { var uu = 0; for (n = 0; n <= D; n++) uu += theta[e + 3 + n * L1 + m] * hx[n]; Py += SQ_2N[m] * hy[m - 1] * uu; }
              var f = E * Pv, ex = Math.exp(-5 * f * f), G = 1 - ex, gc = g0 * colr + g1 * colg + g2 * colb;
              var dLdf = g.al * gc * 10 * f * ex;
              for (n = 0; n <= D; n++) for (m = 0; m <= D; m++) grad[e + 3 + n * L1 + m] += dLdf * E * hx[n] * hy[m];
              grad[e] += g0 * g.al * G * colr * (1 - colr); grad[e + 1] += g1 * g.al * G * colg * (1 - colg); grad[e + 2] += g2 * g.al * G * colb * (1 - colb);
              dal += G * gc;
              Gx = dLdf * E * (-xp * Pv + Px); Gy = dLdf * E * (-yp * Pv + Py);
            } else { // GB
              var u = xp / GSIG, v = yp / GSIG, inU = Math.abs(u) < 1, inV = Math.abs(v) < 1;
              u = clamp(u, -1, 1); v = clamp(v, -1, 1);
              var up = (u + 1) * 0.5 * (GR - 1), vp = (v + 1) * 0.5 * (GR - 1);
              var iu = Math.min(Math.floor(up), GR - 1), iv = Math.min(Math.floor(vp), GR - 1), iu1 = Math.min(iu + 1, GR - 1), iv1 = Math.min(iv + 1, GR - 1), fu = up - iu, fv = vp - iv;
              var w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv), w01 = (1 - fu) * fv, w11 = fu * fv, a2 = g.al * E;
              var gcs = [g0, g1, g2], dcol_du = 0, dcol_dv = 0, sumgc = 0;
              for (c = 0; c < 3; c++) {
                var cb = e + c * GR * GR, s00 = sigmoid(theta[cb + iv * GR + iu]), s10 = sigmoid(theta[cb + iv * GR + iu1]), s01 = sigmoid(theta[cb + iv1 * GR + iu]), s11 = sigmoid(theta[cb + iv1 * GR + iu1]);
                var cv = w00 * s00 + w10 * s10 + w01 * s01 + w11 * s11, gg = gcs[c];
                grad[cb + iv * GR + iu] += gg * a2 * w00 * s00 * (1 - s00); grad[cb + iv * GR + iu1] += gg * a2 * w10 * s10 * (1 - s10);
                grad[cb + iv1 * GR + iu] += gg * a2 * w01 * s01 * (1 - s01); grad[cb + iv1 * GR + iu1] += gg * a2 * w11 * s11 * (1 - s11);
                dal += gg * E * cv; sumgc += gg * cv;
                dcol_du += gg * ((1 - fv) * (s10 - s00) + fv * (s11 - s01));
                dcol_dv += gg * ((1 - fu) * (s01 - s00) + fu * (s11 - s10));
              }
              var kk = (GR - 1) * 0.5 / GSIG;
              Gx = a2 * (-xp * sumgc + (inU ? kk * dcol_du : 0)); Gy = a2 * (-yp * sumgc + (inV ? kk * dcol_dv : 0));
            }
            // geometry chain: x', y' -> (cx, cy, sx, sy, rho)
            dcx += Gx * (-g.co / g.sx) + Gy * (g.si / g.sy);
            dcy += Gx * (-g.si / g.sx) + Gy * (-g.co / g.sy);
            dsx += Gx * (-xp / g.sx); dsy += Gy * (-yp / g.sy);
            drho += Gx * (yp * g.sy / g.sx) + Gy * (-xp * g.sx / g.sy);
          }
        }
        grad[o] += dcx * g.cx * (1 - g.cx);
        grad[o + 1] += dcy * g.cy * (1 - g.cy);
        grad[o + 2] += dsx * sigmoid(theta[o + 2]);
        grad[o + 3] += dsy * sigmoid(theta[o + 3]);
        grad[o + 4] += drho * Math.PI * (1 - g.th * g.th);
        grad[o + 5] += dal * g.al * (1 - g.al);
        if (method === 'GS') { // colors: sigmoid
          var cc = [sigmoid(theta[e]), sigmoid(theta[e + 1]), sigmoid(theta[e + 2])];
          // GS goes through the RBS branch (D = 0), whose coefficient gradient already includes the sigmoid chain.
          void cc;
        }
      }
      return s / (NP * 3);
    }

    function adamw(n) {
      var b1 = 0.9, b2 = 0.999, eps = 1e-8;
      for (var s = 0; s < n; s++) {
        loss = lossAndGrad();
        if (step < history.length) history[step] = loss;
        step++;
        var c1 = 1 - Math.pow(b1, step), c2 = 1 - Math.pow(b2, step);
        for (var i = 0; i < P; i++) {
          theta[i] *= (1 - lr * wd);
          m1[i] = b1 * m1[i] + (1 - b1) * grad[i]; m2[i] = b2 * m2[i] + (1 - b2) * grad[i] * grad[i];
          theta[i] -= lr * (m1[i] / c1) / (Math.sqrt(m2[i] / c2) + eps);
        }
      }
      return loss;
    }

    return {
      method: method, K: K, D: D, size: size, stride: stride, theta: theta, grad: grad,
      step: adamw,
      get loss() { return loss; },
      get steps() { return step; },
      history: history,
      lossOnly: lossOnly, lossAndGrad: lossAndGrad,
      image: function () { forward(); for (var p = 0; p < NP * 3; p++) { var v = raw[p]; pred[p] = v < 0 ? 0 : v > 1 ? 1 : v; } return pred; },
      prims: function () {
        var out = [];
        for (var k = 0; k < K; k++) { var g = geometry(k * stride); out.push({ cx: g.cx, cy: g.cy, sx: g.sx, sy: g.sy, rho: g.rho, alpha: g.al }); }
        return out;
      },
      paramCount: function () { // free parameters of the primitive set as they are used at render time
        return K * (method === 'GS' ? 9 : method === 'RBS' ? 6 + 3 * L : method === '2DGH' ? 6 + 3 + L - 1 : 6 + 3 * GR * GR);
      }
    };
  }

  /* ------------------------------------------------------------------ patches: resampling helpers (browser only use) */
  function mse(a, b) { var s = 0; for (var i = 0; i < a.length; i++) { var d = a[i] - b[i]; s += d * d; } return s / a.length; }
  function psnr(m) { return m <= 0 ? 99 : -10 * Math.log10(m); }

  var API = {
    hermite: hermite, hermiteRoots: hermiteRoots, turbo: turbo, modeColors: modeColors, rmaxOf: rmaxOf,
    renderBranes: renderBranes, toRGBA: toRGBA, createFitter: createFitter, makeRng: makeRng, mse: mse, psnr: psnr,
    sigmoid: sigmoid, softplus: softplus, clamp: clamp
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.RBS = API;
})(typeof window !== 'undefined' ? window : globalThis);
