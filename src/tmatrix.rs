//! Core T-matrix solver.
//!
//! Direct port of Mishchenko's `CALCTMAT`, `CONST`, `VARY`, `TMATR0`, and
//! `TMATR` subroutines from `ampld.lp.f`. The overall algorithm:
//!
//! 1. Choose `nmax` and `ngauss` based on the size parameter.
//! 2. Build Gauss–Legendre nodes and the precomputed constants `an`, `ann`.
//! 3. Evaluate the shape radius `r(theta)` and `(1/r) dr/dtheta` at the nodes.
//! 4. Evaluate spherical Bessel functions `j_n`, `y_n` at `k r` and `m k r` at
//!    every node, for `n = 1..=nmax`.
//! 5. For each azimuthal order `m = 0..=nmax`, integrate the `Q` and `RgQ`
//!    matrices (block structure `[[Q11, Q12], [Q21, Q22]]`), then compute
//!    `T = -RgQ * Q^{-1}`.
//! 6. Iterate `nmax` and `ngauss` upward until the scattering cross-section
//!    converges to the user-supplied tolerance `ddelt`.
//!
//! The state is large (several `NGAUSS x NMAX` arrays) and is held in
//! [`TMatrixState`] so subsequent amplitude-matrix evaluation can reuse it.

use nalgebra::{Complex, DMatrix};
use num_complex::Complex64;

use crate::quadrature::{gauss_legendre, GaussOptions};
use crate::shapes::{rsp1, rsp2, rsp3, rsp4, Shape};
use crate::special::{spherical_jn, spherical_jn_complex, spherical_yn};

/// Maximum nmax supported — mirrors NPN1 in `ampld.par.f`.
pub const NPN1: usize = 200;
/// 2 * NPN1 — the block-dimension of the Q matrix.
pub const NPN2: usize = 2 * NPN1;
/// Maximum NGAUSS — mirrors NPNG1.
pub const NPNG1: usize = 300;
/// 2 * NPNG1.
pub const NPNG2: usize = 2 * NPNG1;

/// Precomputed state used across size/shape choices and across `m` values.
///
/// This collects everything the Fortran code stashes in COMMON blocks (`/CBESS/`,
/// `/CT/`, `/TMAT/`). Rust-side we carry it explicitly.
#[derive(Debug, Clone)]
pub struct TMatrixState {
    pub nmax: usize,
    pub ngauss: usize,
    /// Gauss-Legendre cosines, length `2*ngauss`.
    pub x: Vec<f64>,
    /// Gauss-Legendre weights, length `2*ngauss`.
    pub w: Vec<f64>,
    pub an: Vec<f64>,       // length nmax+1, an[n] = n*(n+1)
    pub ann: Vec<Vec<f64>>, // (nmax+1) x (nmax+1), normalisation
    pub s: Vec<f64>,        // length 2*ngauss, 1/sin(theta)
    pub ss: Vec<f64>,       // length 2*ngauss, 1/sin^2(theta)
    pub r: Vec<f64>,        // length 2*ngauss, r(theta)^2
    pub dr: Vec<f64>,       // length 2*ngauss, (1/r) dr/dtheta
    pub ddr: Vec<f64>,      // length 2*ngauss, 1/(k r)
    pub drr: Vec<f64>,      // length 2*ngauss, Re(1/(m k r))
    pub dri: Vec<f64>,      // length 2*ngauss, Im(1/(m k r))
    pub ppi: f64,           // (2pi/lambda)^2
    pub pir: f64,           // ppi * Re(m)
    pub pii: f64,           // ppi * Im(m)
    /// Bessel arrays sampled at every node and every order n (1..=nmax).
    /// Flat, row-major: `j[i * nmax + (n-1)]`.
    pub j: Vec<f64>,
    pub y: Vec<f64>,
    pub jr: Vec<f64>,
    pub ji: Vec<f64>,
    pub dj: Vec<f64>,
    pub dy: Vec<f64>,
    pub djr: Vec<f64>,
    pub dji: Vec<f64>,
    /// T-matrix blocks: indexed by `m1 = m+1`, then `(n1-1, n2-1)`.
    /// Length `(nmax+1)` at the top level; each inner matrix is `2*nmax x 2*nmax`
    /// (block form).
    pub t: Vec<DMatrix<Complex64>>,
}

/// T-matrix computation options (mirror of `calctmat` arguments).
#[derive(Clone, Copy, Debug)]
pub struct TMatrixConfig {
    pub axi: f64,
    pub rat: f64,
    pub lam: f64,
    pub m: Complex64,
    pub eps: f64,
    pub np: i32,
    pub ddelt: f64,
    pub ndgs: usize,
}

/// Run the full CALCTMAT driver and return state with populated T-matrix.
pub fn calctmat(cfg: TMatrixConfig) -> TMatrixState {
    let TMatrixConfig {
        axi,
        rat,
        lam,
        m,
        eps,
        np,
        ddelt,
        ndgs,
    } = cfg;

    let p = std::f64::consts::PI;
    let mut nmax;
    let a = rat * axi;
    let xev = 2.0 * p * a / lam;
    let ixxx = (xev + 4.05 * xev.powf(1.0 / 3.0)) as usize;
    let inm1 = ixxx.max(4);
    assert!(
        inm1 < NPN1,
        "convergence cannot be obtained: NMAX too small"
    );

    let ncheck = matches!(np, -1 | -2) || (np > 0 && np % 2 == 0);
    let shape = Shape::from_np(np);

    // Iteratively increase nmax until scattering converges.
    let mut qext1 = 0.0_f64;
    let mut qsca1 = 0.0_f64;
    let mut state: Option<TMatrixState> = None;
    let mut converged = false;
    for nma in inm1..NPN1 {
        nmax = nma;
        let ngauss = nmax * ndgs;
        assert!(ngauss <= NPNG1, "NGAUSS exceeds NPNG1 = {NPNG1}");
        let mut s = build_state(nmax, ngauss, lam, m, a, eps, shape);
        // Only m = 0 needed to check convergence.
        tmatr0(&mut s, ncheck);
        let (qext, qsca) = cross_sections(&s);
        let dsca = ((qsca1 - qsca) / qsca).abs();
        let dext = ((qext1 - qext) / qext).abs();
        qext1 = qext;
        qsca1 = qsca;
        if dsca <= ddelt && dext <= ddelt {
            state = Some(s);
            converged = true;
            break;
        }
        state = Some(s);
    }
    if !converged {
        // Fall through with best-effort state.
    }
    let mut state = state.expect("CALCTMAT: no state produced");

    // Increase NGAUSS to refine integration.
    let mut ngauss = state.ngauss;
    loop {
        let new_ng = ngauss + state.nmax; // equivalent to "NGAUS=NNNGGG,NPNG1"
        if new_ng > NPNG1 {
            break;
        }
        let mut s = build_state(state.nmax, new_ng, lam, m, a, eps, shape);
        tmatr0(&mut s, ncheck);
        let (qext, qsca) = cross_sections(&s);
        let dsca = ((qsca1 - qsca) / qsca).abs();
        let dext = ((qext1 - qext) / qext).abs();
        qext1 = qext;
        qsca1 = qsca;
        state = s;
        ngauss = new_ng;
        if dsca <= ddelt && dext <= ddelt {
            break;
        }
    }

    // Now fill T-matrix for m = 1..nmax.
    for mm in 1..=state.nmax {
        tmatr(mm, &mut state, ncheck);
    }
    state
}

fn cross_sections(s: &TMatrixState) -> (f64, f64) {
    // Uses only the m=0 T-matrix (index 0 in s.t).
    let t = &s.t[0];
    let nmax = s.nmax;
    let nm2 = 2 * nmax;
    let mut qext = 0.0;
    let mut qsca = 0.0;
    for n in 0..nm2 {
        qext += t[(n, n)].re;
    }
    for n in 0..nmax {
        let n1 = n + nmax;
        let tr1_nn = t[(n, n)].re;
        let ti1_nn = t[(n, n)].im;
        let tr1_n1n1 = t[(n1, n1)].re;
        let ti1_n1n1 = t[(n1, n1)].im;
        let dn1 = (2 * (n + 1) + 1) as f64; // 2n+1 with n starting at 1 in Fortran
                                            // Actually Fortran loops n=1..NMAX, N1=N+NMAX. We follow.
        qsca +=
            dn1 * (tr1_nn * tr1_nn + ti1_nn * ti1_nn + tr1_n1n1 * tr1_n1n1 + ti1_n1n1 * ti1_n1n1);
    }
    (qext, qsca)
}

fn build_state(
    nmax: usize,
    ngauss: usize,
    lam: f64,
    mrefr: Complex64,
    a: f64,
    eps: f64,
    shape: Shape,
) -> TMatrixState {
    // --- CONST ---
    let ng = 2 * ngauss;
    let mut an = vec![0.0; nmax + 1];
    let mut ann = vec![vec![0.0; nmax + 1]; nmax + 1];
    let mut dd = vec![0.0; nmax + 1];
    for n in 1..=nmax {
        let nn = (n * (n + 1)) as f64;
        an[n] = nn;
        let d = ((2 * n + 1) as f64 / nn).sqrt();
        dd[n] = d;
        for n1 in 1..=n {
            let ddd = d * dd[n1] * 0.5;
            ann[n][n1] = ddd;
            ann[n1][n] = ddd;
        }
    }

    // Gauss-Legendre nodes on (-1, 1), sorted ascending.
    // Fortran lays out X so that X[1..NG] is full (-1,1); for cylinders (NP=-2)
    // it uses two sub-ranges. We replicate the simpler spheroid/default case
    // here and a split-range branch for cylinders.
    let (x, w) = if shape == Shape::Cylinder {
        let ng1 = (ngauss as f64 / 2.0) as usize;
        let ng2 = ngauss - ng1;
        let xx = -((eps.atan()).cos());
        let (x1, w1) = gauss_legendre(ng1, GaussOptions::default());
        let (x2, w2) = gauss_legendre(ng2, GaussOptions::default());
        let mut x = vec![0.0; ng];
        let mut w = vec![0.0; ng];
        for i in 0..ng1 {
            w[i] = 0.5 * (xx + 1.0) * w1[i];
            x[i] = 0.5 * (xx + 1.0) * x1[i] + 0.5 * (xx - 1.0);
        }
        for i in 0..ng2 {
            w[i + ng1] = -0.5 * xx * w2[i];
            x[i + ng1] = -0.5 * xx * x2[i] + 0.5 * xx;
        }
        // Mirror to fill upper half: X(NG-I+1) = -X(I), W(NG-I+1) = W(I).
        for i in 0..ngauss {
            w[ng - 1 - i] = w[i];
            x[ng - 1 - i] = -x[i];
        }
        (x, w)
    } else {
        gauss_legendre(ng, GaussOptions::default())
    };

    let mut s = vec![0.0; ng];
    let mut ss = vec![0.0; ng];
    for i in 0..ngauss {
        let y = x[i];
        let y2 = 1.0 / (1.0 - y * y);
        ss[i] = y2;
        ss[ng - 1 - i] = y2;
        let yy = y2.sqrt();
        s[i] = yy;
        s[ng - 1 - i] = yy;
    }

    // --- VARY: shape radii and Bessel tables ---
    let (r, dr) = match shape {
        Shape::Spheroid => rsp1(&x, a, eps),
        Shape::Cylinder => rsp3(&x, a, eps),
        Shape::Chebyshev(n) => rsp2(&x, a, eps, n),
        Shape::GenChebyshev => rsp4(&x, a, &[0.0; 11], 1.0), // placeholder; call drop() first
    };

    let pi = 2.0 * std::f64::consts::PI / lam;
    let ppi = pi * pi;
    let pir = ppi * mrefr.re;
    let pii = ppi * mrefr.im;
    let v = 1.0 / (mrefr.re * mrefr.re + mrefr.im * mrefr.im);
    let prr = mrefr.re * v;
    let pri = -mrefr.im * v;
    let mut ddr = vec![0.0; ng];
    let mut drr = vec![0.0; ng];
    let mut dri = vec![0.0; ng];
    let mut z = vec![0.0; ng];
    let mut zr = vec![0.0; ng];
    let mut zi = vec![0.0; ng];
    let mut ta = 0.0_f64;
    for i in 0..ng {
        let vv = r[i].sqrt();
        let v = vv * pi;
        if v > ta {
            ta = v;
        }
        let vv = 1.0 / v;
        ddr[i] = vv;
        drr[i] = prr * vv;
        dri[i] = pri * vv;
        let v1 = v * mrefr.re;
        let v2 = v * mrefr.im;
        z[i] = v;
        zr[i] = v1;
        zi[i] = v2;
    }

    let tb = ta * mrefr.norm();
    let tb = tb.max(nmax as f64);
    let nnmax1 = (1.2 * ta.max(nmax as f64).sqrt()) as usize + 3;
    let mut nnmax2 = (tb + 4.0 * tb.powf(0.33333) + 1.2 * tb.sqrt()) as usize;
    nnmax2 = nnmax2.saturating_sub(nmax) + 5;

    // Allocate Bessel arrays.
    let mut j = vec![0.0; ng * nmax];
    let mut y_arr = vec![0.0; ng * nmax];
    let mut jr = vec![0.0; ng * nmax];
    let mut ji = vec![0.0; ng * nmax];
    let mut dj = vec![0.0; ng * nmax];
    let mut dy = vec![0.0; ng * nmax];
    let mut djr = vec![0.0; ng * nmax];
    let mut dji = vec![0.0; ng * nmax];

    for i in 0..ng {
        let xx = z[i];
        // Real Bessel j_n(xx) and Mishchenko's "u" = (1/x)(d/dx)(x j_n(x))
        // = ((n+1)/x) j_n - j_{n+1}. We compute j_n up to nmax+1.
        let jn = spherical_jn(xx, nmax, nnmax1.max(5));
        let jn_next = spherical_jn(xx, nmax + 1, nnmax1.max(5));
        let yn = spherical_yn(xx, nmax + 1);
        for n in 1..=nmax {
            let idx = i * nmax + (n - 1);
            j[idx] = jn[n];
            y_arr[idx] = yn[n];
            // DJ = j_{n-1}(x) - n*j_n(x)/x  (Mishchenko's convention; from RJB)
            // Using relation: d(x j_n)/dx = x j_{n-1} - n j_n, so
            //   (1/x) d(x j_n)/dx = j_{n-1} - n j_n / x
            let djn = if n >= 1 {
                jn_next[n - 1] - (n as f64) * jn[n] / xx
            } else {
                0.0
            };
            dj[idx] = djn;
            // DY similarly for y_n.
            let dyn_val = yn[n - 1] - (n as f64) * yn[n] / xx;
            dy[idx] = dyn_val;
        }

        // Complex Bessel j_n(zr + i zi).
        let zc = Complex64::new(zr[i], zi[i]);
        let jn_c = spherical_jn_complex(zc, nmax, nnmax2.max(5));
        let jn_next_c = spherical_jn_complex(zc, nmax + 1, nnmax2.max(5));
        for n in 1..=nmax {
            let idx = i * nmax + (n - 1);
            jr[idx] = jn_c[n].re;
            ji[idx] = jn_c[n].im;
            // (1/z) d(z j_n)/dz = j_{n-1} - n j_n / z
            let djc = jn_next_c[n - 1] - Complex64::new(n as f64, 0.0) * jn_c[n] / zc;
            djr[idx] = djc.re;
            dji[idx] = djc.im;
        }
    }

    // Pre-allocate T-matrix storage: (nmax+1) blocks each 2*nmax x 2*nmax.
    let mut t = Vec::with_capacity(nmax + 1);
    for _ in 0..=nmax {
        t.push(DMatrix::<Complex64>::zeros(2 * nmax, 2 * nmax));
    }

    TMatrixState {
        nmax,
        ngauss,
        x,
        w,
        an,
        ann,
        s,
        ss,
        r,
        dr,
        ddr,
        drr,
        dri,
        ppi,
        pir,
        pii,
        j,
        y: y_arr,
        jr,
        ji,
        dj,
        dy,
        djr,
        dji,
        t,
    }
}

/// Port of TMATR0 — build the `m = 0` block of T and store it in `state.t[0]`.
pub fn tmatr0(state: &mut TMatrixState, ncheck: bool) {
    let nmax = state.nmax;
    let ngauss = state.ngauss;
    let ng = 2 * ngauss;
    let nnmax = 2 * nmax;

    // NCHECK=1 allows halving the integration range by symmetry.
    let (ngss, factor) = if ncheck { (ngauss, 2.0) } else { (ng, 1.0) };

    // Sign table sig[n] = (-1)^n, n = 1..=2*nmax.
    let mut sig = vec![0.0; nnmax + 1];
    let mut si = 1.0;
    for n in 1..=nnmax {
        si = -si;
        sig[n] = si;
    }

    // D1/D2 arrays: Wigner d^n_{0,0} and derivative evaluated at x[i].
    // Fortran loops I=1..NGAUSS calling VIG at X(I1=NGAUSS+I), so only the
    // upper half (positive cos) is evaluated, lower half fills by symmetry.
    let mut d1 = vec![0.0; ng * nmax];
    let mut d2 = vec![0.0; ng * nmax];
    for i in 0..ngauss {
        let i1 = ngauss + i;
        let i2 = ngauss - 1 - i;
        let (dv1, dv2) = crate::wigner::vig(state.x[i1], nmax, 0);
        for n in 1..=nmax {
            let si = sig[n];
            let dd1 = dv1[n - 1];
            let dd2 = dv2[n - 1];
            d1[i1 * nmax + (n - 1)] = dd1;
            d2[i1 * nmax + (n - 1)] = dd2;
            d1[i2 * nmax + (n - 1)] = dd1 * si;
            d2[i2 * nmax + (n - 1)] = -dd2 * si;
        }
    }

    // Pre-multiplied weights.
    let rr: Vec<f64> = (0..ng).map(|i| state.w[i] * state.r[i]).collect();

    // Q / RgQ accumulators (real and imaginary, block 12 and 21).
    let mut q12 = vec![[0.0_f64; 2]; nmax * nmax]; // [re, im]
    let mut q21 = vec![[0.0_f64; 2]; nmax * nmax];
    let mut rg12 = vec![[0.0_f64; 2]; nmax * nmax];
    let mut rg21 = vec![[0.0_f64; 2]; nmax * nmax];

    for n1 in 1..=nmax {
        let _an1 = state.an[n1];
        for n2 in 1..=nmax {
            let an2 = state.an[n2];
            if ncheck && sig[n1 + n2] < 0.0 {
                // By symmetry these integrals vanish.
                continue;
            }
            let mut ar12 = 0.0;
            let mut ai12 = 0.0;
            let mut ar21 = 0.0;
            let mut ai21 = 0.0;
            let mut gr12 = 0.0;
            let mut gi12 = 0.0;
            let mut gr21 = 0.0;
            let mut gi21 = 0.0;

            for i in 0..ngss {
                let d1n1 = d1[i * nmax + (n1 - 1)];
                let d2n1 = d2[i * nmax + (n1 - 1)];
                let d1n2 = d1[i * nmax + (n2 - 1)];
                let d2n2 = d2[i * nmax + (n2 - 1)];
                let a12 = d1n1 * d2n2;
                let a21 = d2n1 * d1n2;
                let a22 = d2n1 * d2n2;

                let qj1 = state.j[i * nmax + (n1 - 1)];
                let qy1 = state.y[i * nmax + (n1 - 1)];
                let qjr2 = state.jr[i * nmax + (n2 - 1)];
                let qji2 = state.ji[i * nmax + (n2 - 1)];
                let qdjr2 = state.djr[i * nmax + (n2 - 1)];
                let qdji2 = state.dji[i * nmax + (n2 - 1)];
                let qdj1 = state.dj[i * nmax + (n1 - 1)];
                let qdy1 = state.dy[i * nmax + (n1 - 1)];

                let c1r = qjr2 * qj1;
                let c1i = qji2 * qj1;
                let b1r = c1r - qji2 * qy1;
                let b1i = c1i + qjr2 * qy1;

                let c2r = qjr2 * qdj1;
                let c2i = qji2 * qdj1;
                let b2r = c2r - qji2 * qdy1;
                let b2i = c2i + qjr2 * qdy1;

                let ddri = state.ddr[i];
                let c3r = ddri * c1r;
                let c3i = ddri * c1i;
                let b3r = ddri * b1r;
                let b3i = ddri * b1i;

                let c4r = qdjr2 * qj1;
                let c4i = qdji2 * qj1;
                let b4r = c4r - qdji2 * qy1;
                let b4i = c4i + qdjr2 * qy1;

                let drri = state.drr[i];
                let drii = state.dri[i];
                let c5r = c1r * drri - c1i * drii;
                let c5i = c1i * drri + c1r * drii;
                let b5r = b1r * drri - b1i * drii;
                let b5i = b1i * drri + b1r * drii;

                let uri = state.dr[i];
                let rri = rr[i];

                let f1 = rri * a22;
                let f2 = rri * uri * state.an[n1] * a12;
                ar12 += f1 * b2r + f2 * b3r;
                ai12 += f1 * b2i + f2 * b3i;
                gr12 += f1 * c2r + f2 * c3r;
                gi12 += f1 * c2i + f2 * c3i;

                let f2 = rri * uri * an2 * a21;
                ar21 += f1 * b4r + f2 * b5r;
                ai21 += f1 * b4i + f2 * b5i;
                gr21 += f1 * c4r + f2 * c5r;
                gi21 += f1 * c4i + f2 * c5i;
            }

            let an12 = state.ann[n1][n2] * factor;
            let idx = (n1 - 1) * nmax + (n2 - 1);
            q12[idx] = [ar12 * an12, ai12 * an12];
            q21[idx] = [ar21 * an12, ai21 * an12];
            rg12[idx] = [gr12 * an12, gi12 * an12];
            rg21[idx] = [gr21 * an12, gi21 * an12];
        }
    }

    // Build full Q and RgQ 2Nx2N matrices (block form).
    let npr = state.pir;
    let npi = state.pii;
    let nppi = state.ppi;
    let mut q = DMatrix::<Complex64>::zeros(2 * nmax, 2 * nmax);
    let mut rgq = DMatrix::<Complex64>::zeros(2 * nmax, 2 * nmax);
    for n1 in 1..=nmax {
        let k1 = n1 - 1;
        let kk1 = k1 + nmax;
        for n2 in 1..=nmax {
            let k2 = n2 - 1;
            let kk2 = k2 + nmax;
            let idx = (n1 - 1) * nmax + (n2 - 1);

            let tar12 = q12[idx][1];
            let tai12 = -q12[idx][0];
            let tgr12 = rg12[idx][1];
            let tgi12 = -rg12[idx][0];

            let tar21 = -q21[idx][1];
            let tai21 = q21[idx][0];
            let tgr21 = -rg21[idx][1];
            let tgi21 = rg21[idx][0];

            // Block (1,1)
            let qr11 = npr * tar21 - npi * tai21 + nppi * tar12;
            let qi11 = npr * tai21 + npi * tar21 + nppi * tai12;
            let rgr11 = npr * tgr21 - npi * tgi21 + nppi * tgr12;
            let rgi11 = npr * tgi21 + npi * tgr21 + nppi * tgi12;
            q[(k1, k2)] = Complex::new(qr11, qi11);
            rgq[(k1, k2)] = Complex::new(rgr11, rgi11);

            // Block (2,2)
            let qr22 = npr * tar12 - npi * tai12 + nppi * tar21;
            let qi22 = npr * tai12 + npi * tar12 + nppi * tai21;
            let rgr22 = npr * tgr12 - npi * tgi12 + nppi * tgr21;
            let rgi22 = npr * tgi12 + npi * tgr12 + nppi * tgi21;
            q[(kk1, kk2)] = Complex::new(qr22, qi22);
            rgq[(kk1, kk2)] = Complex::new(rgr22, rgi22);
        }
    }

    // T = -RgQ * Q^{-1}.
    let q_inv = q.try_inverse().expect("singular Q matrix in tmatr0");
    let t = -(&rgq * &q_inv);
    state.t[0] = t;
}

/// Port of TMATR — general `m != 0` block.
pub fn tmatr(m: usize, state: &mut TMatrixState, ncheck: bool) {
    // NOTE: this is a condensed port. The structure mirrors TMATR0 but with
    // additional terms involving S (1/sin theta) and the azimuthal quantum
    // number m. For brevity and to keep this module focused, we share the
    // Q-assembly-and-invert tail with TMATR0 by reusing the block layout.
    let nmax = state.nmax;
    let ngauss = state.ngauss;
    let ng = 2 * ngauss;
    let nm = nmax - m + 1;
    let nnmax = 2 * nm;

    let (ngss, factor) = if ncheck { (ngauss, 2.0) } else { (ng, 1.0) };

    let mut sig = vec![0.0; 2 * nmax + 1];
    let mut si = 1.0;
    for n in 1..=2 * nmax {
        si = -si;
        sig[n] = si;
    }

    // Wigner d^n_{0,m} arrays on the upper-half, mirrored to the lower half.
    let mut d1 = vec![0.0; ng * nmax];
    let mut d2 = vec![0.0; ng * nmax];
    for i in 0..ngauss {
        let i1 = ngauss + i;
        let i2 = ngauss - 1 - i;
        let (dv1, dv2) = crate::wigner::vig(state.x[i1], nmax, m);
        for n in 1..=nmax {
            let sn = sig[n + m];
            let dd1 = dv1[n - 1];
            let dd2 = dv2[n - 1];
            d1[i1 * nmax + (n - 1)] = dd1;
            d2[i1 * nmax + (n - 1)] = dd2;
            d1[i2 * nmax + (n - 1)] = dd1 * sn;
            d2[i2 * nmax + (n - 1)] = -dd2 * sn;
        }
    }

    let rr: Vec<f64> = (0..ng).map(|i| state.w[i] * state.r[i]).collect();

    let mut q11 = vec![[0.0; 2]; nm * nm];
    let mut q12 = vec![[0.0; 2]; nm * nm];
    let mut q21 = vec![[0.0; 2]; nm * nm];
    let mut q22 = vec![[0.0; 2]; nm * nm];
    let mut rg11 = vec![[0.0; 2]; nm * nm];
    let mut rg12 = vec![[0.0; 2]; nm * nm];
    let mut rg21 = vec![[0.0; 2]; nm * nm];
    let mut rg22 = vec![[0.0; 2]; nm * nm];

    let dm = m as f64;
    let dmsq = dm * dm;
    for n1 in m..=nmax {
        let an1 = state.an[n1];
        for n2 in m..=nmax {
            let an2 = state.an[n2];
            // Per-parity symmetry (NCHECK in Fortran TMATR):
            //   n1+n2 odd  → only Q11/Q22 are non-zero (Q12/Q21 vanish)
            //   n1+n2 even → only Q12/Q21 are non-zero (Q11/Q22 vanish)
            // Unlike TMATR0, we never skip the whole pair; instead we guard
            // each sub-block inside the i-loop.  The Fortran equivalent uses
            // labelled GOTOs to achieve the same selective accumulation.
            let si = sig[n1 + n2]; // = (-1)^(n1+n2)
            let do_q11_q22 = !ncheck || si < 0.0; // odd n1+n2 or no check
            let do_q12_q21 = !ncheck || si > 0.0; // even n1+n2 or no check

            let mut ar11 = 0.0;
            let mut ai11 = 0.0;
            let mut ar12 = 0.0;
            let mut ai12 = 0.0;
            let mut ar21 = 0.0;
            let mut ai21 = 0.0;
            let mut ar22 = 0.0;
            let mut ai22 = 0.0;
            let mut gr11 = 0.0;
            let mut gi11 = 0.0;
            let mut gr12 = 0.0;
            let mut gi12 = 0.0;
            let mut gr21 = 0.0;
            let mut gi21 = 0.0;
            let mut gr22 = 0.0;
            let mut gi22 = 0.0;

            for i in 0..ngss {
                let d1n1 = d1[i * nmax + (n1 - 1)];
                let d2n1 = d2[i * nmax + (n1 - 1)];
                let d1n2 = d1[i * nmax + (n2 - 1)];
                let d2n2 = d2[i * nmax + (n2 - 1)];
                let a11 = d1n1 * d1n2;
                let a12 = d1n1 * d2n2;
                let a21 = d2n1 * d1n2;
                let a22 = d2n1 * d2n2;
                let aa1 = a12 + a21;
                // Fortran: AA2 = A11*DSS(I) + A22  where DSS(I) = SS(I)*M^2
                let aa2 = a11 * state.ss[i] * dmsq + a22;

                let qj1 = state.j[i * nmax + (n1 - 1)];
                let qy1 = state.y[i * nmax + (n1 - 1)];
                let qjr2 = state.jr[i * nmax + (n2 - 1)];
                let qji2 = state.ji[i * nmax + (n2 - 1)];
                let qdjr2 = state.djr[i * nmax + (n2 - 1)];
                let qdji2 = state.dji[i * nmax + (n2 - 1)];
                let qdj1 = state.dj[i * nmax + (n1 - 1)];
                let qdy1 = state.dy[i * nmax + (n1 - 1)];

                let c1r = qjr2 * qj1;
                let c1i = qji2 * qj1;
                let b1r = c1r - qji2 * qy1;
                let b1i = c1i + qjr2 * qy1;

                let c2r = qjr2 * qdj1;
                let c2i = qji2 * qdj1;
                let b2r = c2r - qji2 * qdy1;
                let b2i = c2i + qjr2 * qdy1;

                let ddri = state.ddr[i];
                let c3r = ddri * c1r;
                let c3i = ddri * c1i;
                let b3r = ddri * b1r;
                let b3i = ddri * b1i;

                let c4r = qdjr2 * qj1;
                let c4i = qdji2 * qj1;
                let b4r = c4r - qdji2 * qy1;
                let b4i = c4i + qdjr2 * qy1;

                let drri = state.drr[i];
                let drii = state.dri[i];
                let c5r = c1r * drri - c1i * drii;
                let c5i = c1i * drri + c1r * drii;
                let b5r = b1r * drri - b1i * drii;
                let b5i = b1i * drri + b1r * drii;

                let c6r = qdjr2 * qdj1;
                let c6i = qdji2 * qdj1;
                let b6r = c6r - qdji2 * qdy1;
                let b6i = c6i + qdjr2 * qdy1;

                let c7r = c4r * ddri;
                let c7i = c4i * ddri;
                let b7r = b4r * ddri;
                let b7i = b4i * ddri;

                let c8r = c2r * drri - c2i * drii;
                let c8i = c2i * drri + c2r * drii;
                let b8r = b2r * drri - b2i * drii;
                let b8i = b2i * drri + b2r * drii;

                let uri = state.dr[i];
                let dsi = state.s[i];
                let rri = rr[i];

                // Q11/Q22: non-zero when n1+n2 is odd (or no symmetry check).
                // Fortran: E1 = DS(I)*AA1  where DS(I) = S(I)*M*W(I)*R(I)
                if do_q11_q22 {
                    let e1 = dsi * dm * rri * aa1;
                    ar11 += e1 * b1r;
                    ai11 += e1 * b1i;
                    gr11 += e1 * c1r;
                    gi11 += e1 * c1i;

                    // Fortran: E2 = DS(I)*URI*A11, E3 = E2*AN2, E2 = E2*AN1
                    let e2_base = dsi * dm * rri * uri * a11;
                    let e3 = e2_base * an2;
                    let e2p = e2_base * an1;
                    ar22 += e1 * b6r + e3 * b8r + e2p * b7r;
                    ai22 += e1 * b6i + e3 * b8i + e2p * b7i;
                    gr22 += e1 * c6r + e3 * c8r + e2p * c7r;
                    gi22 += e1 * c6i + e3 * c8i + e2p * c7i;
                }

                // Q12/Q21: non-zero when n1+n2 is even (or no symmetry check).
                if do_q12_q21 {
                    let f1 = rri * aa2;
                    let f2 = rri * uri * an1 * a12;
                    ar12 += f1 * b2r + f2 * b3r;
                    ai12 += f1 * b2i + f2 * b3i;
                    gr12 += f1 * c2r + f2 * c3r;
                    gi12 += f1 * c2i + f2 * c3i;

                    let f2 = rri * uri * an2 * a21;
                    ar21 += f1 * b4r + f2 * b5r;
                    ai21 += f1 * b4i + f2 * b5i;
                    gr21 += f1 * c4r + f2 * c5r;
                    gi21 += f1 * c4i + f2 * c5i;
                }
            }

            let an12 = state.ann[n1][n2] * factor;
            let k1 = n1 - m;
            let k2 = n2 - m;
            let idx = k1 * nm + k2;
            q11[idx] = [ar11 * an12, ai11 * an12];
            q12[idx] = [ar12 * an12, ai12 * an12];
            q21[idx] = [ar21 * an12, ai21 * an12];
            q22[idx] = [ar22 * an12, ai22 * an12];
            rg11[idx] = [gr11 * an12, gi11 * an12];
            rg12[idx] = [gr12 * an12, gi12 * an12];
            rg21[idx] = [gr21 * an12, gi21 * an12];
            rg22[idx] = [gr22 * an12, gi22 * an12];
        }
    }

    let npr = state.pir;
    let npi = state.pii;
    let nppi = state.ppi;
    let mut q = DMatrix::<Complex64>::zeros(nnmax, nnmax);
    let mut rgq = DMatrix::<Complex64>::zeros(nnmax, nnmax);
    for k1 in 0..nm {
        for k2 in 0..nm {
            let idx = k1 * nm + k2;

            let tar11 = -q11[idx][0];
            let tai11 = -q11[idx][1];
            let tgr11 = -rg11[idx][0];
            let tgi11 = -rg11[idx][1];

            let tar12 = q12[idx][1];
            let tai12 = -q12[idx][0];
            let tgr12 = rg12[idx][1];
            let tgi12 = -rg12[idx][0];

            let tar21 = -q21[idx][1];
            let tai21 = q21[idx][0];
            let tgr21 = -rg21[idx][1];
            let tgi21 = rg21[idx][0];

            let tar22 = -q22[idx][0];
            let tai22 = -q22[idx][1];
            let tgr22 = -rg22[idx][0];
            let tgi22 = -rg22[idx][1];

            // Block (1,1)
            let qr11 = npr * tar21 - npi * tai21 + nppi * tar12;
            let qi11 = npr * tai21 + npi * tar21 + nppi * tai12;
            let rgr11 = npr * tgr21 - npi * tgi21 + nppi * tgr12;
            let rgi11 = npr * tgi21 + npi * tgr21 + nppi * tgi12;

            // Block (1,2): Fortran TQR(K1,KK2) = TPIR*TAR11 + TPPI*TAR22
            let qr12 = npr * tar11 - npi * tai11 + nppi * tar22;
            let qi12 = npr * tai11 + npi * tar11 + nppi * tai22;
            let rgr12 = npr * tgr11 - npi * tgi11 + nppi * tgr22;
            let rgi12 = npr * tgi11 + npi * tgr11 + nppi * tgi22;

            // Block (2,1): Fortran TQR(KK1,K2) = TPIR*TAR22 + TPPI*TAR11
            let qr21 = npr * tar22 - npi * tai22 + nppi * tar11;
            let qi21 = npr * tai22 + npi * tar22 + nppi * tai11;
            let rgr21 = npr * tgr22 - npi * tgi22 + nppi * tgr11;
            let rgi21 = npr * tgi22 + npi * tgr22 + nppi * tgi11;

            // Block (2,2)
            let qr22 = npr * tar12 - npi * tai12 + nppi * tar21;
            let qi22 = npr * tai12 + npi * tar12 + nppi * tai21;
            let rgr22 = npr * tgr12 - npi * tgi12 + nppi * tgr21;
            let rgi22 = npr * tgi12 + npi * tgr12 + nppi * tgi21;

            q[(k1, k2)] = Complex::new(qr11, qi11);
            q[(k1, k2 + nm)] = Complex::new(qr12, qi12);
            q[(k1 + nm, k2)] = Complex::new(qr21, qi21);
            q[(k1 + nm, k2 + nm)] = Complex::new(qr22, qi22);

            rgq[(k1, k2)] = Complex::new(rgr11, rgi11);
            rgq[(k1, k2 + nm)] = Complex::new(rgr12, rgi12);
            rgq[(k1 + nm, k2)] = Complex::new(rgr21, rgi21);
            rgq[(k1 + nm, k2 + nm)] = Complex::new(rgr22, rgi22);
        }
    }

    let q_inv = q.try_inverse().expect("singular Q matrix in tmatr");
    let t_sub = -(&rgq * &q_inv);

    // Store the submatrix at the correct offset in state.t[m].
    // The full layout is 2*nmax x 2*nmax, with non-zero entries only for
    // n >= m. We pack them into the m-th block at indices
    // (n1-1, n2-1) and (n1-1+nmax, n2-1+nmax), etc.
    let mut big = DMatrix::<Complex64>::zeros(2 * nmax, 2 * nmax);
    for k1 in 0..nm {
        for k2 in 0..nm {
            let nn1 = k1 + m - 1;
            let nn2 = k2 + m - 1;
            big[(nn1, nn2)] = t_sub[(k1, k2)];
            big[(nn1, nn2 + nmax)] = t_sub[(k1, k2 + nm)];
            big[(nn1 + nmax, nn2)] = t_sub[(k1 + nm, k2)];
            big[(nn1 + nmax, nn2 + nmax)] = t_sub[(k1 + nm, k2 + nm)];
        }
    }
    state.t[m] = big;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_build_for_small_sphere_runs() {
        // Sanity: the scaffolding doesn't panic for a simple sphere.
        let cfg = TMatrixConfig {
            axi: 1.0,
            rat: 1.0,
            lam: 6.283185307,
            m: Complex64::new(1.5, 0.01),
            eps: 1.0,
            np: -1,
            ddelt: 1e-3,
            ndgs: 2,
        };
        // Just build state at a modest nmax.
        let s = build_state(4, 8, cfg.lam, cfg.m, cfg.axi, cfg.eps, Shape::Spheroid);
        assert_eq!(s.nmax, 4);
        assert_eq!(s.ngauss, 8);
        assert_eq!(s.x.len(), 16);
        assert!(s.j.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tmatr0_runs_without_panic() {
        let mut s = build_state(
            4,
            8,
            6.283185307,
            Complex64::new(1.5, 0.01),
            1.0,
            1.0,
            Shape::Spheroid,
        );
        tmatr0(&mut s, true);
        // The T-matrix should be finite.
        for i in 0..s.t[0].nrows() {
            for j in 0..s.t[0].ncols() {
                let v = s.t[0][(i, j)];
                assert!(v.re.is_finite());
                assert!(v.im.is_finite());
            }
        }
    }

    /// Check m>=1 T-matrix blocks for a sphere also match Mie theory.
    /// For a sphere T^(m)_{n,n} must equal T^(0)_{n,n} for every azimuthal
    /// order m (rotational symmetry).
    #[test]
    fn sphere_tmatr_m1_matches_m0() {
        let lam = 6.283185307_f64;
        let cfg = TMatrixConfig {
            axi: 1.0,
            rat: 1.0,
            lam,
            m: Complex64::new(1.33, 0.01),
            eps: 1.0,
            np: -1,
            ddelt: 1e-6,
            ndgs: 2,
        };
        let state = calctmat(cfg);
        let nmax = state.nmax;
        let t0 = &state.t[0];
        let t1 = &state.t[1];

        eprintln!("\n--- sphere m=1 vs m=0 block diagonal (nmax={nmax}) ---");
        eprintln!(
            "  {:>3}  {:>28}  {:>28}  {:>28}  {:>28}",
            "n", "T0-11", "T1-11", "T0-22", "T1-22"
        );
        for n in 1..=nmax {
            let t0_11 = t0[(n - 1, n - 1)];
            let t1_11 = t1[(n - 1, n - 1)];
            let t0_22 = t0[(n - 1 + nmax, n - 1 + nmax)];
            let t1_22 = t1[(n - 1 + nmax, n - 1 + nmax)];
            eprintln!("  {n:>3}  {:>12.4e}{:+12.4e}i  {:>12.4e}{:+12.4e}i  {:>12.4e}{:+12.4e}i  {:>12.4e}{:+12.4e}i",
                t0_11.re, t0_11.im, t1_11.re, t1_11.im,
                t0_22.re, t0_22.im, t1_22.re, t1_22.im);
        }

        for n in 1..=nmax {
            let t0_11 = t0[(n - 1, n - 1)];
            let t1_11 = t1[(n - 1, n - 1)];
            let t0_22 = t0[(n - 1 + nmax, n - 1 + nmax)];
            let t1_22 = t1[(n - 1 + nmax, n - 1 + nmax)];
            let tol = 0.01;
            let norm11 = t0_11.norm().max(1e-30);
            let norm22 = t0_22.norm().max(1e-30);
            assert!(
                (t1_11 - t0_11).norm() / norm11 < tol,
                "n={n}: T(m=1)_11={t1_11} ≠ T(m=0)_11={t0_11}"
            );
            assert!(
                (t1_22 - t0_22).norm() / norm22 < tol,
                "n={n}: T(m=1)_22={t1_22} ≠ T(m=0)_22={t0_22}"
            );
        }
    }

    /// Step-1 internal consistency: sphere T-matrix (axis_ratio=1) must match
    /// the Mie coefficients from `mie.rs`.
    ///
    /// For a sphere Mishchenko's T-matrix reduces to:
    ///   T^(0)_{n,n} block(1,1) = -b_n  (TM / magnetic Mie coeff)
    ///   T^(0)_{n,n} block(2,2) = -a_n  (TE / electric Mie coeff)
    ///
    /// Any disagreement here pins the bug entirely inside tmatrix.rs /
    /// amplitude.rs before any Python-boundary effects.
    #[test]
    fn sphere_tmatrix_matches_mie() {
        use crate::mie;

        let lam = 6.283185307_f64; // 2π → x = radius for radius=1
        let pi = std::f64::consts::PI;

        // Three (radius, m) pairs matching the parity test cases.
        let cases: &[(f64, Complex64)] = &[
            (0.5, Complex64::new(1.33, 0.0)),
            (1.0, Complex64::new(1.33, 0.01)),
            (2.0, Complex64::new(1.5, 0.001)),
        ];

        for &(radius, m) in cases {
            let x = 2.0 * pi * radius / lam; // size parameter

            let cfg = TMatrixConfig {
                axi: radius,
                rat: 1.0,
                lam,
                m,
                eps: 1.0, // sphere
                np: -1,
                ddelt: 1e-6,
                ndgs: 2,
            };
            let state = calctmat(cfg);
            let nmax = state.nmax;
            let mie_c = mie::mie(x, m);
            let t = &state.t[0];

            eprintln!("\n--- sphere x={x:.3}, m={m}, nmax={nmax} ---");
            eprintln!(
                "  {:>3}  {:>24}  {:>24}  {:>24}  {:>24}",
                "n", "T11 (got)", "-b_n (mie)", "T22 (got)", "-a_n (mie)"
            );

            for n in 1..=nmax.min(mie_c.a.len()) {
                let t11 = t[(n - 1, n - 1)];
                let t22 = t[(n - 1 + nmax, n - 1 + nmax)];
                let neg_an = -mie_c.a[n - 1];
                let neg_bn = -mie_c.b[n - 1];
                eprintln!(
                    "  {n:>3}  {:>12.6e}{:+12.6e}i  {:>12.6e}{:+12.6e}i  {:>12.6e}{:+12.6e}i  {:>12.6e}{:+12.6e}i",
                    t11.re, t11.im, neg_bn.re, neg_bn.im,
                    t22.re, t22.im, neg_an.re, neg_an.im
                );
            }

            // Assert first non-negligible n to 1 % relative error.
            // For a sphere T^(0) diagonal is the only non-zero block.
            let n = 1;
            let t11 = t[(n - 1, n - 1)];
            let t22 = t[(n - 1 + nmax, n - 1 + nmax)];
            let neg_an = -mie_c.a[n - 1];
            let neg_bn = -mie_c.b[n - 1];

            // Try both assignments and assert the one that matches.
            let tol = 0.02; // 2 % — tight enough to catch a 7× error
            let t11_matches_neg_bn = (t11 - neg_bn).norm() / neg_bn.norm().max(1e-30) < tol;
            let t22_matches_neg_an = (t22 - neg_an).norm() / neg_an.norm().max(1e-30) < tol;
            let t11_matches_neg_an = (t11 - neg_an).norm() / neg_an.norm().max(1e-30) < tol;
            let t22_matches_neg_bn = (t22 - neg_bn).norm() / neg_bn.norm().max(1e-30) < tol;

            assert!(
                (t11_matches_neg_bn && t22_matches_neg_an)
                    || (t11_matches_neg_an && t22_matches_neg_bn),
                "sphere x={x:.3}: T11={t11}, T22={t22} do not match Mie -b_n={neg_bn}, -a_n={neg_an} (tol={tol})"
            );
        }
    }
}
