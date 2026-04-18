//! Amplitude and phase matrix evaluation — port of Mishchenko's `CALCAMPL`
//! and `AMPL` subroutines.
//!
//! Given the T-matrix produced by [`crate::tmatrix::calctmat`], compute the
//! 2x2 complex amplitude matrix `S` and the 4x4 real phase matrix `Z` for a
//! specified incident / scattered direction and Euler orientation.

use num_complex::Complex64;

use crate::tmatrix::TMatrixState;
use crate::wigner::vigampl;

/// Evaluate amplitude matrix S (2x2) and phase matrix Z (4x4).
///
/// Angles are in degrees and match the Fortran signature:
///
/// * `thet0`, `thet`: incident and scattered zenith angles (0..=180)
/// * `phi0`, `phi`: incident and scattered azimuth (0..=360)
/// * `alpha`, `beta`: Euler angles of the particle orientation (degrees)
pub fn calcampl(
    state: &TMatrixState,
    lam: f64,
    thet0: f64,
    thet: f64,
    phi0: f64,
    phi: f64,
    alpha: f64,
    beta: f64,
) -> ([[Complex64; 2]; 2], [[f64; 4]; 4]) {
    let s = ampl(state, lam, thet0, thet, phi0, phi, alpha, beta);
    // Phase matrix Z from S — Mishchenko Eqs. (13)-(29), Ref. 6.
    let s11 = s[0][0];
    let s12 = s[0][1];
    let s21 = s[1][0];
    let s22 = s[1][1];
    let c = |z: Complex64| z.conj();
    let re = |z: Complex64| z.re;
    let im = |z: Complex64| z.im;

    let mut z = [[0.0f64; 4]; 4];
    z[0][0] = 0.5 * (re(s11 * c(s11)) + re(s12 * c(s12)) + re(s21 * c(s21)) + re(s22 * c(s22)));
    z[0][1] = 0.5 * (re(s11 * c(s11)) - re(s12 * c(s12)) + re(s21 * c(s21)) - re(s22 * c(s22)));
    z[0][2] = -re(s11 * c(s12)) - re(s22 * c(s21));
    z[0][3] = im(s11 * c(s12)) - im(s22 * c(s21));
    z[1][0] = 0.5 * (re(s11 * c(s11)) + re(s12 * c(s12)) - re(s21 * c(s21)) - re(s22 * c(s22)));
    z[1][1] = 0.5 * (re(s11 * c(s11)) - re(s12 * c(s12)) - re(s21 * c(s21)) + re(s22 * c(s22)));
    z[1][2] = -re(s11 * c(s12)) + re(s22 * c(s21));
    z[1][3] = im(s11 * c(s12)) + im(s22 * c(s21));
    z[2][0] = -re(s11 * c(s21)) - re(s22 * c(s12));
    z[2][1] = -re(s11 * c(s21)) + re(s22 * c(s12));
    z[2][2] = re(s11 * c(s22)) + re(s12 * c(s21));
    z[2][3] = -im(s11 * c(s22)) - im(s21 * c(s12));
    z[3][0] = im(s21 * c(s11)) + im(s22 * c(s12));
    z[3][1] = im(s21 * c(s11)) - im(s22 * c(s12));
    z[3][2] = -im(s22 * c(s11)) + im(s12 * c(s21));
    z[3][3] = re(s22 * c(s11)) - re(s12 * c(s21));

    let mut s_out = [[Complex64::new(0.0, 0.0); 2]; 2];
    s_out[0][0] = s11;
    s_out[0][1] = s12;
    s_out[1][0] = s21;
    s_out[1][1] = s22;
    (s_out, z)
}

fn ampl(
    state: &TMatrixState,
    lam: f64,
    tl: f64,
    tl1: f64,
    pl: f64,
    pl1: f64,
    alpha: f64,
    beta: f64,
) -> [[Complex64; 2]; 2] {
    let nmax = state.nmax;
    let pin = std::f64::consts::PI;
    let pin2 = pin * 0.5;
    let deg = pin / 180.0;
    let alph = alpha * deg;
    let bet = beta * deg;
    let mut thetl = tl * deg;
    let mut phil = pl * deg;
    let mut thetl1 = tl1 * deg;
    let mut phil1 = pl1 * deg;
    let mut bet = bet;

    let eps = 1e-7;
    if thetl < pin2 {
        thetl += eps;
    }
    if thetl > pin2 {
        thetl -= eps;
    }
    if thetl1 < pin2 {
        thetl1 += eps;
    }
    if thetl1 > pin2 {
        thetl1 -= eps;
    }
    if phil < pin {
        phil += eps;
    }
    if phil > pin {
        phil -= eps;
    }
    if phil1 < pin {
        phil1 += eps;
    }
    if phil1 > pin {
        phil1 -= eps;
    }
    if bet <= pin2 && (pin2 - bet) <= eps {
        bet -= eps;
    }
    if bet > pin2 && (bet - pin2) <= eps {
        bet += eps;
    }

    let cb = bet.cos();
    let sb = bet.sin();
    let ct = thetl.cos();
    let st = thetl.sin();
    let mut cp = (phil - alph).cos();
    let mut sp = (phil - alph).sin();
    let ctp = ct * cb + st * sb * cp;
    let thetp = ctp.acos();
    let cpp = cb * st * cp - sb * ct;
    let spp = st * sp;
    let mut phip = (spp / cpp).atan();
    if phip > 0.0 && sp < 0.0 {
        phip += pin;
    }
    if phip < 0.0 && sp > 0.0 {
        phip += pin;
    }
    if phip < 0.0 {
        phip += 2.0 * pin;
    }

    let ct1 = thetl1.cos();
    let st1 = thetl1.sin();
    let mut cp1 = (phil1 - alph).cos();
    let mut sp1 = (phil1 - alph).sin();
    let ctp1 = ct1 * cb + st1 * sb * cp1;
    let thetp1 = ctp1.acos();
    let cpp1 = cb * st1 * cp1 - sb * ct1;
    let spp1 = st1 * sp1;
    let mut phip1 = (spp1 / cpp1).atan();
    if phip1 > 0.0 && sp1 < 0.0 {
        phip1 += pin;
    }
    if phip1 < 0.0 && sp1 > 0.0 {
        phip1 += pin;
    }
    if phip1 < 0.0 {
        phip1 += 2.0 * pin;
    }

    let ca = alph.cos();
    let sa = alph.sin();
    let b = [
        [ca * cb, sa * cb, -sb],
        [-sa, ca, 0.0],
        [ca * sb, sa * sb, cb],
    ];

    cp = phil.cos();
    sp = phil.sin();
    cp1 = phil1.cos();
    sp1 = phil1.sin();
    let al = [[ct * cp, -sp], [ct * sp, cp], [-st, 0.0]];
    let al1 = [[ct1 * cp1, -sp1], [ct1 * sp1, cp1], [-st1, 0.0]];

    let ct_p = ctp;
    let st_p = thetp.sin();
    cp = phip.cos();
    sp = phip.sin();
    let ct1_p = ctp1;
    let st1_p = thetp1.sin();
    cp1 = phip1.cos();
    sp1 = phip1.sin();
    let ap = [[ct_p * cp, ct_p * sp, -st_p], [-sp, cp, 0.0]];
    let ap1 = [[ct1_p * cp1, ct1_p * sp1, -st1_p], [-sp1, cp1, 0.0]];

    // R = AP * B * AL, R1 = AP1 * B * AL1, then invert R1.
    let mut c = [[0.0f64; 2]; 3];
    for i in 0..3 {
        for jj in 0..2 {
            let mut x = 0.0;
            for k in 0..3 {
                x += b[i][k] * al[k][jj];
            }
            c[i][jj] = x;
        }
    }
    let mut r_mat = [[0.0f64; 2]; 2];
    for i in 0..2 {
        for jj in 0..2 {
            let mut x = 0.0;
            for k in 0..3 {
                x += ap[i][k] * c[k][jj];
            }
            r_mat[i][jj] = x;
        }
    }
    for i in 0..3 {
        for jj in 0..2 {
            let mut x = 0.0;
            for k in 0..3 {
                x += b[i][k] * al1[k][jj];
            }
            c[i][jj] = x;
        }
    }
    let mut r1 = [[0.0f64; 2]; 2];
    for i in 0..2 {
        for jj in 0..2 {
            let mut x = 0.0;
            for k in 0..3 {
                x += ap1[i][k] * c[k][jj];
            }
            r1[i][jj] = x;
        }
    }
    let d = 1.0 / (r1[0][0] * r1[1][1] - r1[0][1] * r1[1][0]);
    let tmp = r1[0][0];
    r1[0][0] = r1[1][1] * d;
    r1[0][1] = -r1[0][1] * d;
    r1[1][0] = -r1[1][0] * d;
    r1[1][1] = tmp * d;

    // Precompute i^{nn-n-1} * normalisation, stored as cal[(n-1, nn-1)].
    let ci = Complex64::new(0.0, 1.0);
    let mut cal = vec![Complex64::new(0.0, 0.0); nmax * nmax];
    for n in 1..=nmax {
        for nn in 1..=nmax {
            let k = (nn as i64) - (n as i64) - 1;
            let cn = complex_i_pow(ci, k);
            let dnn_num = ((2 * n + 1) * (2 * nn + 1)) as f64;
            let dnn_den = (n * nn * (n + 1) * (nn + 1)) as f64;
            let rn = (dnn_num / dnn_den).sqrt();
            cal[(n - 1) * nmax + (nn - 1)] = cn * rn;
        }
    }

    let dcth0 = ctp;
    let dcth = ctp1;
    let ph = phip1 - phip;
    let mut vv = Complex64::new(0.0, 0.0);
    let mut vh = Complex64::new(0.0, 0.0);
    let mut hv = Complex64::new(0.0, 0.0);
    let mut hh = Complex64::new(0.0, 0.0);
    for mm in 0..=nmax {
        let nmin = mm.max(1);
        let (dv1, dv2) = vigampl(dcth, nmax, mm);
        let (dv01, dv02) = vigampl(dcth0, nmax, mm);
        let fc = 2.0 * (mm as f64 * ph).cos();
        let fs = 2.0 * (mm as f64 * ph).sin();
        for nn in nmin..=nmax {
            let dv1nn = mm as f64 * dv01[nn - 1];
            let dv2nn = dv02[nn - 1];
            for n in nmin..=nmax {
                let dv1n = mm as f64 * dv1[n - 1];
                let dv2n = dv2[n - 1];

                let t_mm = &state.t[mm];
                // In state.t[mm], row-block layout is (n-1) in 0..nmax and (n-1+nmax)
                // in nmax..2*nmax, correspondingly for columns.
                let ct11 = t_mm[(n - 1, nn - 1)];
                let ct22 = t_mm[(n - 1 + nmax, nn - 1 + nmax)];

                if mm == 0 {
                    let cn = cal[(n - 1) * nmax + (nn - 1)] * dv2n * dv2nn;
                    vv += cn * ct22;
                    hh += cn * ct11;
                } else {
                    let ct12 = t_mm[(n - 1, nn - 1 + nmax)];
                    let ct21 = t_mm[(n - 1 + nmax, nn - 1)];
                    let cn1 = cal[(n - 1) * nmax + (nn - 1)] * fc;
                    let cn2 = cal[(n - 1) * nmax + (nn - 1)] * fs;
                    let d11 = dv1n * dv1nn;
                    let d12 = dv1n * dv2nn;
                    let d21 = dv2n * dv1nn;
                    let d22 = dv2n * dv2nn;
                    vv += (ct11 * d11 + ct21 * d21 + ct12 * d12 + ct22 * d22) * cn1;
                    vh += (ct11 * d12 + ct21 * d22 + ct12 * d11 + ct22 * d21) * cn2;
                    hv -= (ct11 * d21 + ct21 * d11 + ct12 * d22 + ct22 * d12) * cn2;
                    hh += (ct11 * d22 + ct21 * d12 + ct12 * d21 + ct22 * d11) * cn1;
                }
            }
        }
    }
    let dk = 2.0 * pin / lam;
    vv /= dk;
    vh /= dk;
    hv /= dk;
    hh /= dk;

    let cvv = vv * r_mat[0][0] + vh * r_mat[1][0];
    let cvh = vv * r_mat[0][1] + vh * r_mat[1][1];
    let chv = hv * r_mat[0][0] + hh * r_mat[1][0];
    let chh = hv * r_mat[0][1] + hh * r_mat[1][1];
    let vv = r1[0][0] * cvv + r1[0][1] * chv;
    let vh = r1[0][0] * cvh + r1[0][1] * chh;
    let hv = r1[1][0] * cvv + r1[1][1] * chv;
    let hh = r1[1][0] * cvh + r1[1][1] * chh;

    [[vv, vh], [hv, hh]]
}

/// Complex i^k for integer k.
fn complex_i_pow(i: Complex64, k: i64) -> Complex64 {
    let m = (((k % 4) + 4) % 4) as u32;
    match m {
        0 => Complex64::new(1.0, 0.0),
        1 => i,
        2 => Complex64::new(-1.0, 0.0),
        _ => -i,
    }
}
