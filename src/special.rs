//! Spherical Bessel functions — ports of Mishchenko's `RJB`, `RYB`, `CJB`.
//!
//! The T-matrix formulation uses *Riccati–Bessel* functions
//!
//! ```text
//!   psi_n(x) = x j_n(x)
//!   chi_n(x) = -x y_n(x)   (sign convention used in Mishchenko's code)
//! ```
//!
//! where `j_n` and `y_n` are the spherical Bessel functions of the first
//! and second kind. The code returns `U_n = psi_n(x) / x = j_n(x)` and its
//! logarithmic derivative `U_n' / U_n` for stability. We follow the same
//! conventions exactly so results match the Fortran bit-for-bit up to
//! rounding.

use num_complex::Complex64;

/// Spherical Bessel function `j_n(x)` for `n = 0 .. nmax` (inclusive), real argument.
///
/// Uses downward recurrence (Miller's algorithm) for stability when `n > x`:
///
/// ```text
///   j_{n-1}(x) = (2n+1)/x * j_n(x) - j_{n+1}(x)
/// ```
///
/// starting from arbitrary values at `n = nmax + nnmax` and normalising
/// against the known closed form `j_0(x) = sin(x) / x`.
///
/// `nnmax` is the number of extra terms used to seed the recurrence; Mishchenko
/// uses `nnmax1 = 1.2*sqrt(max(x, nmax)) + 3` or similar.
pub fn spherical_jn(x: f64, nmax: usize, nnmax: usize) -> Vec<f64> {
    assert!(x > 0.0, "spherical_jn requires x > 0 (got {x})");
    let ntot = nmax + nnmax;
    // Seed high-order values with arbitrary magnitudes.
    let mut y = vec![0.0f64; ntot + 2];
    // j_{ntot+1} = 0, j_{ntot} = 1 (arbitrary); scale later.
    y[ntot + 1] = 0.0;
    y[ntot] = 1.0;
    for n in (1..=ntot).rev() {
        let nf = n as f64;
        y[n - 1] = (2.0 * nf + 1.0) / x * y[n] - y[n + 1];
    }
    let j0_true = x.sin() / x;
    let scale = j0_true / y[0];
    let mut out = Vec::with_capacity(nmax + 1);
    for n in 0..=nmax {
        out.push(y[n] * scale);
    }
    out
}

/// Spherical Bessel function of the second kind `y_n(x)`, `n = 0 .. nmax`.
///
/// Computed by upward recurrence, which is stable for `y_n`:
/// `y_{n+1}(x) = (2n+1)/x * y_n(x) - y_{n-1}(x)`.
pub fn spherical_yn(x: f64, nmax: usize) -> Vec<f64> {
    assert!(x > 0.0, "spherical_yn requires x > 0 (got {x})");
    let mut y = vec![0.0f64; nmax + 1];
    y[0] = -x.cos() / x;
    if nmax >= 1 {
        y[1] = -x.cos() / (x * x) - x.sin() / x;
    }
    for n in 1..nmax {
        let nf = n as f64;
        y[n + 1] = (2.0 * nf + 1.0) / x * y[n] - y[n - 1];
    }
    y
}

/// Complex-argument spherical Bessel `j_n(z)`, `n = 0 .. nmax`, via downward
/// recurrence — port of `CJB`. `nnmax` is the recurrence seed depth.
pub fn spherical_jn_complex(z: Complex64, nmax: usize, nnmax: usize) -> Vec<Complex64> {
    assert!(z.norm() > 0.0, "spherical_jn_complex requires |z| > 0");
    let ntot = nmax + nnmax;
    let mut y = vec![Complex64::new(0.0, 0.0); ntot + 2];
    y[ntot] = Complex64::new(1.0, 0.0);
    for n in (1..=ntot).rev() {
        let nf = n as f64;
        y[n - 1] = Complex64::new(2.0 * nf + 1.0, 0.0) / z * y[n] - y[n + 1];
    }
    // j_0(z) = sin(z) / z
    let j0_true = z.sin() / z;
    let scale = j0_true / y[0];
    (0..=nmax).map(|n| y[n] * scale).collect()
}

/// Return `(psi, psi_over_x, dpsi)` where
///
/// * `psi[n]   = j_n(x) * x`                (Riccati–Bessel)
/// * `psi_over_x[n] = j_n(x)`               (convenience for the T-matrix block fill)
/// * `dpsi[n] = d/dx [ x j_n(x) ]`
///
/// for `n = 1 .. nmax`.
pub fn riccati_bessel_j(x: f64, nmax: usize, nnmax: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let jn = spherical_jn(x, nmax, nnmax);
    let mut psi = vec![0.0; nmax + 1];
    let mut dpsi = vec![0.0; nmax + 1];
    for n in 0..=nmax {
        psi[n] = x * jn[n];
    }
    // d/dx [ x j_n(x) ] = j_n(x) + x j_n'(x)
    //                   = j_n(x) + x * ( (n/x) j_n - j_{n+1} )
    //                   = (n+1) j_n(x) - x j_{n+1}(x)
    for n in 0..nmax {
        dpsi[n] = (n as f64 + 1.0) * jn[n] - x * jn[n + 1];
    }
    // Last one needs one more term — recompute with an extra element.
    let jn2 = spherical_jn(x, nmax + 1, nnmax);
    dpsi[nmax] = (nmax as f64 + 1.0) * jn[nmax] - x * jn2[nmax + 1];
    (psi, jn, dpsi)
}

/// Same as [`riccati_bessel_j`] but for `y_n`, returning `chi_n = -x y_n(x)`
/// and its derivative. Sign follows Mishchenko's convention so that the
/// Riccati–Hankel function is `xi_n = psi_n - i*chi_n`.
pub fn riccati_bessel_y(x: f64, nmax: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let yn = spherical_yn(x, nmax + 1);
    let mut chi = vec![0.0; nmax + 1];
    let mut dchi = vec![0.0; nmax + 1];
    for n in 0..=nmax {
        chi[n] = -x * yn[n];
        dchi[n] = -((n as f64 + 1.0) * yn[n] - x * yn[n + 1]);
    }
    (chi, yn, dchi)
}

/// Complex-argument Riccati-Bessel `psi(z) = z * j_n(z)` and derivative.
pub fn riccati_bessel_j_complex(
    z: Complex64,
    nmax: usize,
    nnmax: usize,
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let jn = spherical_jn_complex(z, nmax + 1, nnmax);
    let mut psi = vec![Complex64::new(0.0, 0.0); nmax + 1];
    let mut dpsi = vec![Complex64::new(0.0, 0.0); nmax + 1];
    for n in 0..=nmax {
        psi[n] = z * jn[n];
        dpsi[n] = Complex64::new(n as f64 + 1.0, 0.0) * jn[n] - z * jn[n + 1];
    }
    (psi, jn[0..=nmax].to_vec(), dpsi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn spherical_j0_matches_closed_form() {
        for x in [0.1, 0.5, 1.0, 3.14159, 7.5, 15.0] {
            let j = spherical_jn(x, 0, 20);
            assert_abs_diff_eq!(j[0], x.sin() / x, epsilon = 1e-13);
        }
    }

    #[test]
    fn spherical_j1_matches_closed_form() {
        for x in [0.1, 0.5, 1.0, 3.14159, 7.5, 15.0] {
            let j = spherical_jn(x, 3, 30);
            let j1_true = x.sin() / (x * x) - x.cos() / x;
            assert_abs_diff_eq!(j[1], j1_true, epsilon = 1e-11);
        }
    }

    #[test]
    fn spherical_y_recurrence_consistent() {
        for x in [0.5, 1.0, 3.0, 10.0] {
            let y = spherical_yn(x, 10);
            assert_abs_diff_eq!(y[0], -x.cos() / x, epsilon = 1e-13);
            assert_abs_diff_eq!(y[1], -x.cos() / (x * x) - x.sin() / x, epsilon = 1e-13);
            // Check recurrence j_{n+1} y_n - j_n y_{n+1} = 1/x^2 Wronskian
            let j = spherical_jn(x, 10, 30);
            for n in 0..10 {
                let w = j[n + 1] * y[n] - j[n] * y[n + 1];
                assert_abs_diff_eq!(w, 1.0 / (x * x), epsilon = 1e-11);
            }
        }
    }

    #[test]
    fn riccati_bessel_j_has_right_derivative() {
        // Numerical derivative vs analytic.
        let x = 5.5;
        let (psi, _j, dpsi) = riccati_bessel_j(x, 6, 30);
        let h = 1e-6;
        let (psi_ph, _, _) = riccati_bessel_j(x + h, 6, 30);
        let (psi_mh, _, _) = riccati_bessel_j(x - h, 6, 30);
        for n in 1..=6 {
            let num = (psi_ph[n] - psi_mh[n]) / (2.0 * h);
            assert_abs_diff_eq!(dpsi[n], num, epsilon = 1e-7);
            // Unused.
            let _ = psi[n];
        }
    }

    #[test]
    fn complex_jn_reduces_to_real_for_real_argument() {
        let x = 3.0;
        let jr = spherical_jn(x, 8, 30);
        let jc = spherical_jn_complex(Complex64::new(x, 0.0), 8, 30);
        for n in 0..=8 {
            assert_abs_diff_eq!(jc[n].re, jr[n], epsilon = 1e-12);
            assert_abs_diff_eq!(jc[n].im, 0.0, epsilon = 1e-13);
        }
    }
}
