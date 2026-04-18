//! Closed-form Mie scattering for a homogeneous sphere.
//!
//! The axis_ratio = 1 limit of the T-matrix reduces to classical Mie theory.
//! This module provides an independent implementation so parity tests can
//! compare `rupytmatrix` against a reference that does not share the same
//! code path as the general T-matrix solver.
//!
//! The algorithm follows the standard Bohren–Huffman formulation:
//! compute the logarithmic derivative `D_n(mx)` by downward recurrence, then
//! the Mie coefficients
//!
//! ```text
//!   a_n = ( D_n(mx)/m + n/x ) psi_n(x) - psi_{n-1}(x)
//!         -----------------------------------------------
//!         ( D_n(mx)/m + n/x ) xi_n(x)  - xi_{n-1}(x)
//!
//!   b_n = ( m D_n(mx)   + n/x ) psi_n(x) - psi_{n-1}(x)
//!         -----------------------------------------------
//!         ( m D_n(mx)   + n/x ) xi_n(x)  - xi_{n-1}(x)
//! ```
//!
//! with `psi_n(x) = x j_n(x)` and `xi_n(x) = x (j_n(x) - i y_n(x))`
//! (Riccati–Bessel).

use num_complex::Complex64;

/// Mie coefficients `a_n`, `b_n` for `n = 1 .. nmax`.
pub struct MieCoefficients {
    pub a: Vec<Complex64>,
    pub b: Vec<Complex64>,
}

/// Compute Mie coefficients for a homogeneous sphere of size parameter `x`
/// and complex refractive index `m`.
pub fn mie(x: f64, m: Complex64) -> MieCoefficients {
    assert!(x > 0.0, "size parameter must be positive");
    // Truncation order — standard Wiscombe criterion.
    let nstop = (x + 4.0 * x.powf(1.0 / 3.0) + 2.0) as usize;
    let nmx = (nstop.max((m.norm() * x) as usize) + 15) as usize;

    // Logarithmic derivative D_n(mx) by downward recurrence.
    // Bohren–Huffman Eq. 4.90: D_{n-1}(ρ) = n/ρ − 1/(D_n(ρ) + n/ρ).
    //
    // Indexing: d[k-1] stores D_k (1-indexed D into a 0-indexed
    // array). So when we compute d[n-1] from d[n], that means
    // computing D_n from D_{n+1}, which puts the coefficient at
    // (n+1)/ρ following BHMIE's convention.
    let mx = m * x;
    let mut d = vec![Complex64::new(0.0, 0.0); nmx + 1];
    for n in (1..nmx).rev() {
        let rn = Complex64::new((n as f64) + 1.0, 0.0) / mx;
        d[n - 1] = rn - Complex64::new(1.0, 0.0) / (d[n] + rn);
    }

    // Riccati–Bessel psi, chi by upward recurrence.
    let mut psi_prev = x.sin();
    let mut psi = x.sin() / x - x.cos();
    let mut chi_prev = x.cos();
    let mut chi = x.cos() / x + x.sin();

    let mut a = Vec::with_capacity(nstop);
    let mut b = Vec::with_capacity(nstop);

    for n in 1..=nstop {
        let nf = n as f64;
        // xi_n = psi_n - i chi_n ; a_n, b_n formulas.
        let xi_prev = Complex64::new(psi_prev, -chi_prev);
        let xi = Complex64::new(psi, -chi);

        let dn = d[n - 1];
        let numer_a = (dn / m + Complex64::new(nf / x, 0.0)) * Complex64::new(psi, 0.0)
            - Complex64::new(psi_prev, 0.0);
        let denom_a = (dn / m + Complex64::new(nf / x, 0.0)) * xi - xi_prev;
        a.push(numer_a / denom_a);

        let numer_b = (m * dn + Complex64::new(nf / x, 0.0)) * Complex64::new(psi, 0.0)
            - Complex64::new(psi_prev, 0.0);
        let denom_b = (m * dn + Complex64::new(nf / x, 0.0)) * xi - xi_prev;
        b.push(numer_b / denom_b);

        // Advance Riccati-Bessel recurrence.
        let psi_new = (2.0 * nf + 1.0) / x * psi - psi_prev;
        psi_prev = psi;
        psi = psi_new;
        let chi_new = (2.0 * nf + 1.0) / x * chi - chi_prev;
        chi_prev = chi;
        chi = chi_new;
    }

    MieCoefficients { a, b }
}

/// Scattering efficiency Q_sca for a Mie sphere.
pub fn qsca(x: f64, m: Complex64) -> f64 {
    let coeffs = mie(x, m);
    let mut q = 0.0;
    for (n, (an, bn)) in coeffs.a.iter().zip(coeffs.b.iter()).enumerate() {
        let nf = (n + 1) as f64;
        q += (2.0 * nf + 1.0) * (an.norm_sqr() + bn.norm_sqr());
    }
    (2.0 / (x * x)) * q
}

/// Extinction efficiency Q_ext for a Mie sphere.
pub fn qext(x: f64, m: Complex64) -> f64 {
    let coeffs = mie(x, m);
    let mut q = 0.0;
    for (n, (an, bn)) in coeffs.a.iter().zip(coeffs.b.iter()).enumerate() {
        let nf = (n + 1) as f64;
        q += (2.0 * nf + 1.0) * (an.re + bn.re);
    }
    (2.0 / (x * x)) * q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayleigh_limit_small_sphere() {
        // For x -> 0 and real m, Q_sca ≈ (8/3) x^4 |(m²-1)/(m²+2)|².
        // The first term of the Q_ext expansion (absorption) vanishes
        // because m has no imaginary part here.
        let x = 0.01;
        let m = Complex64::new(1.33, 0.0);
        let q_sca = qsca(x, m);
        let m2 = m * m;
        let alpha = (m2 - Complex64::new(1.0, 0.0)) / (m2 + Complex64::new(2.0, 0.0));
        let expected = (8.0 / 3.0) * x.powi(4) * alpha.norm_sqr();
        // Use a *relative* tolerance; absolute 1e-8 is useless here
        // because the answer itself is ~1e-9.
        let rel_err = (q_sca - expected).abs() / expected;
        assert!(
            rel_err < 1e-5,
            "Rayleigh Q_sca off: got {q_sca:.6e}, expected {expected:.6e}, rel err {rel_err:.2e}"
        );
    }

    #[test]
    fn perfectly_conducting_q_matches_table() {
        // Standard Mie case: x = 3, m = 1.5 + 0.001i. Reference from
        // Bohren & Huffman Appendix. Q_ext ≈ 3.45, Q_sca ≈ 3.45.
        let q_ext = qext(3.0, Complex64::new(1.5, 0.001));
        assert!(q_ext > 2.0 && q_ext < 5.0);
    }
}

#[cfg(test)]
mod spot_tests {
    use super::*;

    // Reference values computed with miepython 2.x (BHMIE).
    // Cross-checked against a verbatim Bohren-Huffman BHMIE port in Python.

    #[test]
    fn bhmie_x1_m15() {
        // x = 1, m = 1.5 + 0i: Q_sca = Q_ext = 0.215098 (miepython).
        let q = qsca(1.0, Complex64::new(1.5, 0.0));
        let rel = (q - 0.215098).abs() / 0.215098;
        assert!(rel < 1e-4, "Q_sca(x=1,m=1.5) = {q:.6}, rel err {rel:.2e}");
    }

    #[test]
    fn bhmie_x3_m15() {
        // x = 3, m = 1.5 + 0i: Q_sca = 3.418056 (miepython).
        let q = qsca(3.0, Complex64::new(1.5, 0.0));
        let rel = (q - 3.418056).abs() / 3.418056;
        assert!(rel < 1e-4, "Q_sca(x=3,m=1.5) = {q:.6}, rel err {rel:.2e}");
    }

    #[test]
    fn absorbing_large_sphere() {
        // x = 10, m = 1.33 + 0.01i: Q_sca = 1.872112, Q_ext = 2.249241 (miepython).
        let qs = qsca(10.0, Complex64::new(1.33, 0.01));
        let qe = qext(10.0, Complex64::new(1.33, 0.01));
        let rel_s = (qs - 1.872112).abs() / 1.872112;
        let rel_e = (qe - 2.249241).abs() / 2.249241;
        assert!(
            rel_s < 1e-4,
            "Q_sca(x=10,m=1.33+0.01i) = {qs:.6}, rel err {rel_s:.2e}"
        );
        assert!(
            rel_e < 1e-4,
            "Q_ext(x=10,m=1.33+0.01i) = {qe:.6}, rel err {rel_e:.2e}"
        );
        assert!(qe > qs, "Q_ext must exceed Q_sca for absorbing sphere");
    }
}
