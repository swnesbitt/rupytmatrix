//! Associated Legendre / Wigner d-function helpers.
//!
//! These are direct ports of Mishchenko's `VIG` and `VIGAMPL` subroutines.
//! They compute the auxiliary functions
//!
//! ```text
//!   d^l_{0,m}(theta)   — "DV1"
//!   d d^l_{0,m}/dtheta — "DV2"
//! ```
//!
//! used to fill the T-matrix blocks. The normalisation matches the
//! "reduced rotation matrix elements" used throughout the T-matrix
//! literature (Mishchenko 2000, Eq. 14).

/// VIGAMPL variant — used in amplitude matrix evaluation. Computes
/// `d^n_{0,m}(cos theta) / sin theta` in `dv1[n-1]` and the theta-derivative
/// in `dv2[n-1]`, for `n = 1..=nmax`.
///
/// Special cases:
/// * If `|x| == 1`, the output is the Mishchenko limiting form
///   (see Fortran lines 100-110 of `VIGAMPL`).
pub fn vigampl(x: f64, nmax: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut dv1 = vec![0.0f64; nmax];
    let mut dv2 = vec![0.0f64; nmax];
    let dx = x.abs();
    if (1.0 - dx).abs() <= 1.0e-10 {
        if m != 1 {
            return (dv1, dv2);
        }
        for n in 1..=nmax {
            let dn0 = (n as f64) * (n as f64 + 1.0);
            let mut dn = 0.5 * dn0.sqrt();
            if x < 0.0 {
                dn *= if (n + 1) % 2 == 0 { 1.0 } else { -1.0 };
            }
            dv1[n - 1] = dn;
            let mut dn2 = dn;
            if x < 0.0 {
                dn2 = -dn;
            }
            dv2[n - 1] = dn2;
        }
        return (dv1, dv2);
    }

    let qs = (1.0 - x * x).sqrt();
    let qs1 = 1.0 / qs;
    let dsi = qs1;

    if m == 0 {
        let mut d1 = 1.0;
        let mut d2 = x;
        for n in 1..=nmax {
            let qn = n as f64;
            let qn1 = (n + 1) as f64;
            let qn2 = (2 * n + 1) as f64;
            let d3 = (qn2 * x * d2 - qn * d1) / qn1;
            let der = qs1 * (qn1 * qn / qn2) * (-d1 + d3);
            dv1[n - 1] = d2 * dsi;
            dv2[n - 1] = der;
            d1 = d2;
            d2 = d3;
        }
    } else {
        let qmm = (m * m) as f64;
        let mut a = 1.0f64;
        for i in 1..=m {
            let i2 = 2 * i;
            a *= ((i2 as f64 - 1.0) / (i2 as f64)).sqrt() * qs;
        }
        let mut d1 = 0.0f64;
        let mut d2 = a;
        for n in m..=nmax {
            let qn = n as f64;
            let qn2 = (2 * n + 1) as f64;
            let qn1 = (n + 1) as f64;
            let qnm = (qn * qn - qmm).sqrt();
            let qnm1 = (qn1 * qn1 - qmm).sqrt();
            let d3 = (qn2 * x * d2 - qnm * d1) / qnm1;
            let der = qs1 * (-qn1 * qnm * d1 + qn * qnm1 * d3) / qn2;
            dv1[n - 1] = d2 * dsi;
            dv2[n - 1] = der;
            d1 = d2;
            d2 = d3;
        }
    }
    (dv1, dv2)
}

/// VIG variant — same as `vigampl` but does NOT divide by `sin theta`.
/// Used in the T-matrix block fill where the 1/sin factor is absorbed
/// elsewhere.
pub fn vig(x: f64, nmax: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut dv1 = vec![0.0f64; nmax];
    let mut dv2 = vec![0.0f64; nmax];
    let dx = x.abs();
    if (1.0 - dx).abs() <= 1.0e-10 {
        if m != 1 {
            return (dv1, dv2);
        }
        for n in 1..=nmax {
            let dn0 = (n as f64) * (n as f64 + 1.0);
            let mut dn = 0.5 * dn0.sqrt();
            if x < 0.0 {
                dn *= if (n + 1) % 2 == 0 { 1.0 } else { -1.0 };
            }
            dv1[n - 1] = dn;
            let mut dn2 = dn;
            if x < 0.0 {
                dn2 = -dn;
            }
            dv2[n - 1] = dn2;
        }
        return (dv1, dv2);
    }
    let qs = (1.0 - x * x).sqrt();
    let qs1 = 1.0 / qs;
    if m == 0 {
        let mut d1 = 1.0;
        let mut d2 = x;
        for n in 1..=nmax {
            let qn = n as f64;
            let qn1 = (n + 1) as f64;
            let qn2 = (2 * n + 1) as f64;
            let d3 = (qn2 * x * d2 - qn * d1) / qn1;
            let der = qs1 * (qn1 * qn / qn2) * (-d1 + d3);
            dv1[n - 1] = d2;
            dv2[n - 1] = der;
            d1 = d2;
            d2 = d3;
        }
    } else {
        let qmm = (m * m) as f64;
        let mut a = 1.0f64;
        for i in 1..=m {
            let i2 = 2 * i;
            a *= ((i2 as f64 - 1.0) / (i2 as f64)).sqrt() * qs;
        }
        let mut d1 = 0.0f64;
        let mut d2 = a;
        for n in m..=nmax {
            let qn = n as f64;
            let qn2 = (2 * n + 1) as f64;
            let qn1 = (n + 1) as f64;
            let qnm = (qn * qn - qmm).sqrt();
            let qnm1 = (qn1 * qn1 - qmm).sqrt();
            let d3 = (qn2 * x * d2 - qnm * d1) / qnm1;
            let der = qs1 * (-qn1 * qnm * d1 + qn * qnm1 * d3) / qn2;
            dv1[n - 1] = d2;
            dv2[n - 1] = der;
            d1 = d2;
            d2 = d3;
        }
    }
    (dv1, dv2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn vigampl_m0_n1_known_values() {
        // For m=0, d^1_{0,0}(cos theta) = cos theta; divided by sin theta gives
        // cot theta. At theta = pi/2 (x=0) it should be 0.
        let (dv1, _) = vigampl(0.0, 3, 0);
        assert_abs_diff_eq!(dv1[0], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn vig_symmetric_m_zero() {
        // d^n_{0,0}(cos(-theta)) = (-1)^n d^n_{0,0}(cos(theta))? Actually
        // d^n_{0,0}(x) = P_n(x), so P_n(-x) = (-1)^n P_n(x).
        let (dv_pos, _) = vig(0.3, 5, 0);
        let (dv_neg, _) = vig(-0.3, 5, 0);
        for n in 1..=5 {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert_abs_diff_eq!(dv_pos[n - 1], sign * dv_neg[n - 1], epsilon = 1e-13);
        }
    }

    #[test]
    fn vig_m_zero_matches_legendre() {
        // dv1 for m=0 equals the Legendre polynomial P_n(x).
        // P_1 = x; P_2 = (3x^2-1)/2; P_3 = (5x^3 - 3x)/2.
        let x = 0.4;
        let (dv, _) = vig(x, 4, 0);
        assert_abs_diff_eq!(dv[0], x, epsilon = 1e-14);
        assert_abs_diff_eq!(dv[1], 0.5 * (3.0 * x * x - 1.0), epsilon = 1e-14);
        assert_abs_diff_eq!(dv[2], 0.5 * (5.0 * x.powi(3) - 3.0 * x), epsilon = 1e-13);
    }
}
