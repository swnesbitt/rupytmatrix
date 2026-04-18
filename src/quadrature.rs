//! Gauss-Legendre quadrature — port of Mishchenko's `GAUSS` subroutine.
//!
//! Returns nodes on `(-1, 1)` and weights for integration of polynomials up
//! to order `2n - 1`. The implementation uses the standard Newton iteration
//! on the Legendre polynomial `P_n(x)` with the recurrence
//!
//! ```text
//!   (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
//!   P_n'(x) = n (x P_n(x) - P_{n-1}(x)) / (x^2 - 1)
//! ```
//!
//! and `w_i = 2 / ((1 - x_i^2) [P_n'(x_i)]^2)`.
//!
//! `ind1` / `ind2` in the original Fortran control endpoint inclusion and
//! half-range output; we expose those via [`GaussOptions`].

#[derive(Clone, Copy, Debug)]
pub struct GaussOptions {
    /// If `true`, return only the nodes/weights for `(0, 1)` (used by the
    /// T-matrix code to exploit symmetry of axially symmetric particles).
    pub half_range: bool,
    /// If `true`, add the endpoints `-1` and `+1` with zero weight. This
    /// mirrors the `IND1` flag in Mishchenko's code.
    pub add_endpoints: bool,
}

impl Default for GaussOptions {
    fn default() -> Self {
        Self {
            half_range: false,
            add_endpoints: false,
        }
    }
}

/// Compute Gauss-Legendre nodes and weights on `(-1, 1)` (unless `half_range`).
///
/// Returns `(nodes, weights)` of length `n` (plus 2 if `add_endpoints`).
pub fn gauss_legendre(n: usize, opts: GaussOptions) -> (Vec<f64>, Vec<f64>) {
    assert!(n > 0, "gauss_legendre requires n >= 1");

    // Roots are symmetric about 0; compute half of them, mirror.
    let m = (n + 1) / 2;
    let mut x = vec![0.0f64; n];
    let mut w = vec![0.0f64; n];

    let eps = 1.0e-15;
    let pi = std::f64::consts::PI;

    for i in 1..=m {
        // Initial guess — Tricomi's asymptotic formula.
        let mut z = (pi * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
        let mut pp;
        loop {
            // Evaluate P_n(z) and P_{n-1}(z) via recurrence.
            let mut p1 = 1.0f64;
            let mut p2 = 0.0f64;
            for k in 1..=n {
                let p3 = p2;
                p2 = p1;
                let kf = k as f64;
                p1 = ((2.0 * kf - 1.0) * z * p2 - (kf - 1.0) * p3) / kf;
            }
            // pp = n!*derivative at z using closed form.
            pp = (n as f64) * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z -= p1 / pp;
            if (z - z1).abs() < eps {
                break;
            }
        }
        // Symmetric pair.
        x[i - 1] = -z;
        x[n - i] = z;
        let wi = 2.0 / ((1.0 - z * z) * pp * pp);
        w[i - 1] = wi;
        w[n - i] = wi;
    }

    if opts.half_range {
        // Map (-1,1) nodes to (0,1) and scale weights by 1/2.
        // Mishchenko uses only the positive half of the full (-1,1) nodes.
        // When n is odd, the middle node sits at exactly 0; we skip it.
        let mut xh = Vec::with_capacity(n);
        let mut wh = Vec::with_capacity(n);
        for (&xi, &wi) in x.iter().zip(w.iter()) {
            if xi > 0.0 {
                xh.push(xi);
                wh.push(wi);
            }
        }
        // Pad back up to n by duplicating — matches the historical layout
        // where `Z(1..NGAUSS)` holds half-range nodes and `Z(NGAUSS+1..2*NGAUSS)`
        // holds their negatives. We keep length = number of positive nodes.
        (xh, wh)
    } else if opts.add_endpoints {
        let mut xe = Vec::with_capacity(n + 2);
        let mut we = Vec::with_capacity(n + 2);
        xe.push(-1.0);
        we.push(0.0);
        xe.extend_from_slice(&x);
        we.extend_from_slice(&w);
        xe.push(1.0);
        we.push(0.0);
        (xe, we)
    } else {
        (x, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gauss_n5_matches_known_values() {
        let (x, w) = gauss_legendre(5, GaussOptions::default());
        // Reference: classical 5-point Gauss-Legendre table.
        let ref_x = [
            -0.9061798459386640,
            -0.5384693101056831,
            0.0,
            0.5384693101056831,
            0.9061798459386640,
        ];
        let ref_w = [
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891,
        ];
        for (a, b) in x.iter().zip(ref_x.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
        for (a, b) in w.iter().zip(ref_w.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn gauss_integrates_polynomials_exactly() {
        // Integrate x^k on (-1,1) for k = 0..2n-1. Even k -> 2/(k+1), odd -> 0.
        for n in 2..10 {
            let (x, w) = gauss_legendre(n, GaussOptions::default());
            for k in 0..2 * n {
                let approx: f64 = x
                    .iter()
                    .zip(w.iter())
                    .map(|(&xi, &wi)| wi * xi.powi(k as i32))
                    .sum();
                let exact = if k % 2 == 1 {
                    0.0
                } else {
                    2.0 / (k as f64 + 1.0)
                };
                assert_abs_diff_eq!(approx, exact, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn weights_sum_to_two() {
        for n in [3, 8, 16, 32, 64] {
            let (_, w) = gauss_legendre(n, GaussOptions::default());
            let s: f64 = w.iter().sum();
            assert_abs_diff_eq!(s, 2.0, epsilon = 1e-13);
        }
    }

    #[test]
    fn half_range_has_positive_nodes_only() {
        let (x, _) = gauss_legendre(
            8,
            GaussOptions {
                half_range: true,
                add_endpoints: false,
            },
        );
        assert_eq!(x.len(), 4);
        for xi in x {
            assert!(xi > 0.0);
        }
    }
}
