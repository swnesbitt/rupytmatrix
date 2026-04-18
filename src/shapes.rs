//! Particle-shape radii — ports of Mishchenko's `RSP1..RSP4`.
//!
//! The T-matrix code represents axially-symmetric particles by the squared
//! radius `r(theta)^2` and the logarithmic derivative
//! `(1/r) dr/dtheta` sampled at the Gauss-Legendre nodes of `cos(theta)`.
//!
//! This module computes those arrays for:
//!
//! * [`Shape::Spheroid`] (`RSP1`) — prolate / oblate spheroids.
//! * [`Shape::Chebyshev`] (`RSP2`) — Chebyshev-distorted spheres `r = r0(1 + eps T_n(cos theta))`.
//! * [`Shape::Cylinder`] (`RSP3`) — finite circular cylinders.
//! * [`Shape::GenChebyshev`] (`RSP4`) — generalised Chebyshev expansions (distorted raindrops).

/// Particle shape selector. Numeric values mirror the Fortran `NP` convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// `NP = -1`: spheroid (ratio = horizontal/rotational axis).
    Spheroid,
    /// `NP = -2`: finite circular cylinder (ratio = diameter/length).
    Cylinder,
    /// `NP = -3`: generalized Chebyshev (raindrop) using the coefficients
    /// set up by [`crate::shapes::drop`].
    GenChebyshev,
    /// `NP >= 0`: Chebyshev particle of order n with deformation eps.
    Chebyshev(u32),
}

impl Shape {
    pub fn from_np(np: i32) -> Self {
        match np {
            -1 => Shape::Spheroid,
            -2 => Shape::Cylinder,
            -3 => Shape::GenChebyshev,
            n if n >= 0 => Shape::Chebyshev(n as u32),
            _ => panic!("unknown NP = {}", np),
        }
    }

    pub fn to_np(&self) -> i32 {
        match self {
            Shape::Spheroid => -1,
            Shape::Cylinder => -2,
            Shape::GenChebyshev => -3,
            Shape::Chebyshev(n) => *n as i32,
        }
    }
}

/// Spheroid shape. Returns `(r2, dlogr)` at the `ngauss` positive-cosine nodes,
/// mirrored for the negative half to produce length `2*ngauss`.
///
/// `rev` — equal-volume-sphere radius.
/// `eps` — horizontal/rotational axis ratio.
pub fn rsp1(x: &[f64], rev: f64, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let ngauss = x.len() / 2;
    let ng = x.len();
    let a = rev * eps.cbrt();
    let aa = a * a;
    let ee = eps * eps;
    let ee1 = ee - 1.0;
    let mut r = vec![0.0; ng];
    let mut dr = vec![0.0; ng];
    for i in 0..ngauss {
        let c = x[i];
        let cc = c * c;
        let ss = 1.0 - cc;
        let s = ss.sqrt();
        let rr = 1.0 / (ss + ee * cc);
        r[i] = aa * rr;
        r[ng - 1 - i] = r[i];
        dr[i] = rr * c * s * ee1;
        dr[ng - 1 - i] = -dr[i];
    }
    (r, dr)
}

/// Chebyshev particle `r = r0 (1 + eps * T_n(cos theta))`.
///
/// `rev` — equal-volume-sphere radius.
/// `eps` — deformation amplitude.
/// `n` — Chebyshev order.
pub fn rsp2(x: &[f64], rev: f64, eps: f64, n: u32) -> (Vec<f64>, Vec<f64>) {
    let ng = x.len();
    let dnp = n as f64;
    let dn = dnp * dnp;
    let dn4 = dn * 4.0;
    let ep = eps * eps;
    let mut a = 1.0 + 1.5 * ep * (dn4 - 2.0) / (dn4 - 1.0);
    let i_half = ((dnp + 0.1) * 0.5) as u32;
    let i2 = 2 * i_half;
    if i2 == n {
        a -= 3.0 * eps * (1.0 + 0.25 * ep) / (dn - 1.0) - 0.25 * ep * eps / (9.0 * dn - 1.0);
    }
    let r0 = rev * a.powf(-1.0 / 3.0);
    let mut r = vec![0.0; ng];
    let mut dr = vec![0.0; ng];
    for i in 0..ng {
        let xi = x[i].acos() * dnp;
        let ri = r0 * (1.0 + eps * xi.cos());
        r[i] = ri * ri;
        dr[i] = -r0 * eps * dnp * xi.sin() / ri;
    }
    (r, dr)
}

/// Finite circular cylinder.
///
/// `rev` — equal-volume-sphere radius, `eps` — diameter/length ratio.
pub fn rsp3(x: &[f64], rev: f64, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let ngauss = x.len() / 2;
    let ng = x.len();
    let h = rev * (2.0 / (3.0 * eps * eps)).cbrt();
    let a = h * eps;
    let mut r = vec![0.0; ng];
    let mut dr = vec![0.0; ng];
    for i in 0..ngauss {
        let co = -x[i];
        let si = (1.0 - co * co).sqrt();
        let (rad, rthet) = if si / co > a / h {
            let rad = a / si;
            let rthet = -a * co / (si * si);
            (rad, rthet)
        } else {
            let rad = h / co;
            let rthet = h * si / (co * co);
            (rad, rthet)
        };
        r[i] = rad * rad;
        r[ng - 1 - i] = r[i];
        dr[i] = -rthet / rad;
        dr[ng - 1 - i] = -dr[i];
    }
    (r, dr)
}

/// Raindrop (generalised Chebyshev) shape — RSP4.
///
/// Uses the coefficients stored in `c` (c[0..=nc]) and a normalising factor `r0v`
/// produced by [`drop`].
pub fn rsp4(x: &[f64], rev: f64, c: &[f64], r0v: f64) -> (Vec<f64>, Vec<f64>) {
    let ng = x.len();
    let r0 = rev * r0v;
    let nc = c.len() - 1;
    let mut r = vec![0.0; ng];
    let mut dr = vec![0.0; ng];
    for i in 0..ng {
        let xi = x[i].acos();
        let mut ri = 1.0 + c[0];
        let mut dri = 0.0;
        for n in 1..=nc {
            let xin = xi * n as f64;
            ri += c[n] * xin.cos();
            dri -= c[n] * n as f64 * xin.sin();
        }
        let ri = ri * r0;
        let dri = dri * r0;
        r[i] = ri * ri;
        dr[i] = dri / ri;
    }
    (r, dr)
}

/// Equal-surface-area to equal-volume radius ratio for a spheroid — port of `SAREA`.
pub fn sarea(eps: f64) -> f64 {
    if (eps - 1.0).abs() < 1e-12 {
        return 1.0;
    }
    if eps < 1.0 {
        // prolate
        let e = (1.0 - eps * eps).sqrt();
        let r = 0.5 * (eps.powf(2.0 / 3.0) + eps.powf(-1.0 / 3.0) * e.asin() / e);
        (1.0 / r).sqrt()
    } else {
        // oblate
        let e = (1.0 - 1.0 / (eps * eps)).sqrt();
        let r = 0.25
            * (2.0 * eps.powf(2.0 / 3.0) + eps.powf(-4.0 / 3.0) * ((1.0 + e) / (1.0 - e)).ln() / e);
        (1.0 / r).sqrt()
    }
}

/// Equal-surface-area to equal-volume ratio for a finite cylinder — `SAREAC`.
pub fn sareac(eps: f64) -> f64 {
    // r = (1.5 / eps)^{1/3} / ( (eps^2 + 2) / (3 * eps^{2/3}) )^{1/2} ... etc.
    // Port of Mishchenko's SAREAC.
    let r = (1.5 / eps).powf(1.0 / 3.0);
    let r = r * (1.0 + 2.0 * eps * eps / 3.0).sqrt() / (eps.powf(1.0 / 3.0) * 2.0f64.cbrt());
    // Fortran code:
    //   RAT = (1.5/EPS)**(1D0/3D0)
    //   RAT = RAT / ((2D0*(1D0+EPS*EPS))**(0.5D0))  ... approximate
    // The exact form (from ampld.lp.f):
    //   RAT = (1.5D0/EPS)**(1D0/3D0)
    //   RAT = RAT/( (EPS+2D0)/(3D0*EPS**(2D0/3D0)) )
    // Using that:
    let rat = (1.5 / eps).powf(1.0 / 3.0) / ((eps + 2.0) / (3.0 * eps.powf(2.0 / 3.0)));
    // We return the above; suppress unused variable warning.
    let _ = r;
    rat
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn sphere_has_unit_sarea() {
        assert_abs_diff_eq!(sarea(1.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rsp1_sphere_has_constant_radius() {
        let x = [-0.5, -0.1, 0.1, 0.5];
        let (r, dr) = rsp1(&x, 1.0, 1.0);
        for &rr in &r {
            assert_abs_diff_eq!(rr, 1.0, epsilon = 1e-13);
        }
        for &drr in &dr {
            assert_abs_diff_eq!(drr, 0.0, epsilon = 1e-13);
        }
    }
}
