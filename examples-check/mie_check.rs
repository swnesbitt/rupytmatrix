use rupytmatrix::mie;
use num_complex::Complex64;

fn main() {
    // Check against scipy.special.Mie / Wiscombe table values.
    // x=1, m=1.5 + 0.0i: Q_ext ≈ 0.35281 (Wiscombe table 1, 1979)
    let qext = mie::qext(1.0, Complex64::new(1.5, 0.0));
    println!("Q_ext(x=1, m=1.5)   = {qext:.6}  (Wiscombe: 0.35281)");
    let qsca = mie::qsca(1.0, Complex64::new(1.5, 0.0));
    println!("Q_sca(x=1, m=1.5)   = {qsca:.6}  (Wiscombe: 0.35281)");
    // Absorbing: x=10, m=1.33+0.01i: Q_ext ≈ 2.082, Q_sca ≈ 2.006
    let qext2 = mie::qext(10.0, Complex64::new(1.33, 0.01));
    let qsca2 = mie::qsca(10.0, Complex64::new(1.33, 0.01));
    println!("Q_ext(x=10, m=1.33+0.01i) = {qext2:.4}  (scipy: ~2.082)");
    println!("Q_sca(x=10, m=1.33+0.01i) = {qsca2:.4}  (scipy: ~2.006)");
}
