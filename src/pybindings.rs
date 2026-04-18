//! PyO3 bindings that expose the Rust T-matrix core to Python.
//!
//! The bindings mirror the Fortran entrypoints `calctmat` / `calcampl` that
//! upstream pytmatrix calls. A `TMatrixHandle` opaque class keeps the
//! precomputed `TMatrixState` alive between the two calls (equivalent to
//! Fortran's COMMON blocks).

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::amplitude::calcampl;
use crate::mie;
use crate::tmatrix::{calctmat as rs_calctmat, TMatrixConfig, TMatrixState};

/// Opaque handle wrapping `TMatrixState`. Python calls `calctmat(...)` and
/// gets back this handle plus `nmax`; subsequent `calcampl(handle, ...)`
/// reuses the state.
#[pyclass]
pub struct TMatrixHandle {
    state: TMatrixState,
    lam: f64,
}

#[pymethods]
impl TMatrixHandle {
    #[getter]
    fn nmax(&self) -> usize {
        self.state.nmax
    }

    #[getter]
    fn ngauss(&self) -> usize {
        self.state.ngauss
    }

    fn __repr__(&self) -> String {
        format!(
            "<TMatrixHandle nmax={} ngauss={}>",
            self.state.nmax, self.state.ngauss
        )
    }
}

/// Python-visible `calctmat`. Returns `(handle, nmax)`.
#[pyfunction]
#[pyo3(signature = (axi, rat, lam, mrr, mri, eps, np, ddelt, ndgs))]
pub fn calctmat(
    axi: f64,
    rat: f64,
    lam: f64,
    mrr: f64,
    mri: f64,
    eps: f64,
    np: i32,
    ddelt: f64,
    ndgs: usize,
) -> PyResult<(TMatrixHandle, usize)> {
    if axi <= 0.0 || lam <= 0.0 {
        return Err(PyValueError::new_err("axi and lam must be positive"));
    }
    if eps <= 0.0 {
        return Err(PyValueError::new_err("eps (axis ratio) must be positive"));
    }
    let cfg = TMatrixConfig {
        axi,
        rat,
        lam,
        m: Complex64::new(mrr, mri),
        eps,
        np,
        ddelt,
        ndgs,
    };
    let state = rs_calctmat(cfg);
    let nmax = state.nmax;
    Ok((TMatrixHandle { state, lam }, nmax))
}

/// Python-visible `calcampl`. Returns `(S (2x2 complex128), Z (4x4 float64))`.
#[pyfunction]
#[pyo3(signature = (handle, lam, thet0, thet, phi0, phi, alpha, beta))]
pub fn calcampl_py<'py>(
    py: Python<'py>,
    handle: &TMatrixHandle,
    lam: f64,
    thet0: f64,
    thet: f64,
    phi0: f64,
    phi: f64,
    alpha: f64,
    beta: f64,
) -> PyResult<(Bound<'py, PyArray2<Complex64>>, Bound<'py, PyArray2<f64>>)> {
    // Allow overriding lam (mirrors Fortran signature); otherwise use the one
    // cached on the handle.
    let lam_eff = if lam > 0.0 { lam } else { handle.lam };
    let (s, z) = calcampl(&handle.state, lam_eff, thet0, thet, phi0, phi, alpha, beta);
    let s_arr = ndarray::Array2::from_shape_fn((2, 2), |(i, j)| s[i][j]);
    let z_arr = ndarray::Array2::from_shape_fn((4, 4), |(i, j)| z[i][j]);
    Ok((s_arr.into_pyarray_bound(py), z_arr.into_pyarray_bound(py)))
}

/// Mie scattering efficiency — exposed for testing and convenience.
#[pyfunction]
pub fn mie_qsca(x: f64, mrr: f64, mri: f64) -> f64 {
    mie::qsca(x, Complex64::new(mrr, mri))
}

#[pyfunction]
pub fn mie_qext(x: f64, mrr: f64, mri: f64) -> f64 {
    mie::qext(x, Complex64::new(mrr, mri))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TMatrixHandle>()?;
    m.add_function(wrap_pyfunction!(calctmat, m)?)?;
    m.add_function(wrap_pyfunction!(calcampl_py, m)?)?;
    m.add_function(wrap_pyfunction!(mie_qsca, m)?)?;
    m.add_function(wrap_pyfunction!(mie_qext, m)?)?;
    // Shape constants.
    m.add("SHAPE_SPHEROID", -1i32)?;
    m.add("SHAPE_CYLINDER", -2i32)?;
    m.add("SHAPE_CHEBYSHEV", 1i32)?;
    m.add("RADIUS_EQUAL_VOLUME", 1.0_f64)?;
    m.add("RADIUS_EQUAL_AREA", 0.0_f64)?;
    m.add("RADIUS_MAXIMUM", 2.0_f64)?;
    Ok(())
}
