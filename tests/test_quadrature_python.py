"""Cross-check that the Rust Gauss-Legendre quadrature agrees with NumPy."""

from __future__ import annotations

import numpy as np

# No direct Python binding for quadrature (it's internal), but we can
# integrate polynomials and compare with numpy.polynomial.legendre.
# This test is a placeholder that documents the intended verification
# approach; the corresponding Rust-side unit tests in src/quadrature.rs
# cover the actual check.


def test_numpy_legendre_sanity():
    x, w = np.polynomial.legendre.leggauss(8)
    # Sum of weights = 2 on (-1, 1).
    assert np.isclose(w.sum(), 2.0)
    # Integrates x^7 exactly to 0.
    assert np.isclose(np.sum(w * x ** 7), 0.0, atol=1e-12)
