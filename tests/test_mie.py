"""Unit tests for the standalone Mie implementation in the Rust core.

These run without pytmatrix and cover the axis_ratio = 1 limit, which is
where the T-matrix reduces to classical Mie theory.
"""

from __future__ import annotations

import numpy as np
import pytest

from rupytmatrix import mie_qext, mie_qsca


def test_rayleigh_limit_water_sphere():
    # For x → 0 and real m, Q_sca ≈ (8/3) x^4 |(m^2-1)/(m^2+2)|^2.
    x = 0.01
    m = complex(1.33, 0.0)
    q = mie_qsca(x, m.real, m.imag)
    alpha = (m * m - 1) / (m * m + 2)
    expected = (8.0 / 3.0) * x ** 4 * abs(alpha) ** 2
    assert q == pytest.approx(expected, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize(
    "x, mrr, mri, q_ext_min, q_ext_max",
    [
        # Standard Mie regime: 2 <= x <= 10, water-like refractive indices.
        (3.0, 1.5, 0.001, 2.0, 5.0),
        (5.0, 1.33, 0.0, 1.0, 5.0),
        (8.0, 1.33, 0.01, 1.5, 3.5),
    ],
)
def test_qext_in_expected_range(x, mrr, mri, q_ext_min, q_ext_max):
    q = mie_qext(x, mrr, mri)
    assert q_ext_min < q < q_ext_max


def test_scipy_mie_agreement():
    """Cross-check against scipy / miepython if available."""
    pytest.importorskip("scipy")
    try:
        import miepython
    except ImportError:
        pytest.skip("miepython not installed.")
    x = 5.0
    m = complex(1.33, 0.01)
    qext_ref, qsca_ref, _, _ = miepython.mie(m, x)
    qsca_ours = mie_qsca(x, m.real, m.imag)
    qext_ours = mie_qext(x, m.real, m.imag)
    assert qsca_ours == pytest.approx(qsca_ref, rel=1e-4)
    assert qext_ours == pytest.approx(qext_ref, rel=1e-4)
