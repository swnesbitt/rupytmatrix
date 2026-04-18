"""Unit tests for refractive-index helpers."""

from __future__ import annotations

import numpy as np
import pytest

from rupytmatrix import refractive
from rupytmatrix.tmatrix_aux import wl_C, wl_S, wl_X


# ---------- effective-medium approximations ----------

def test_mg_refractive_air_matrix_returns_unity():
    """Pure air matrix with zero inclusion fraction → vacuum refractive index."""
    m = refractive.mg_refractive((complex(1.0, 0.0), complex(1.5, 0.01)), (1.0, 0.0))
    assert m == pytest.approx(complex(1.0, 0.0))


def test_mg_refractive_pure_inclusion_returns_inclusion_index():
    m_incl = complex(1.78, 0.002)  # ice-ish
    m = refractive.mg_refractive((complex(1.0, 0.0), m_incl), (0.0, 1.0))
    np.testing.assert_allclose(m, m_incl, rtol=1e-10)


def test_mg_refractive_three_component_reduces_to_two_component():
    """If one fraction is zero, three-component MG should match two-component."""
    m_three = refractive.mg_refractive(
        (complex(1.0, 0.0), complex(1.78, 0.002), complex(1.33, 0.01)),
        (0.5, 0.5, 0.0),
    )
    m_two = refractive.mg_refractive(
        (complex(1.0, 0.0), complex(1.78, 0.002)), (0.5, 0.5)
    )
    # The three-component path mixes recursively; with zero volume of the
    # last component it must collapse onto the two-component answer.
    np.testing.assert_allclose(m_three, m_two, rtol=1e-10)


def test_bruggeman_symmetry_pure_components():
    """Bruggeman with a 100/0 mix should return the single-component index."""
    m_a = complex(1.0, 0.0)
    m_b = complex(1.78, 0.002)
    # Pure a.
    m = refractive.bruggeman_refractive((m_a, m_b), (1.0, 0.0))
    np.testing.assert_allclose(m, m_a, rtol=1e-10)


def test_bruggeman_matches_mg_limit_for_small_inclusion():
    """For dilute inclusions Bruggeman ≈ Maxwell-Garnett."""
    m_a = complex(1.0, 0.0)
    m_b = complex(1.78, 0.002)
    m_bg = refractive.bruggeman_refractive((m_a, m_b), (0.99, 0.01))
    m_mg = refractive.mg_refractive((m_a, m_b), (0.99, 0.01))
    np.testing.assert_allclose(m_bg, m_mg, rtol=5e-3)


# ---------- tabulated water indices ----------

def test_water_tables_cover_all_radar_bands():
    bands = {wl_S, wl_C, wl_X}
    for T in (refractive.m_w_0C, refractive.m_w_10C, refractive.m_w_20C):
        assert bands.issubset(T.keys())


def test_water_imag_part_positive():
    """All tabulated refractive indices have positive imaginary part (lossy)."""
    for T in (refractive.m_w_0C, refractive.m_w_10C, refractive.m_w_20C):
        for m in T.values():
            assert m.imag > 0


# ---------- ice interpolator ----------

def test_ice_interpolator_scalar_and_array():
    """The bundled ``mi`` interpolator handles scalar and array inputs."""
    m_scalar = refractive.mi(wl_X, 0.9167)
    assert isinstance(m_scalar, complex) or np.iscomplexobj(m_scalar)

    m_array = refractive.mi(np.array([wl_X, wl_C]), 0.5)
    assert m_array.shape == (2,)
    assert np.all(np.imag(m_array) > 0)


def test_ice_density_limits():
    """At snow_density=0 the effective medium is air (index ≈ 1)."""
    m_air = refractive.mi(wl_X, 0.0)
    # With snow_density=0 the EMA reduces to the matrix (vacuum/air).
    np.testing.assert_allclose(abs(m_air), 1.0, rtol=1e-10)
