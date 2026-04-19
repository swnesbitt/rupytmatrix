"""Unit tests for the radar-observables module.

These use a water sphere at C-band-ish wavelength — for a sphere, Z_dr
should be unity, delta_hv and Kdp should vanish, and rho_hv should be 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from rustmatrix import Scatterer, radar
from rustmatrix.tmatrix_aux import geom_horiz_back, geom_horiz_forw


@pytest.fixture
def sphere_back():
    s = Scatterer(
        radius=1.0,
        wavelength=6.283185307,
        axis_ratio=1.0,
        m=complex(7.99, 2.21),
        ddelt=1e-4,
        ndgs=2,
    )
    s.set_geometry(geom_horiz_back)
    return s


@pytest.fixture
def sphere_forward():
    s = Scatterer(
        radius=1.0,
        wavelength=6.283185307,
        axis_ratio=1.0,
        m=complex(7.99, 2.21),
        ddelt=1e-4,
        ndgs=2,
    )
    s.set_geometry(geom_horiz_forw)
    return s


def test_radar_xsect_real_and_positive(sphere_back):
    sig = radar.radar_xsect(sphere_back)
    assert sig.imag == pytest.approx(0.0, abs=1e-12)
    assert sig.real > 0


def test_zdr_is_unity_for_sphere(sphere_back):
    np.testing.assert_allclose(radar.Zdr(sphere_back), 1.0, rtol=1e-6)


def test_delta_hv_zero_for_sphere(sphere_back):
    # Backscatter differential phase is 0 for an isotropic sphere.
    np.testing.assert_allclose(radar.delta_hv(sphere_back), 0.0, atol=1e-6)


def test_rho_hv_unity_for_sphere(sphere_back):
    np.testing.assert_allclose(radar.rho_hv(sphere_back), 1.0, rtol=1e-6)


def test_refl_dBZ_matches_formula(sphere_back):
    """``refl`` is (wl^4 / (pi^5 K_w^2)) * sigma_h."""
    sig = radar.radar_xsect(sphere_back, h_pol=True)
    expected = sphere_back.wavelength ** 4 / (np.pi ** 5 * sphere_back.Kw_sqr) * sig
    np.testing.assert_allclose(radar.refl(sphere_back), expected, rtol=1e-10)


def test_zi_alias(sphere_back):
    assert radar.Zi is radar.refl


def test_kdp_requires_forward_geometry(sphere_back):
    with pytest.raises(ValueError):
        radar.Kdp(sphere_back)


def test_kdp_zero_for_sphere_forward(sphere_forward):
    np.testing.assert_allclose(radar.Kdp(sphere_forward), 0.0, atol=1e-6)


def test_ai_positive_forward(sphere_forward):
    # Specific attenuation must be positive for a lossy particle.
    assert radar.Ai(sphere_forward, h_pol=True) > 0
