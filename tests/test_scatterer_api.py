"""Smoke tests for the Python-level Scatterer API.

These don't assert numerical parity — they just verify the API surface
and basic shape / type contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from rupytmatrix import Scatterer


def test_scatterer_defaults():
    s = Scatterer()
    assert s.radius == 1.0
    assert s.wavelength == 1.0
    assert s.axis_ratio == 1.0
    assert s.shape == Scatterer.SHAPE_SPHEROID


def test_scatterer_kwargs_override():
    s = Scatterer(radius=2.5, wavelength=5.0, m=complex(1.5, 0.01), axis_ratio=1.2)
    assert s.radius == 2.5
    assert s.wavelength == 5.0
    assert s.m == complex(1.5, 0.01)
    assert s.axis_ratio == 1.2


def test_set_get_geometry_roundtrip():
    s = Scatterer()
    geom = (45.0, 135.0, 10.0, 190.0, 30.0, 60.0)
    s.set_geometry(geom)
    assert s.get_geometry() == geom


def test_sphere_get_sz_shapes():
    # For a small sphere we expect (S, Z) with shapes (2,2) complex and (4,4) real.
    s = Scatterer(radius=1.0, wavelength=6.283185307, axis_ratio=1.0,
                  m=complex(1.5, 0.01), ddelt=1e-3, ndgs=2)
    s.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    S, Z = s.get_SZ()
    assert S.shape == (2, 2)
    assert S.dtype == np.complex128
    assert Z.shape == (4, 4)
    assert Z.dtype == np.float64


def test_deprecated_alias_warns():
    with pytest.warns(DeprecationWarning):
        Scatterer(axi=1.0, lam=6.28)


def test_equal_volume_from_maximum_spheroid():
    s = Scatterer(shape=Scatterer.SHAPE_SPHEROID, radius=2.0, axis_ratio=2.0)
    # oblate: r_eq = r_max / eps^{1/3}
    expected = 2.0 / 2.0 ** (1.0 / 3.0)
    assert s.equal_volume_from_maximum() == pytest.approx(expected)
