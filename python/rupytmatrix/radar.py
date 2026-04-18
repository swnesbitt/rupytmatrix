"""Polarimetric radar observables.

Direct port of ``pytmatrix.radar``. Given a configured ``Scatterer``
(optionally with a ``psd_integrator`` for N(D)-weighted averages), these
helpers compute the standard radar observables:

* :func:`radar_xsect` — radar cross-section.
* :func:`refl` / :func:`Zi` — reflectivity (with N=1 when no PSD).
* :func:`Zdr` — differential reflectivity (linear ratio).
* :func:`delta_hv` — backscatter differential phase.
* :func:`rho_hv` — copolar correlation coefficient.
* :func:`Kdp` — specific differential phase [°/km] (forward geometry).
* :func:`Ai` — specific attenuation [dB/km] (forward geometry).

Inputs: particle diameter and wavelength must be given in mm for ``Kdp``
and ``Ai`` to come out in the documented units. ``Zi`` returns linear
reflectivity; take ``10 * log10(Zi)`` for dBZ.
"""

from __future__ import annotations

import numpy as np

from .scatter import ext_xsect


def radar_xsect(scatterer, h_pol: bool = True):
    """Radar cross-section for the scatterer's current setup.

    Args:
        scatterer: a :class:`Scatterer` instance.
        h_pol: If True (default), horizontal polarisation; otherwise vertical.
    """
    Z = scatterer.get_Z()
    if h_pol:
        return 2 * np.pi * (Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1])
    return 2 * np.pi * (Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1])


def refl(scatterer, h_pol: bool = True):
    """Reflectivity (with number concentration N=1 when no PSD is set).

    For dBZ use ``10 * log10(refl(...))`` with diameter and wavelength in mm.
    """
    return (
        scatterer.wavelength ** 4
        / (np.pi ** 5 * scatterer.Kw_sqr)
        * radar_xsect(scatterer, h_pol)
    )


# Compatibility alias mirroring pytmatrix.
Zi = refl


def Zdr(scatterer):
    """Differential reflectivity Z_dr (linear H/V ratio)."""
    return radar_xsect(scatterer, True) / radar_xsect(scatterer, False)


def delta_hv(scatterer):
    """Backscatter differential phase δ_hv [rad]."""
    Z = scatterer.get_Z()
    return np.arctan2(Z[2, 3] - Z[3, 2], -Z[2, 2] - Z[3, 3])


def rho_hv(scatterer):
    """Copolar correlation coefficient ρ_hv."""
    Z = scatterer.get_Z()
    a = (Z[2, 2] + Z[3, 3]) ** 2 + (Z[3, 2] - Z[2, 3]) ** 2
    b = Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1]
    c = Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1]
    return np.sqrt(a / (b * c))


def Kdp(scatterer):
    """Specific differential phase K_dp [°/km].

    Requires a forward-scatter geometry (thet0==thet, phi0==phi).
    Wavelength and diameters must be in mm.
    """
    if scatterer.thet0 != scatterer.thet or scatterer.phi0 != scatterer.phi:
        raise ValueError(
            "A forward scattering geometry is needed to compute the "
            "specific differential phase."
        )
    S = scatterer.get_S()
    return 1e-3 * (180.0 / np.pi) * scatterer.wavelength * (S[1, 1] - S[0, 0]).real


def Ai(scatterer, h_pol: bool = True):
    """Specific attenuation A [dB/km] (forward geometry, mm inputs)."""
    return 4.343e-3 * ext_xsect(scatterer, h_pol=h_pol)
