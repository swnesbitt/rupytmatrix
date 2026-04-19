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
    """Radar cross-section σ [mm²] for the scatterer's current geometry.

    Parameters
    ----------
    scatterer : Scatterer
    h_pol : bool
        True (default) for horizontal polarisation, False for vertical.

    Returns
    -------
    sigma : float
        Polarised radar cross-section. When ``scatterer.psd_integrator`` is
        set, the PSD-integrated value is returned.
    """
    Z = scatterer.get_Z()
    if h_pol:
        return 2 * np.pi * (Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1])
    return 2 * np.pi * (Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1])


def refl(scatterer, h_pol: bool = True):
    """Linear reflectivity [mm⁶·m⁻³] (a.k.a. Z_h or Z_v).

    Parameters
    ----------
    scatterer : Scatterer
        With ``wavelength`` in mm and ``Kw_sqr`` set. A ``psd_integrator``
        is optional; without one, the value is for N = 1 particle / m³.
    h_pol : bool
        Horizontal (default) or vertical polarisation.

    Returns
    -------
    Z : float
        Linear reflectivity. Convert to dBZ with ``10 * log10(Z)``.

    Notes
    -----
    Uses the radar convention ``Z = λ⁴ / (π⁵ |K_w|²) · σ``. Wavelengths
    and diameters must be in mm for the unit to be mm⁶·m⁻³.
    """
    return (
        scatterer.wavelength ** 4
        / (np.pi ** 5 * scatterer.Kw_sqr)
        * radar_xsect(scatterer, h_pol)
    )


# Compatibility alias mirroring pytmatrix.
Zi = refl


def Zdr(scatterer):
    """Differential reflectivity as a linear H/V ratio.

    Convert to dB with ``10 * log10(Zdr(...))``. Identical to the ratio
    ``refl(s, True) / refl(s, False)``.
    """
    return radar_xsect(scatterer, True) / radar_xsect(scatterer, False)


def delta_hv(scatterer):
    """Backscatter differential phase δ_hv in radians.

    Positive for oblate hydrometeors at typical radar wavelengths. Convert
    to degrees with ``numpy.degrees``.
    """
    Z = scatterer.get_Z()
    return np.arctan2(Z[2, 3] - Z[3, 2], -Z[2, 2] - Z[3, 3])


def rho_hv(scatterer):
    """Copolar correlation coefficient ρ_hv (dimensionless, 0–1).

    Drops from 1 as the particle population becomes more heterogeneous
    in shape, orientation, or composition.
    """
    Z = scatterer.get_Z()
    a = (Z[2, 2] + Z[3, 3]) ** 2 + (Z[3, 2] - Z[2, 3]) ** 2
    b = Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1]
    c = Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1]
    return np.sqrt(a / (b * c))


def Kdp(scatterer):
    """Specific differential phase K_dp in ° per km.

    Parameters
    ----------
    scatterer : Scatterer
        Must be in a forward-scatter geometry (``thet0 == thet`` and
        ``phi0 == phi``). Wavelength must be in mm.

    Returns
    -------
    Kdp : float
        Specific differential phase, ° / km.

    Raises
    ------
    ValueError
        If the geometry is not forward-scatter.
    """
    if scatterer.thet0 != scatterer.thet or scatterer.phi0 != scatterer.phi:
        raise ValueError(
            "A forward scattering geometry is needed to compute the "
            "specific differential phase."
        )
    S = scatterer.get_S()
    return 1e-3 * (180.0 / np.pi) * scatterer.wavelength * (S[1, 1] - S[0, 0]).real


def Ai(scatterer, h_pol: bool = True):
    """Specific attenuation A in dB per km.

    Parameters
    ----------
    scatterer : Scatterer
        In a forward-scatter geometry, with wavelength in mm.
    h_pol : bool
        Horizontal (default) or vertical polarisation.

    Returns
    -------
    A : float
        Specific attenuation in dB / km. Computed from the extinction
        cross-section via the optical theorem.
    """
    return 4.343e-3 * ext_xsect(scatterer, h_pol=h_pol)
