"""Angular-integrated scattering quantities.

Port of ``pytmatrix.scatter``. These helpers operate on a
:class:`~rustmatrix.scatterer.Scatterer` (not the Rust core directly)
and are used by :class:`~rustmatrix.psd.PSDIntegrator` to tabulate
scattering / extinction cross-sections and asymmetry per diameter.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import dblquad

_deg_to_rad = np.pi / 180.0
_rad_to_deg = 180.0 / np.pi


def sca_intensity(scatterer, h_pol: bool = True) -> float:
    """Differential scattering cross-section (value of the phase function).

    Parameters
    ----------
    scatterer : Scatterer
    h_pol : bool

    Returns
    -------
    float
        ``Z[0,0] ± Z[0,1]`` depending on polarisation.
    """
    Z = scatterer.get_Z()
    return (Z[0, 0] - Z[0, 1]) if h_pol else (Z[0, 0] + Z[0, 1])


def ldr(scatterer, h_pol: bool = True) -> float:
    """Linear depolarisation ratio (linear H/H or V/V ratio).

    Convert to dB with ``10 * log10(ldr(...))``.
    """
    Z = scatterer.get_Z()
    if h_pol:
        return (Z[0, 0] - Z[0, 1] + Z[1, 0] - Z[1, 1]) / (
            Z[0, 0] - Z[0, 1] - Z[1, 0] + Z[1, 1]
        )
    return (Z[0, 0] + Z[0, 1] - Z[1, 0] - Z[1, 1]) / (
        Z[0, 0] + Z[0, 1] + Z[1, 0] + Z[1, 1]
    )


def sca_xsect(scatterer, h_pol: bool = True) -> float:
    """Polarised scattering cross-section σ_sca [mm²].

    Integrates ``sca_intensity · sin(θ)`` over the full scattering sphere
    using :func:`scipy.integrate.dblquad`. When a ``psd_integrator`` is
    attached, the pre-tabulated value is returned instead (much faster).
    """
    if scatterer.psd_integrator is not None:
        return scatterer.psd_integrator.get_angular_integrated(
            scatterer.psd, scatterer.get_geometry(), "sca_xsect", h_pol=h_pol
        )

    old_geom = scatterer.get_geometry()

    def d_xsect(thet, phi):
        scatterer.phi = phi * _rad_to_deg
        scatterer.thet = thet * _rad_to_deg
        I = sca_intensity(scatterer, h_pol)
        return I * np.sin(thet)

    try:
        xsect = dblquad(d_xsect, 0.0, 2 * np.pi, lambda x: 0.0, lambda x: np.pi)[0]
    finally:
        scatterer.set_geometry(old_geom)
    return xsect


def ext_xsect(scatterer, h_pol: bool = True) -> float:
    """Extinction cross-section σ_ext [mm²] from the optical theorem.

    Temporarily rotates the scatterer into forward-scatter geometry,
    evaluates ``S``, and returns ``2 λ Im(S_ii)``. Restores the original
    geometry before returning.
    """
    if scatterer.psd_integrator is not None:
        try:
            return scatterer.psd_integrator.get_angular_integrated(
                scatterer.psd, scatterer.get_geometry(), "ext_xsect", h_pol=h_pol
            )
        except AttributeError:
            # Fall back if the table wasn't populated.
            pass

    old_geom = scatterer.get_geometry()
    thet0, thet, phi0, phi, alpha, beta = old_geom
    try:
        # Forward-scattering geometry (thet == thet0, phi == phi0) for the
        # optical theorem: sigma_ext = (2 lambda) * Im(S_forward).
        scatterer.set_geometry((thet0, thet0, phi0, phi0, alpha, beta))
        S = scatterer.get_S()
    finally:
        scatterer.set_geometry(old_geom)

    if h_pol:
        return 2 * scatterer.wavelength * S[1, 1].imag
    return 2 * scatterer.wavelength * S[0, 0].imag


def ssa(scatterer, h_pol: bool = True) -> float:
    """Single-scattering albedo ω = σ_sca / σ_ext (dimensionless, 0–1).

    Returns 0 when σ_ext is zero (e.g. perfectly non-attenuating medium).
    """
    ext = ext_xsect(scatterer, h_pol=h_pol)
    return sca_xsect(scatterer, h_pol=h_pol) / ext if ext > 0.0 else 0.0


def asym(scatterer, h_pol: bool = True) -> float:
    """Asymmetry parameter g = ⟨cos Θ⟩ (dimensionless).

    g = 0 for isotropic scattering, g → 1 for forward-peaked scattering,
    g < 0 for backscattering-dominated regimes.
    """
    if scatterer.psd_integrator is not None:
        return scatterer.psd_integrator.get_angular_integrated(
            scatterer.psd, scatterer.get_geometry(), "asym", h_pol=h_pol
        )

    old_geom = scatterer.get_geometry()
    cos_t0 = np.cos(scatterer.thet0 * _deg_to_rad)
    sin_t0 = np.sin(scatterer.thet0 * _deg_to_rad)
    p0 = scatterer.phi0 * _deg_to_rad

    def integrand(thet, phi):
        scatterer.phi = phi * _rad_to_deg
        scatterer.thet = thet * _rad_to_deg
        cos_T_sin_t = 0.5 * (
            np.sin(2 * thet) * cos_t0
            + (1 - np.cos(2 * thet)) * sin_t0 * np.cos(p0 - phi)
        )
        I = sca_intensity(scatterer, h_pol)
        return I * cos_T_sin_t

    try:
        cos_int = dblquad(
            integrand, 0.0, 2 * np.pi, lambda x: 0.0, lambda x: np.pi
        )[0]
    finally:
        scatterer.set_geometry(old_geom)

    return cos_int / sca_xsect(scatterer, h_pol)
