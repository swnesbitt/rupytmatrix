"""Orientation-averaging strategies and PDFs.

Port of ``pytmatrix.orientation``. The three ``orient_*`` functions all
take a :class:`~rustmatrix.scatterer.Scatterer` instance and return the
``(S, Z)`` pair averaged (or not) over the Euler angles ``(alpha, beta)``
according to the scatterer's ``or_pdf``.

The module is pure Python — it calls :meth:`Scatterer.get_SZ_single`
repeatedly with different orientations, relying on the Rust core only
for the per-orientation evaluation.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.integrate import dblquad, quad


def gaussian_pdf(std: float = 10.0, mean: float = 0.0) -> Callable[[float], float]:
    """Gaussian orientation PDF with the spherical Jacobian baked in.

    Parameters
    ----------
    std : float
        Standard deviation of the Gaussian in degrees. Default 10°.
    mean : float
        Mean canting angle β in degrees. Default 0° (vertical symmetry
        axis). For horizontally-oriented falling particles use 90°.

    Returns
    -------
    pdf : callable
        ``pdf(beta)`` with ``beta`` in degrees. Includes the ``sin(β)``
        spherical weight and is normalised to integrate to 1 on ``[0, 180]``.

    Examples
    --------
    >>> pdf = gaussian_pdf(std=20.0, mean=90.0)
    >>> pdf(90.0) > pdf(60.0)
    True
    """
    norm_const = [1.0]

    def pdf(x):
        return (
            norm_const[0]
            * np.exp(-0.5 * ((x - mean) / std) ** 2)
            * np.sin(np.pi / 180.0 * x)
        )

    norm_dev = quad(pdf, 0.0, 180.0)[0]
    norm_const[0] /= norm_dev
    return pdf


def uniform_pdf() -> Callable[[float], float]:
    """Uniform-on-the-sphere orientation PDF.

    Returns
    -------
    pdf : callable
        ``pdf(beta)`` equal to ``sin(β·π/180) / 2``, so that ∫pdf dβ on
        ``[0, 180]`` equals 1. Use this when the particle has no preferred
        orientation.
    """
    norm_const = [1.0]

    def pdf(x):
        return norm_const[0] * np.sin(np.pi / 180.0 * x)

    norm_dev = quad(pdf, 0.0, 180.0)[0]
    norm_const[0] /= norm_dev
    return pdf


def orient_single(tm) -> Tuple[np.ndarray, np.ndarray]:
    """No averaging — evaluate S, Z at the scatterer's current (α, β).

    Parameters
    ----------
    tm : Scatterer

    Returns
    -------
    S : ndarray (2, 2) complex
    Z : ndarray (4, 4) float
    """
    return tm.get_SZ_single()


def orient_averaged_adaptive(tm) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive orientation averaging via ``scipy.integrate.dblquad``.

    Integrates each of the 4 (re, im) components of ``S`` and the 16
    components of ``Z`` separately over ``α ∈ [0, 360], β ∈ [0, 180]``,
    weighted by ``tm.or_pdf(β)`` and divided by 360 for the uniform α.

    Parameters
    ----------
    tm : Scatterer
        Must have ``tm.or_pdf`` set.

    Returns
    -------
    S : ndarray (2, 2) complex
    Z : ndarray (4, 4) float

    Notes
    -----
    Slow: many T-matrix evaluations per diameter. Prefer
    :func:`orient_averaged_fixed` in production; reserve this for
    reference runs. The Rust PSD fast path replaces this per-diameter
    with a dense fixed grid and runs ~400× faster overall — see
    :meth:`PSDIntegrator.init_scatter_table`.
    """
    S = np.zeros((2, 2), dtype=complex)
    Z = np.zeros((4, 4))

    def Sfunc(beta, alpha, i, j, real):
        S_ang, _ = tm.get_SZ_single(alpha=alpha, beta=beta)
        s = S_ang[i, j].real if real else S_ang[i, j].imag
        return s * tm.or_pdf(beta)

    for i in range(2):
        for j in range(2):
            S.real[i, j] = dblquad(
                Sfunc, 0.0, 360.0, lambda x: 0.0, lambda x: 180.0, (i, j, True)
            )[0] / 360.0
            S.imag[i, j] = dblquad(
                Sfunc, 0.0, 360.0, lambda x: 0.0, lambda x: 180.0, (i, j, False)
            )[0] / 360.0

    def Zfunc(beta, alpha, i, j):
        _, Z_ang = tm.get_SZ_single(alpha=alpha, beta=beta)
        return Z_ang[i, j] * tm.or_pdf(beta)

    for i in range(4):
        for j in range(4):
            Z[i, j] = dblquad(
                Zfunc, 0.0, 360.0, lambda x: 0.0, lambda x: 180.0, (i, j)
            )[0] / 360.0

    return S, Z


def orient_averaged_fixed(tm) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-quadrature orientation averaging.

    α is sampled uniformly at ``tm.n_alpha`` points on ``[0, 360)``; β is
    integrated by Gauss-Gautschi quadrature against ``tm.or_pdf`` with
    nodes ``tm.beta_p`` and weights ``tm.beta_w`` (populated by
    :meth:`Scatterer._init_orient`).

    Parameters
    ----------
    tm : Scatterer

    Returns
    -------
    S : ndarray (2, 2) complex
    Z : ndarray (4, 4) float

    Notes
    -----
    Much faster than :func:`orient_averaged_adaptive` and accurate to a
    few hundredths of a dB in Z_dr for smooth Gaussian PDFs. Default
    choice for orientation-averaged radar forward modelling.
    """
    S = np.zeros((2, 2), dtype=complex)
    Z = np.zeros((4, 4))
    ap = np.linspace(0, 360, tm.n_alpha + 1)[:-1]
    aw = 1.0 / tm.n_alpha

    for alpha in ap:
        for beta, w in zip(tm.beta_p, tm.beta_w):
            S_ang, Z_ang = tm.get_SZ_single(alpha=alpha, beta=beta)
            S += w * S_ang
            Z += w * Z_ang

    sw = tm.beta_w.sum()
    S *= aw / sw
    Z *= aw / sw

    return S, Z
