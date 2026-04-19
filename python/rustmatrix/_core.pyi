"""Type stubs for the Rust extension module.

These are thin signatures; the narrative docstrings live on the pure-Python
wrappers in :mod:`rustmatrix` (scatterer, psd, etc.) that end users
interact with. Each function here is the low-level PyO3 entry point.
"""

from __future__ import annotations

import numpy as np

RADIUS_EQUAL_VOLUME: float
"""radius_type flag: Scatterer.radius is the equal-volume-sphere radius."""

RADIUS_EQUAL_AREA: float
"""radius_type flag: Scatterer.radius is the equal-surface-area radius."""

RADIUS_MAXIMUM: float
"""radius_type flag: Scatterer.radius is the maximum dimension."""

SHAPE_SPHEROID: int
"""shape flag: oblate/prolate spheroid (default, ``-1``)."""

SHAPE_CYLINDER: int
"""shape flag: finite cylinder (``-2``)."""

SHAPE_CHEBYSHEV: int
"""shape flag: Chebyshev particle (``1``)."""


class TMatrixHandle:
    """Opaque handle to a built T-matrix (held on the Rust side)."""
    nmax: int
    ngauss: int
    def __repr__(self) -> str: ...


def calctmat(
    axi: float,
    rat: float,
    lam: float,
    mrr: float,
    mri: float,
    eps: float,
    np: int,
    ddelt: float,
    ndgs: int,
) -> tuple[TMatrixHandle, int]:
    """Build a T-matrix. Returns ``(handle, nmax)``."""
    ...


def calcampl_py(
    handle: TMatrixHandle,
    lam: float,
    thet0: float,
    thet: float,
    phi0: float,
    phi: float,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude + phase matrix at one orientation & geometry."""
    ...


def mie_qsca(x: float, mrr: float, mri: float) -> float:
    """Closed-form Mie scattering efficiency Q_sca."""
    ...


def mie_qext(x: float, mrr: float, mri: float) -> float:
    """Closed-form Mie extinction efficiency Q_ext."""
    ...


def tabulate_scatter_table(
    diameters: np.ndarray,
    axis_ratios: np.ndarray,
    ms_real: np.ndarray,
    ms_imag: np.ndarray,
    geometries: list[tuple[float, float, float, float, float, float]],
    rat: float,
    lam: float,
    np: int,
    ddelt: float,
    ndgs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate S(D), Z(D) in parallel across diameters (single-orient)."""
    ...


def tabulate_scatter_table_orient_avg(
    diameters: np.ndarray,
    axis_ratios: np.ndarray,
    ms_real: np.ndarray,
    ms_imag: np.ndarray,
    geometries: list[tuple[float, float, float, float, float, float]],
    alphas: np.ndarray,
    betas: np.ndarray,
    beta_weights: np.ndarray,
    rat: float,
    lam: float,
    np: int,
    ddelt: float,
    ndgs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate orientation-averaged S(D), Z(D) in parallel across diameters."""
    ...


def tabulate_scatter_table_with_angular(
    diameters: np.ndarray,
    axis_ratios: np.ndarray,
    ms_real: np.ndarray,
    ms_imag: np.ndarray,
    geometries: list[tuple[float, float, float, float, float, float]],
    thet_nodes: np.ndarray,
    thet_weights: np.ndarray,
    phi_nodes: np.ndarray,
    phi_weights: np.ndarray,
    rat: float,
    lam: float,
    np: int,
    ddelt: float,
    ndgs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tabulate S, Z, σ_sca, σ_ext, g per diameter (single-orient + angular)."""
    ...
