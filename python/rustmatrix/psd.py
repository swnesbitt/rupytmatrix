"""Particle size distributions (PSDs) and size-distribution integration.

Port of ``pytmatrix.psd``. The :class:`PSDIntegrator` builds a lookup
table ``S(D), Z(D)`` at a fixed set of diameters and scattering
geometries, then integrates it against a PSD ``N(D)`` using the
trapezoidal rule. The per-diameter evaluation calls through to the Rust
T-matrix core.

Typical usage::

    sca = Scatterer(wavelength=wl_C, m=complex(7.99, 2.21), axis_ratio=1/0.9)
    sca.psd_integrator = PSDIntegrator()
    sca.psd_integrator.D_max = 8.0
    sca.psd_integrator.num_points = 256
    sca.psd_integrator.init_scatter_table(sca)
    sca.psd = GammaPSD(D0=2.0, Nw=1e3, mu=4)
    S, Z = sca.get_SZ()
"""

from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from scipy.special import gamma

from . import _core, tmatrix_aux
from . import orientation as _orientation

# numpy 2.0 renamed trapz -> trapezoid; scipy >= 1.14 dropped trapz from
# scipy.integrate. Cover both.
try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapezoid


class PSD:
    """Abstract PSD base class.

    Subclasses override ``__call__(D) -> number density [mm⁻¹ m⁻³]`` and
    ``__eq__`` (used by :class:`PSDIntegrator` to decide when the cached
    PSD-weighted integrals are stale). The base class returns 0 for all
    diameters and never equals another PSD.
    """

    def __call__(self, D):
        if np.shape(D) == ():
            return 0.0
        return np.zeros_like(D)

    def __eq__(self, other):
        return False


class ExponentialPSD(PSD):
    """Exponential (Marshall-Palmer-type) PSD.

    ``N(D) = N0 · exp(-Λ·D)`` for ``D ≤ D_max`` and 0 beyond.

    Parameters
    ----------
    N0 : float
        Intercept parameter [mm⁻¹ m⁻³]. Default 1.0.
    Lambda : float
        Slope parameter Λ [mm⁻¹]. Default 1.0.
    D_max : float, optional
        Truncation diameter in mm. Defaults to ``11 / Lambda`` (≈ 3·D0).
    """

    def __init__(self, N0: float = 1.0, Lambda: float = 1.0, D_max: Optional[float] = None):
        self.N0 = float(N0)
        self.Lambda = float(Lambda)
        self.D_max = 11.0 / Lambda if D_max is None else D_max

    def __call__(self, D):
        psd = self.N0 * np.exp(-self.Lambda * D)
        if np.shape(D) == ():
            if D > self.D_max:
                return 0.0
        else:
            psd[D > self.D_max] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return (
                isinstance(other, ExponentialPSD)
                and self.N0 == other.N0
                and self.Lambda == other.Lambda
                and self.D_max == other.D_max
            )
        except AttributeError:
            return False


class UnnormalizedGammaPSD(ExponentialPSD):
    """Unnormalised gamma PSD: ``N(D) = N0 · D^μ · exp(-Λ·D)``.

    Parameters
    ----------
    N0, Lambda, D_max
        As :class:`ExponentialPSD`.
    mu : float
        Shape parameter μ (``0`` reduces to :class:`ExponentialPSD`).

    Notes
    -----
    The ``D^μ`` term is evaluated in log-space to avoid overflow for large
    μ. Taking ``D = 0`` returns 0.
    """

    def __init__(
        self,
        N0: float = 1.0,
        Lambda: float = 1.0,
        mu: float = 0.0,
        D_max: Optional[float] = None,
    ):
        super().__init__(N0=N0, Lambda=Lambda, D_max=D_max)
        self.mu = mu

    def __call__(self, D):
        # log-space is numerically safer than D**mu for large mu
        psd = self.N0 * np.exp(self.mu * np.log(D) - self.Lambda * D)
        if np.shape(D) == ():
            if D > self.D_max or D == 0:
                return 0.0
        else:
            psd[(D > self.D_max) | (D == 0)] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return super().__eq__(other) and self.mu == other.mu
        except AttributeError:
            return False


class GammaPSD(PSD):
    """Normalised gamma PSD (Bringi & Chandrasekar convention).

    ``N(D) = N_w · f(μ) · (D/D0)^μ · exp(-(3.67+μ) D/D0)``, with
    ``f(μ) = 6 / 3.67⁴ · (3.67+μ)^(μ+4) / Γ(μ+4)``.

    Parameters
    ----------
    D0 : float
        Median volume diameter in mm. Default 1.0.
    Nw : float
        Intercept parameter [mm⁻¹ m⁻³]. Default 1.0.
    mu : float
        Shape parameter μ. Default 0 (reduces to exponential).
    D_max : float, optional
        Truncation diameter in mm. Defaults to ``3 · D0``.

    References
    ----------
    Bringi, V. N., & Chandrasekar, V. (2001). *Polarimetric Doppler
    Weather Radar*, Cambridge University Press.
    """

    def __init__(
        self,
        D0: float = 1.0,
        Nw: float = 1.0,
        mu: float = 0.0,
        D_max: Optional[float] = None,
    ):
        self.D0 = float(D0)
        self.mu = float(mu)
        self.D_max = 3.0 * D0 if D_max is None else D_max
        self.Nw = float(Nw)
        self.nf = Nw * 6.0 / 3.67 ** 4 * (3.67 + mu) ** (mu + 4) / gamma(mu + 4)

    def __call__(self, D):
        d = D / self.D0
        psd = self.nf * np.exp(self.mu * np.log(d) - (3.67 + self.mu) * d)
        if np.shape(D) == ():
            if D > self.D_max or D == 0.0:
                return 0.0
        else:
            psd[(D > self.D_max) | (D == 0.0)] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return (
                isinstance(other, GammaPSD)
                and self.D0 == other.D0
                and self.Nw == other.Nw
                and self.mu == other.mu
                and self.D_max == other.D_max
            )
        except AttributeError:
            return False


class BinnedPSD(PSD):
    """Step-function PSD specified by bin edges and per-bin number densities.

    Parameters
    ----------
    bin_edges : sequence of float, length n+1
        Monotonically increasing bin edges in mm.
    bin_psd : sequence of float, length n
        Per-bin number densities [mm⁻¹ m⁻³]. ``bin_psd[i]`` applies to
        ``D`` in ``(bin_edges[i], bin_edges[i+1]]``.
    """

    def __init__(self, bin_edges, bin_psd):
        if len(bin_edges) != len(bin_psd) + 1:
            raise ValueError("There must be n+1 bin edges for n bins.")
        self.bin_edges = bin_edges
        self.bin_psd = bin_psd

    def psd_for_D(self, D):
        if not (self.bin_edges[0] < D <= self.bin_edges[-1]):
            return 0.0
        # Binary search to locate bin.
        start = 0
        end = len(self.bin_edges)
        while end - start > 1:
            half = (start + end) // 2
            if self.bin_edges[start] < D <= self.bin_edges[half]:
                end = half
            else:
                start = half
        return self.bin_psd[start]

    def __call__(self, D):
        if np.shape(D) == ():
            return self.psd_for_D(D)
        return np.array([self.psd_for_D(d) for d in D])

    def __eq__(self, other):
        if other is None:
            return False
        return (
            len(self.bin_edges) == len(other.bin_edges)
            and (np.asarray(self.bin_edges) == np.asarray(other.bin_edges)).all()
            and (np.asarray(self.bin_psd) == np.asarray(other.bin_psd)).all()
        )


class PSDIntegrator:
    """Integrate scattering properties over a particle-size distribution.

    Usage
    -----
    Attach an instance to ``Scatterer.psd_integrator``, configure
    ``num_points``, ``D_max``, optional ``axis_ratio_func``/``m_func``, and
    ``geometries``, then call
    :meth:`init_scatter_table` once to build the diameter-indexed lookup
    tables. Thereafter setting ``Scatterer.psd`` to any :class:`PSD`
    instance and calling :meth:`Scatterer.get_SZ` (or the radar / scatter
    helpers) returns the PSD-integrated result via trapezoidal
    integration over the pre-tabulated ``S(D)`` and ``Z(D)``.

    Attributes
    ----------
    num_points : int
        Number of diameters sampled between ``D_max/num_points`` and
        ``D_max``. Default 1024.
    m_func : callable, optional
        ``m(D) -> complex`` — varies refractive index with diameter.
        ``None`` uses the scatterer's scalar ``m``.
    axis_ratio_func : callable, optional
        ``eps(D) -> float`` — varies axis ratio with diameter (e.g.
        :func:`tmatrix_aux.dsr_thurai_2007`).
    D_max : float
        Largest diameter to tabulate in mm. Must cover every PSD the
        table will be used with.
    geometries : tuple of 6-tuples
        ``(thet0, thet, phi0, phi, alpha, beta)`` scattering geometries
        to precompute. Typically includes both backscatter and forward
        geometries when K_dp / A_i are needed.

    Notes
    -----
    :meth:`init_scatter_table` dispatches to one of four parallel Rust
    fast paths (single orientation, fixed-orient-avg, adaptive-orient-avg,
    single-orient + angular integration); see the module docstring for
    details and the README *Performance* section for benchmarks.
    """

    attrs = {"num_points", "m_func", "axis_ratio_func", "D_max", "geometries"}

    def __init__(self, **kwargs):
        self.num_points = 1024
        self.m_func: Optional[Callable[[float], complex]] = None
        self.axis_ratio_func: Optional[Callable[[float], float]] = None
        self.D_max: Optional[float] = None
        self.geometries = (tmatrix_aux.geom_horiz_back,)

        for k, v in kwargs.items():
            if k in self.attrs:
                setattr(self, k, v)

        self._S_table = None
        self._Z_table = None
        self._angular_table = None
        self._previous_psd = None

    def __call__(self, psd, geometry):
        return self.get_SZ(psd, geometry)

    def get_SZ(self, psd, geometry):
        """PSD-integrated ``(S, Z)`` at the given geometry.

        Parameters
        ----------
        psd : PSD
            Particle-size distribution.
        geometry : 6-tuple
            One of the geometries registered when
            :meth:`init_scatter_table` was called.

        Returns
        -------
        S, Z : ndarray
            PSD-weighted trapezoidal integrals of the precomputed
            ``S(D)`` and ``Z(D)`` lookup tables.
        """
        if self._S_table is None or self._Z_table is None:
            raise AttributeError("Initialize or load the scattering table first.")

        if not isinstance(psd, PSD) or self._previous_psd != psd:
            self._S_dict = {}
            self._Z_dict = {}
            psd_w = psd(self._psd_D)
            for geom in self.geometries:
                # _S_table[geom] has shape (2, 2, num_points); trapezoid
                # integrates along axis=-1 by default.
                self._S_dict[geom] = _trapezoid(
                    self._S_table[geom] * psd_w, self._psd_D
                )
                self._Z_dict[geom] = _trapezoid(
                    self._Z_table[geom] * psd_w, self._psd_D
                )
            self._previous_psd = psd

        return self._S_dict[geometry], self._Z_dict[geometry]

    def get_angular_integrated(self, psd, geometry, property_name, h_pol=True):
        """PSD-integrated angular quantity (sca_xsect / ext_xsect / asym)."""
        if self._angular_table is None:
            raise AttributeError(
                "Initialize or load the table of angular-integrated quantities first."
            )

        pol = "h_pol" if h_pol else "v_pol"
        psd_w = psd(self._psd_D)

        def sca_xsect(geom):
            return _trapezoid(
                self._angular_table["sca_xsect"][pol][geom] * psd_w, self._psd_D
            )

        if property_name == "sca_xsect":
            return sca_xsect(geometry)
        if property_name == "ext_xsect":
            return _trapezoid(
                self._angular_table["ext_xsect"][pol][geometry] * psd_w, self._psd_D
            )
        if property_name == "asym":
            sca_int = sca_xsect(geometry)
            if sca_int <= 0:
                return 0.0
            num = _trapezoid(
                self._angular_table["asym"][pol][geometry]
                * self._angular_table["sca_xsect"][pol][geometry]
                * psd_w,
                self._psd_D,
            )
            return num / sca_int
        raise ValueError(f"Unknown property_name {property_name!r}")

    def init_scatter_table(self, tm, angular_integration: bool = False, verbose: bool = False):
        """Populate the diameter-indexed ``S(D)`` / ``Z(D)`` lookup tables.

        Parameters
        ----------
        tm : Scatterer
            Template scatterer; its ``wavelength``, ``m``, ``ddelt``,
            ``ndgs``, ``shape``, ``radius_type``, ``orient``, and
            ``or_pdf`` are copied.
        angular_integration : bool
            If True, also tabulates the polarised scattering and
            extinction cross-sections and the asymmetry parameter at each
            diameter. Required before calling
            :func:`scatter.sca_xsect`, :func:`scatter.asym`, or their
            radar derivatives (:func:`radar.Ai`) on a PSD-integrated
            scatterer.
        verbose : bool
            Print per-diameter progress when falling back to the Python
            loop (used only for combinations without a Rust fast path).

        Notes
        -----
        Dispatches into one of four Rust fast paths depending on
        ``tm.orient`` and ``angular_integration`` — see the README
        *Performance* section for benchmarks. All four release the GIL
        and parallelise across diameters via rayon.
        """
        if self.D_max is None:
            raise AttributeError("PSDIntegrator.D_max must be set before init_scatter_table.")

        # Deferred to avoid a module-level cycle (scatter imports from here).
        from . import scatter

        self._psd_D = np.linspace(
            self.D_max / self.num_points, self.D_max, self.num_points
        )

        self._S_table = {}
        self._Z_table = {}
        self._previous_psd = None
        self._m_table = np.empty(self.num_points, dtype=complex)

        if angular_integration:
            self._angular_table = {
                "sca_xsect": {"h_pol": {}, "v_pol": {}},
                "ext_xsect": {"h_pol": {}, "v_pol": {}},
                "asym": {"h_pol": {}, "v_pol": {}},
            }
        else:
            self._angular_table = None

        old_m = tm.m
        old_axis_ratio = tm.axis_ratio
        old_radius = tm.radius
        old_geom = tm.get_geometry()
        old_psd_integrator = tm.psd_integrator

        # Evaluate per-diameter m and axis_ratio up front so the Rust
        # tabulator (which can't call back into Python) has plain arrays.
        if self.m_func is not None:
            for i, D in enumerate(self._psd_D):
                self._m_table[i] = self.m_func(D)
        else:
            self._m_table[:] = tm.m
        if self.axis_ratio_func is not None:
            axis_ratios = np.array(
                [self.axis_ratio_func(D) for D in self._psd_D], dtype=float
            )
        else:
            axis_ratios = np.full(self.num_points, float(tm.axis_ratio))

        # Rust fast paths: single orientation, fixed-quadrature orientation
        # averaging, adaptive orientation averaging, and the single-orient
        # ``angular_integration`` path. Each is parallelised across
        # diameters. The fallback Python loop remains for combinations
        # that need per-sample Python callbacks (e.g. angular integration
        # together with orientation averaging).
        orient_fn = getattr(tm, "orient", _orientation.orient_single)
        is_single = orient_fn is _orientation.orient_single
        is_fixed = orient_fn is _orientation.orient_averaged_fixed
        is_adaptive = orient_fn is _orientation.orient_averaged_adaptive
        use_rust_single = not angular_integration and is_single
        use_rust_orient_avg = not angular_integration and (is_fixed or is_adaptive)
        use_rust_angular = angular_integration and is_single

        try:
            # Disable PSD integration on the scatterer to avoid recursion
            # through get_SZ.
            tm.psd_integrator = None

            for geom in self.geometries:
                self._S_table[geom] = np.empty((2, 2, self.num_points), dtype=complex)
                self._Z_table[geom] = np.empty((4, 4, self.num_points))
                if angular_integration:
                    for key in ("sca_xsect", "ext_xsect", "asym"):
                        for pol in ("h_pol", "v_pol"):
                            self._angular_table[key][pol][geom] = np.empty(self.num_points)

            if use_rust_single or use_rust_orient_avg or use_rust_angular:
                geoms = [tuple(g) for g in self.geometries]
                common = (
                    np.ascontiguousarray(self._psd_D, dtype=float),
                    np.ascontiguousarray(axis_ratios, dtype=float),
                    np.ascontiguousarray(self._m_table.real, dtype=float),
                    np.ascontiguousarray(self._m_table.imag, dtype=float),
                    geoms,
                )
                extras = (
                    float(tm.radius_type),
                    float(tm.wavelength),
                    int(tm.shape),
                    float(tm.ddelt),
                    int(tm.ndgs),
                )
                sca_batch = ext_batch = asym_batch = None
                if use_rust_orient_avg:
                    if is_fixed:
                        # Match orient_averaged_fixed: Gautschi quadrature
                        # against or_pdf for β, uniform α.
                        tm._init_orient()
                        alphas = np.linspace(0, 360, tm.n_alpha + 1)[:-1]
                        betas = np.asarray(tm.beta_p, dtype=float)
                        beta_w = np.asarray(tm.beta_w, dtype=float)
                    else:
                        # Match orient_averaged_adaptive: dense uniform α
                        # (32 pts) × Gauss-Legendre β on [0, 180] with
                        # or_pdf(β) folded into the weights. Rust renormalises
                        # by sum(beta_w), so constant prefactors drop out.
                        n_alpha_adapt = 32
                        n_beta_adapt = 32
                        alphas = np.linspace(0, 360, n_alpha_adapt + 1)[:-1]
                        b_nodes, b_w = np.polynomial.legendre.leggauss(n_beta_adapt)
                        betas = 90.0 * (b_nodes + 1.0)  # map [-1,1] -> [0,180]
                        or_pdf = getattr(tm, "or_pdf", _orientation.uniform_pdf())
                        beta_w = b_w * or_pdf(betas)
                    S_batch, Z_batch = _core.tabulate_scatter_table_orient_avg(
                        *common,
                        np.ascontiguousarray(alphas, dtype=float),
                        np.ascontiguousarray(betas, dtype=float),
                        np.ascontiguousarray(beta_w, dtype=float),
                        *extras,
                    )
                elif use_rust_angular:
                    # Gauss-Legendre product grid for (θ, φ) on the full
                    # scattering sphere. 32 × 64 points are enough to match
                    # scipy.dblquad for smooth T-matrix integrands; bump if
                    # the parity tolerance is ever tightened.
                    n_thet = 32
                    n_phi = 64
                    t_nodes, t_w = np.polynomial.legendre.leggauss(n_thet)
                    p_nodes, p_w = np.polynomial.legendre.leggauss(n_phi)
                    thet_rad = 0.5 * np.pi * (t_nodes + 1.0)
                    phi_rad = np.pi * (p_nodes + 1.0)
                    thet_weights = 0.5 * np.pi * t_w
                    phi_weights = np.pi * p_w
                    S_batch, Z_batch, sca_batch, ext_batch, asym_batch = (
                        _core.tabulate_scatter_table_with_angular(
                            *common,
                            np.ascontiguousarray(thet_rad, dtype=float),
                            np.ascontiguousarray(thet_weights, dtype=float),
                            np.ascontiguousarray(phi_rad, dtype=float),
                            np.ascontiguousarray(phi_weights, dtype=float),
                            *extras,
                        )
                    )
                else:
                    S_batch, Z_batch = _core.tabulate_scatter_table(
                        *common, *extras,
                    )
                # S_batch: (num_points, num_geoms, 2, 2); Z_batch: (..., 4, 4).
                # Our on-disk layout is (2, 2, num_points) per geom — reshape.
                for g_idx, geom in enumerate(self.geometries):
                    self._S_table[geom] = np.moveaxis(S_batch[:, g_idx, :, :], 0, -1)
                    self._Z_table[geom] = np.moveaxis(Z_batch[:, g_idx, :, :], 0, -1)
                    if sca_batch is not None:
                        for pol_idx, pol in enumerate(("h_pol", "v_pol")):
                            self._angular_table["sca_xsect"][pol][geom] = (
                                sca_batch[:, g_idx, pol_idx].copy()
                            )
                            self._angular_table["ext_xsect"][pol][geom] = (
                                ext_batch[:, g_idx, pol_idx].copy()
                            )
                            self._angular_table["asym"][pol][geom] = (
                                asym_batch[:, g_idx, pol_idx].copy()
                            )
            else:
                # Fallback: Python loop (orientation-averaged or angular
                # integration). Keeps callbacks like ``tm.orient`` working.
                for i, D in enumerate(self._psd_D):
                    if verbose:
                        print(f"Computing point {i} at D={D}...")
                    tm.m = self._m_table[i]
                    tm.axis_ratio = axis_ratios[i]
                    tm.radius = D / 2.0
                    for geom in self.geometries:
                        tm.set_geometry(geom)
                        S, Z = tm.get_SZ_orient()
                        self._S_table[geom][:, :, i] = S
                        self._Z_table[geom][:, :, i] = Z
                        if angular_integration:
                            for pol in ("h_pol", "v_pol"):
                                h_pol = pol == "h_pol"
                                self._angular_table["sca_xsect"][pol][geom][i] = (
                                    scatter.sca_xsect(tm, h_pol=h_pol)
                                )
                                self._angular_table["ext_xsect"][pol][geom][i] = (
                                    scatter.ext_xsect(tm, h_pol=h_pol)
                                )
                                self._angular_table["asym"][pol][geom][i] = (
                                    scatter.asym(tm, h_pol=h_pol)
                                )
        finally:
            tm.m = old_m
            tm.axis_ratio = old_axis_ratio
            tm.radius = old_radius
            tm.psd_integrator = old_psd_integrator
            tm.set_geometry(old_geom)

    def save_scatter_table(self, fn: str, description: str = "") -> None:
        """Pickle the lookup tables to disk."""
        data = {
            "description": description,
            "time": datetime.now(),
            "psd_scatter": (
                self.num_points,
                self.D_max,
                self._psd_D,
                self._S_table,
                self._Z_table,
                self._angular_table,
                self._m_table,
                self.geometries,
            ),
            "version": tmatrix_aux.VERSION,
        }
        with open(fn, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_scatter_table(self, fn: str):
        """Load a pickled lookup table saved by :meth:`save_scatter_table`."""
        with open(fn, "rb") as f:
            data = pickle.load(f)
        if "version" not in data or data["version"] != tmatrix_aux.VERSION:
            warnings.warn("Loading data saved with another version.", Warning)
        (
            self.num_points,
            self.D_max,
            self._psd_D,
            self._S_table,
            self._Z_table,
            self._angular_table,
            self._m_table,
            self.geometries,
        ) = data["psd_scatter"]
        return data["time"], data["description"]
