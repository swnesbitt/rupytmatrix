"""Tutorial 09 — Reproducing Zhu, Kollias, Yang (2023) particle-inertia spectra.

Reference
---------
Zhu, Z., Kollias, P., and Yang, F., 2023: Particle Inertia Effects on Radar
Doppler Spectra Simulation, *Atmos. Meas. Tech. Discuss.*
MATLAB simulator source: https://zenodo.org/records/7897981

Physics question
----------------
The conventional approach convolves the drop-size reflectivity spectrum
with a single Gaussian of width σ_air (ambient turbulence intensity). But
small drops are passive tracers while large drops are ballistic — they
under-respond to the small-scale eddies of the turbulent wind field. The
upshot is a *diameter-dependent* broadening σ_t(D) that shrinks toward
the large-drop end of the spectrum.

Zhu 2023 demonstrates this by integrating a drag ODE for each drop in a
stochastic wind field and histogramming the resulting Doppler velocities.
rustmatrix's :class:`InertialZeng2023` implements the same physics via the
analytical Stokes-number low-pass response,

    σ_t(D) = σ_air / √(1 + St(D)²),      St = τ_p(D) / τ_eddy

so a side-by-side comparison of conventional vs. inertia-aware spectra at
Zhu 2023's configuration shows how the correction narrows the spectrum on
the fast-falling side.

What this script does
---------------------
1. Builds a W-band rain scatterer with the paper's exponential PSD
   (N₀ = 0.08 cm⁻⁴, Λ = 20 cm⁻¹, D ∈ [0.1, 4] mm).
2. Runs two `SpectralIntegrator` configurations at the paper's setup
   (Nyquist ±12 m/s, 1024 bins, σ_air = 0.29 m/s).
3. Compares conventional Gaussian broadening to the inertia-aware
   ``InertialZeng2023`` kernel.
4. Adds receiver noise at SNR = 40 dB over 20 dBZ reference (paper's
   configuration) and reports the spectrum in dBZ.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import Scatterer, SpectralIntegrator, spectra
from rustmatrix.psd import ExponentialPSD, PSDIntegrator
from rustmatrix.refractive import m_w_10C
from rustmatrix.tmatrix_aux import K_w_sqr, geom_vert_back, wl_W


# --- Paper configuration ---
V_MIN, V_MAX = -12.0, 12.0       # Nyquist range (m/s)
NFFT = 1024                      # FFT size
SNR_DB = 40.0                    # signal-to-noise ratio (dB)
Z_DBZ = 20.0                     # reference reflectivity (dBZ)
SIGMA_AIR = 0.289                # std of uniform [-0.5, 0.5] wind field
EPSILON = 1e-2                   # turbulent dissipation rate (m²/s³)
L_O = 0.1                        # integral length scale (m); small L_o ⇒
                                 # short τ_eddy ⇒ inertial correction is
                                 # appreciable for large raindrops


def build_rain_W() -> Scatterer:
    """W-band rain scatterer with Zhu 2023 exponential PSD.

    Zhu 2023 uses N(D) = N₀ · exp(shape · D_cm) · 1e4  [m⁻³ µm⁻¹],
    with N₀ = 0.08 and shape = -20. Converting to rustmatrix's
    mm-based units: N(D_mm) = 8e5 · exp(-2 · D_mm)  [m⁻³ mm⁻¹].
    """
    s = Scatterer(
        wavelength=wl_W,
        m=m_w_10C[wl_W],
        Kw_sqr=K_w_sqr[wl_W],
        ddelt=1e-4, ndgs=2,
    )
    integ = PSDIntegrator()
    integ.D_max = 4.0
    integ.num_points = 128
    integ.geometries = (geom_vert_back,)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=8e5, Lambda=2.0, D_max=4.0)
    return s


def noise_linear_from_SNR(Z_dBZ: float, SNR_dB: float) -> float:
    """Total linear noise power [mm⁶ m⁻³] for a given reference Z and SNR."""
    return 10 ** (Z_dBZ / 10.0) / 10 ** (SNR_dB / 10.0)


def main() -> None:
    rain = build_rain_W()
    fall = spectra.fall_speed.atlas_srivastava_sekhon_1973
    noise = noise_linear_from_SNR(Z_DBZ, SNR_DB)

    common = dict(
        fall_speed=fall,
        v_min=V_MIN, v_max=V_MAX, n_bins=NFFT,
        geometry_backscatter=geom_vert_back,
        noise=noise,
    )

    conventional = SpectralIntegrator(
        rain,
        turbulence=spectra.GaussianTurbulence(SIGMA_AIR),
        **common,
    ).run()

    inertial = SpectralIntegrator(
        rain,
        turbulence=spectra.InertialZeng2023(
            sigma_air=SIGMA_AIR, epsilon=EPSILON, L_o=L_O,
        ),
        **common,
    ).run()

    # Per-diameter inertial reduction factor σ_t(D) / σ_air
    zeng = spectra.InertialZeng2023(
        sigma_air=SIGMA_AIR, epsilon=EPSILON, L_o=L_O,
    )
    D_demo = np.array([0.2, 0.5, 1.0, 2.0, 3.0, 4.0])
    sigma_rel = zeng(D_demo) / SIGMA_AIR

    print("Zhu, Kollias, Yang (2023) — conventional vs inertia-aware broadening")
    print("-" * 72)
    print(f"  W-band, Nyquist ±{V_MAX:.0f} m/s, NFFT = {NFFT}")
    print(f"  σ_air = {SIGMA_AIR} m/s, ε = {EPSILON} m²/s³, L_o = {L_O} m")
    print(f"  SNR = {SNR_DB} dB over {Z_DBZ} dBZ reference")
    print(f"  total noise power = {noise:.4f} mm⁶/m³")
    print()
    print("  Inertial reduction factor σ_t(D) / σ_air:")
    for D, s in zip(D_demo, sigma_rel):
        print(f"    D = {D:.1f} mm   σ_t/σ_air = {s:.3f}")
    print()
    print("  Large drops respond less to small-scale eddies, so their")
    print("  contribution to the spectrum is narrower than the conventional")
    print("  single-Gaussian model predicts — Zhu 2023's central finding.")
    print()
    print(f"  ∫sZ_h dv (conventional) = {np.trapezoid(conventional.sZ_h, conventional.v):.3f} mm⁶/m³")
    print(f"  ∫sZ_h dv (inertia-aware) = {np.trapezoid(inertial.sZ_h, inertial.v):.3f} mm⁶/m³")


if __name__ == "__main__":
    main()
