"""Tutorial 06 — a hydrometeor mixture at C-band (rain + snow + graupel).

Physics question
----------------
Real radar volumes often contain more than one species. The mixture's
combined polarimetric signature is the incoherent sum of the per-species
amplitude (S) and phase (Z) matrices, but the *non*-linear observables
(Z_dr, ρ_hv, δ_hv) cannot be averaged from per-species values — they
must be recomputed from the summed matrices.

The mixture ρ_hv drops below unity whenever two populations contribute
comparable power *and* have different polarimetric fingerprints:

* rain has Z_dr > 0 (oblate drops),
* low-density snow aggregates have Z_dr near 0 with wide canting,
* graupel is near-spherical with broad orientation and non-Rayleigh
  resonance at C-band for D > ~6 mm.

What this script does
---------------------
1. Builds a C-band rain scatterer (Thurai 2007 drop shape, 10 °C water).
2. Builds a low-density snow aggregate (ρ = 0.2 g/cm³, axis ratio 0.6,
   Gaussian canting σ = 40° to mimic tumbling aggregates — Bringi &
   Chandrasekar 2001, §4.1; Garrett et al. 2015).
3. Builds a graupel scatterer (ρ = 0.4 g/cm³, axis ratio 0.8, canting
   σ = 40°, Dmax = 8 mm) following the parameterisation used by
   Ryzhkov, Zrnić, Burgess 2005 *JAMC*) and Kumjian 2013 *JOM*).
4. Reports Z_h / Z_dr / ρ_hv / δ_hv / K_dp / A_i for each species alone
   and for three mixtures:
     * rain + light snow (original baseline),
     * rain + heavy snow (snow N0 boosted to match rain Z_h),
     * rain + graupel.
   The heavier-snow and graupel mixtures drive the mixture ρ_hv visibly
   below unity via the mechanisms listed above.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import (HydroMix, MixtureComponent, Scatterer, orientation,
                        radar)
from rustmatrix import psd as rs_psd
from rustmatrix.refractive import m_w_10C, mi
from rustmatrix.tmatrix_aux import (
    K_w_sqr,
    dsr_thurai_2007,
    geom_horiz_back,
    geom_horiz_forw,
    wl_C,
)


def build_rain() -> Scatterer:
    s = Scatterer(
        wavelength=wl_C,
        m=m_w_10C[wl_C],
        Kw_sqr=K_w_sqr[wl_C],
        ddelt=1e-4,
        ndgs=2,
    )
    integ = rs_psd.PSDIntegrator()
    integ.D_max = 6.0
    integ.num_points = 64
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    integ.geometries = (geom_horiz_back, geom_horiz_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    return s


def build_snow() -> Scatterer:
    """Low-density aggregate with wide Gaussian canting (σ = 40°)."""
    s = Scatterer(
        wavelength=wl_C,
        m=mi(wl_C, 0.2),
        Kw_sqr=K_w_sqr[wl_C],
        axis_ratio=0.6,
        ddelt=1e-4,
        ndgs=2,
    )
    s.orient = orientation.orient_averaged_fixed
    s.or_pdf = orientation.gaussian_pdf(std=40.0, mean=90.0)
    s.n_alpha = 6
    s.n_beta = 12
    integ = rs_psd.PSDIntegrator()
    integ.D_max = 8.0
    integ.num_points = 64
    integ.geometries = (geom_horiz_back, geom_horiz_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    return s


def build_graupel() -> Scatterer:
    """Near-spherical graupel (ρ = 0.4 g/cm³, axis ratio 0.8)
    with wide tumbling σ = 40° — after Ryzhkov et al. 2005, Kumjian 2013."""
    s = Scatterer(
        wavelength=wl_C,
        m=mi(wl_C, 0.4),
        Kw_sqr=K_w_sqr[wl_C],
        axis_ratio=0.8,
        ddelt=1e-4,
        ndgs=2,
    )
    s.orient = orientation.orient_averaged_fixed
    s.or_pdf = orientation.gaussian_pdf(std=40.0, mean=90.0)
    s.n_alpha = 6
    s.n_beta = 12
    integ = rs_psd.PSDIntegrator()
    integ.D_max = 8.0
    integ.num_points = 64
    integ.geometries = (geom_horiz_back, geom_horiz_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    return s


def observables(x) -> dict:
    x.set_geometry(geom_horiz_back)
    Zh = radar.refl(x, h_pol=True)
    Zdr = radar.Zdr(x)
    rho = radar.rho_hv(x)
    delta = np.degrees(radar.delta_hv(x))
    x.set_geometry(geom_horiz_forw)
    Kdp = radar.Kdp(x)
    Ai = radar.Ai(x, h_pol=True)
    return dict(Zh=10 * np.log10(Zh), Zdr=10 * np.log10(Zdr), rho=rho,
                delta=delta, Kdp=Kdp, Ai=Ai)


def print_row(name: str, x) -> None:
    o = observables(x)
    print(f"{name:<24} {o['Zh']:>7.2f} {o['Zdr']:>+7.3f} "
          f"{o['rho']:>8.5f} {o['delta']:>+8.4f} "
          f"{o['Kdp']:>+9.4f} {o['Ai']:>9.5f}")


def main() -> None:
    rain = build_rain()
    snow = build_snow()
    graupel = build_graupel()

    rain_psd = rs_psd.GammaPSD(D0=1.5, Nw=8e3, mu=4, D_max=6.0)
    snow_psd_light = rs_psd.ExponentialPSD(N0=5e3, Lambda=2.0, D_max=8.0)
    # Heavy-snow PSD: same slope, boost N0 so snow Z_h matches rain Z_h.
    snow_psd_heavy = rs_psd.ExponentialPSD(N0=1.5e5, Lambda=2.0, D_max=8.0)
    # Graupel PSD: exponential, fewer particles but broader to D_max = 8 mm
    # so the largest graupel sit in the C-band non-Rayleigh regime.
    graupel_psd = rs_psd.ExponentialPSD(N0=4e3, Lambda=1.4, D_max=8.0)

    rain.psd = rain_psd

    header = (f"{'case':<24} {'Z_h':>7} {'Z_dr':>7} "
              f"{'ρ_hv':>8} {'δ_hv':>8} {'K_dp':>9} {'A_i':>9}")
    units  = (f"{'':<24} {'[dBZ]':>7} {'[dB]':>7} "
              f"{'':>8} {'[°]':>8} {'[°/km]':>9} {'[dB/km]':>9}")
    print(header)
    print(units)
    print("-" * len(header))

    mix_rain_lightsnow = HydroMix([
        MixtureComponent(rain, rain_psd, "rain"),
        MixtureComponent(snow, snow_psd_light, "snow"),
    ])
    mix_rain_heavysnow = HydroMix([
        MixtureComponent(rain, rain_psd, "rain"),
        MixtureComponent(snow, snow_psd_heavy, "snow"),
    ])
    mix_rain_graupel = HydroMix([
        MixtureComponent(rain, rain_psd, "rain"),
        MixtureComponent(graupel, graupel_psd, "graupel"),
    ])

    # Evaluate per-species cases eagerly — snow is shared between the
    # light- and heavy-snow paths by mutation, so we must read out the
    # observables immediately after setting each .psd.
    print_row("rain only", rain)
    snow.psd = snow_psd_light
    print_row("light snow only", snow)
    snow.psd = snow_psd_heavy
    print_row("heavy snow only", snow)
    graupel.psd = graupel_psd
    print_row("graupel only", graupel)

    print_row("rain + light snow", mix_rain_lightsnow)
    print_row("rain + heavy snow", mix_rain_heavysnow)
    print_row("rain + graupel",    mix_rain_graupel)

    # Linear-Z_h additivity sanity check on the heavy-snow mixture.
    rain.set_geometry(geom_horiz_back)
    snow.psd = snow_psd_heavy
    snow.set_geometry(geom_horiz_back)
    mix_rain_heavysnow.set_geometry(geom_horiz_back)
    zh_sum = radar.refl(rain) + radar.refl(snow)
    zh_mix = radar.refl(mix_rain_heavysnow)
    print(f"\nlinear Z_h additivity (heavy snow mixture): "
          f"rain + snow = {zh_sum:.3e}, mix = {zh_mix:.3e}, "
          f"rel. diff = {abs(zh_sum - zh_mix) / zh_mix:.2e}")


if __name__ == "__main__":
    main()
