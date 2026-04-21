"""Tutorial 12 — Spectral polarimetry of a rain / SLW / hail mixture at 500 hPa.

Reference
---------
Lakshmi K., K., Sahoo, S., Biswas, S. K., and Chandrasekar, V., 2024:
Study of Microphysical Signatures Based on Spectral Polarimetry during
the RELAMPAGO Field Experiment in Argentina. *J. Atmos. Oceanic
Technol.*, 41, 235–256, doi:10.1175/JTECH-D-22-0113.1.

Physics question
----------------
Lakshmi et al. (2024) used C-band Doppler spectral polarimetry from the
CSU-CHIVO radar to dissect mixed-phase and convective precipitation
volumes in RELAMPAGO. The central point of the paper is that
*spectral* Z_h, Z_dr, K_dp, ρ_hv separate hydrometeor populations that
the *bulk* observables mash into one number — because each species
occupies a distinct Doppler-velocity window set by its terminal fall
speed.

This tutorial builds a synthetic C-band radar resolution volume at
altitude where *P* = 500 hPa (*T* ≈ 252 K, *ρ* ≈ 0.69 kg/m³, reached
≈ 6 km MSL in a convective updraft). Three populations coexist:

* **Supercooled cloud liquid water** (SLW) — spherical droplets,
  D ≲ 0.2 mm, v_t ≲ 0.1 m/s.
* **Rain** — oblate drops following the Thurai 2007 shape, D ≲ 5 mm.
  At these diameters v_t is in the 5–10 m/s range after the low-air-
  density correction.
* **Small wet (melting) hail** — water-coated ice, 30 % meltwater by
  volume via Maxwell-Garnett EMA, axis ratio 0.75, canting σ = 40°,
  D up to 12 mm. At C-band the water coating drives a strong Mie
  resonance near D ≈ 8–10 mm with large |δ_hv| — exactly the
  rain-hail mixing regime Lakshmi et al. (2024) flag at the melting
  layer.

The reduced ambient density aloft is encoded via a ``(ρ₀/ρ)`` fall-speed
correction; each species gets a :mod:`rustmatrix.spectra.fall_speed`
callable evaluated at the local (T, P).

Geometry
--------
Polarimetric observables require a slant beam: oblate raindrops at
nadir project to circles, collapsing Z_dr and K_dp to zero. We use an
elevation angle φ = 30° — low enough that sZ_dr, sK_dp carry real
signal and high enough that each species' terminal velocity still maps
cleanly onto the beam's radial velocity:

* radial velocity  v_r(D) = v_t(D) · sin φ
* backscatter geometry  (θ₀, θ) = (60°, 120°)
* forward  geometry  (θ₀, θ) = (60°, 60°)

What this script does
---------------------
1. Builds the three C-band scatterers with their PSDs.
2. Runs a :class:`SpectralIntegrator` per species *and* one on the
   :class:`HydroMix` with per-component kinematics.
3. Prints the spectral tables at a handful of Doppler velocities for
   sZ_h, sZ_dr, sK_dp, sρ_hv.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import (HydroMix, MixtureComponent, Scatterer,
                        SpectralIntegrator, orientation, radar, spectra)
from rustmatrix.psd import ExponentialPSD, GammaPSD, PSDIntegrator
from rustmatrix.refractive import m_w_0C, mg_refractive, mi
from rustmatrix.tmatrix_aux import K_w_sqr, dsr_thurai_2007, wl_C


# --- ambient state at 500 hPa ------------------------------------------
P_HPA = 500.0
T_K = 252.0
R_D = 287.05
RHO_AIR = P_HPA * 100.0 / (R_D * T_K)     # kg/m³ ≈ 0.691
RHO_0 = 1.225
RHO_RATIO = RHO_0 / RHO_AIR               # ≈ 1.77
DENS_CORR_POW4 = RHO_RATIO ** 0.4         # Foote–duToit exponent for rain
DENS_CORR_SQRT = RHO_RATIO ** 0.5         # for hail / large particles

# --- 30° slant geometry -----------------------------------------------
PHI_DEG = 30.0
SIN_PHI = np.sin(np.deg2rad(PHI_DEG))
THETA0 = 90.0 - PHI_DEG                  # zenith angle of incidence
GEOM_BACK = (THETA0, 180.0 - THETA0, 0.0, 180.0, 0.0, 0.0)
GEOM_FORW = (THETA0, THETA0, 0.0, 0.0, 0.0, 0.0)

# --- velocity grid (positive = away from radar along slant beam) -----
V_MIN, V_MAX, N_BINS = -2.0, 15.0, 512

# --- small-wet-hail PSD / EMA knobs ------------------------------------
# Melting hail below / near the freezing level: a thin meltwater coat
# pushes the effective refractive index toward water. Modelled as a
# Maxwell-Garnett mixture with water as the matrix (30 %% water by
# volume), following the standard wet-ice EMA treatment used by
# Ryzhkov & Zrnić, Kumjian 2013 etc.
WATER_VOL_FRAC = 0.30
HAIL_N0 = 150.0
HAIL_LAMBDA = 0.6
HAIL_DMAX = 12.0
HAIL_AXIS = 0.75
HAIL_CANT_STD = 40.0


def wet_hail_m(wl: float) -> complex:
    # Matrix = water, inclusion = ice. Drives strong C-band resonance
    # and a large |δ_hv| at D ~ 8–12 mm.
    return mg_refractive((m_w_0C[wl], mi(wl, 0.917)),
                         (WATER_VOL_FRAC, 1.0 - WATER_VOL_FRAC))


def build_rain() -> Scatterer:
    s = Scatterer(wavelength=wl_C, m=m_w_0C[wl_C], Kw_sqr=K_w_sqr[wl_C],
                  ddelt=1e-4, ndgs=2)
    integ = PSDIntegrator()
    integ.D_max = 5.0
    integ.num_points = 64
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    integ.geometries = (GEOM_BACK, GEOM_FORW)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    # Heavy convective rain: D0 = 2.0 mm, Nw = 8e4 m⁻³ mm⁻¹ ⇒ Z_h ≈ 52 dBZ.
    s.psd = GammaPSD(D0=2.0, Nw=8e4, mu=2, D_max=5.0)
    return s


def build_slw() -> Scatterer:
    """Small spherical supercooled cloud droplets."""
    s = Scatterer(wavelength=wl_C, m=m_w_0C[wl_C], Kw_sqr=K_w_sqr[wl_C],
                  axis_ratio=1.0, ddelt=1e-4, ndgs=2)
    integ = PSDIntegrator()
    integ.D_max = 0.2
    integ.num_points = 64
    integ.geometries = (GEOM_BACK, GEOM_FORW)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    # Gamma PSD centred on D₀ ≈ 30 µm, LWC ≈ 0.5 g/m³ in a vigorous updraft.
    s.psd = GammaPSD(D0=0.03, Nw=1e11, mu=4, D_max=0.2)
    return s


def build_hail() -> Scatterer:
    """Small wet (melting) hail — water-coated ice, oblate, broad canting."""
    s = Scatterer(wavelength=wl_C, m=wet_hail_m(wl_C), Kw_sqr=K_w_sqr[wl_C],
                  axis_ratio=HAIL_AXIS, ddelt=1e-4, ndgs=2)
    s.orient = orientation.orient_averaged_fixed
    s.or_pdf = orientation.gaussian_pdf(std=HAIL_CANT_STD, mean=90.0)
    s.n_alpha = 6
    s.n_beta = 12
    integ = PSDIntegrator()
    integ.D_max = HAIL_DMAX
    integ.num_points = 64
    integ.geometries = (GEOM_BACK, GEOM_FORW)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    s.psd = ExponentialPSD(N0=HAIL_N0, Lambda=HAIL_LAMBDA, D_max=HAIL_DMAX)
    return s


# --- fall-speed callables (Doppler radial velocity along 30° slant) ---
def v_rain(D):
    # Beard 1976 at (T, P) handles the density correction internally.
    v_t = spectra.fall_speed.beard_1976(D, T=T_K, P=P_HPA * 100.0)
    return v_t * SIN_PHI


_hail_power = spectra.fall_speed.power_law(a=9.0, b=0.64, D_ref=10.0)


def v_hail(D):
    # Matson & Huggins-style v ≈ 9 (D/1cm)^0.64, density-corrected.
    return _hail_power(D) * DENS_CORR_SQRT * SIN_PHI


def v_slw(D):
    # Stokes-regime drag on cloud droplets at sea level scaled by (ρ₀/ρ)^0.4.
    return 3.0 * np.asarray(D, dtype=float) ** 2 * DENS_CORR_POW4 * SIN_PHI


def run_single(sc, fall, turb):
    return SpectralIntegrator(
        sc, fall_speed=fall, turbulence=turb,
        v_min=V_MIN, v_max=V_MAX, n_bins=N_BINS,
        geometry_backscatter=GEOM_BACK,
        geometry_forward=GEOM_FORW,
    ).run()


def main() -> None:
    rain = build_rain()
    slw = build_slw()
    hail = build_hail()

    print("Lakshmi et al. 2024 (JTECH) — spectral polarimetry at 500 hPa")
    print("-" * 72)
    print(f"  P = {P_HPA:.0f} hPa, T = {T_K:.1f} K, ρ_air = {RHO_AIR:.3f} kg/m³")
    print(f"  ρ₀/ρ = {RHO_RATIO:.3f}  ⇒  rain correction × {DENS_CORR_POW4:.3f},"
          f"  hail × {DENS_CORR_SQRT:.3f}")
    print(f"  elevation φ = {PHI_DEG:.0f}°  (sin φ = {SIN_PHI:.3f})")
    print()

    def bulk(sc):
        sc.set_geometry(GEOM_BACK)
        Z = 10 * np.log10(radar.refl(sc))
        Zdr = 10 * np.log10(radar.Zdr(sc))
        rho = radar.rho_hv(sc)
        sc.set_geometry(GEOM_FORW)
        Kdp = radar.Kdp(sc)
        return Z, Zdr, rho, Kdp

    print(f"  {'species':<8} {'Z_h':>8} {'Z_dr':>7} {'ρ_hv':>8} {'K_dp':>9}")
    print(f"  {'':<8} {'[dBZ]':>8} {'[dB]':>7} {'':>8} {'[°/km]':>9}")
    for name, sc in (("SLW", slw), ("rain", rain), ("hail", hail)):
        Z, Zdr, rho, Kdp = bulk(sc)
        print(f"  {name:<8} {Z:>8.2f} {Zdr:>+7.3f} {rho:>8.5f} {Kdp:>+9.4f}")
    print()

    # Moderate turbulence: σ_t = 0.5 m/s.
    turb = spectra.GaussianTurbulence(0.5)

    r_slw = run_single(slw, v_slw, turb)
    r_rain = run_single(rain, v_rain, turb)
    r_hail = run_single(hail, v_hail, turb)

    mix = HydroMix([
        MixtureComponent(slw, slw.psd, "slw"),
        MixtureComponent(rain, rain.psd, "rain"),
        MixtureComponent(hail, hail.psd, "hail"),
    ])
    r_mix = SpectralIntegrator(
        mix, component_kinematics={
            "slw":  (v_slw, turb),
            "rain": (v_rain, turb),
            "hail": (v_hail, turb),
        },
        v_min=V_MIN, v_max=V_MAX, n_bins=N_BINS,
        geometry_backscatter=GEOM_BACK,
        geometry_forward=GEOM_FORW,
    ).run()

    # Second scenario: hail concentration halved (N0 = 75) to expose
    # how each spectral observable responds to hail number density.
    hail_psd_half = ExponentialPSD(N0=HAIL_N0 / 2.0,
                                   Lambda=HAIL_LAMBDA, D_max=HAIL_DMAX)
    mix_half = HydroMix([
        MixtureComponent(slw, slw.psd, "slw"),
        MixtureComponent(rain, rain.psd, "rain"),
        MixtureComponent(hail, hail_psd_half, "hail"),
    ])
    r_mix_half = SpectralIntegrator(
        mix_half, component_kinematics={
            "slw":  (v_slw, turb),
            "rain": (v_rain, turb),
            "hail": (v_hail, turb),
        },
        v_min=V_MIN, v_max=V_MAX, n_bins=N_BINS,
        geometry_backscatter=GEOM_BACK,
        geometry_forward=GEOM_FORW,
    ).run()

    def dB(x):
        return 10 * np.log10(np.maximum(x, 1e-12))

    velocities = np.array([0.05, 1.5, 3.0, 5.0, 7.0, 9.0, 12.0])
    idx = [int(np.argmin(np.abs(r_mix.v - v))) for v in velocities]

    def row(name, res, key):
        vals = []
        for i in idx:
            if key == "sZ":
                vals.append(f"{dB(res.sZ_h[i]):>+7.2f}")
            elif key == "sZdr":
                vals.append(f"{dB(res.sZ_dr[i]):>+7.2f}")
            elif key == "sKdp":
                vals.append(f"{res.sKdp[i]:>+8.3f}")
            elif key == "srho":
                vals.append(f"{res.srho_hv[i]:>8.5f}")
        print(f"  {name:<10} " + " ".join(vals))

    for label, key in (("sZ_h [dBZ]", "sZ"),
                       ("sZ_dr [dB]", "sZdr"),
                       ("sK_dp [°/km]", "sKdp"),
                       ("sρ_hv", "srho")):
        print(f"{label}  at v [m/s] = {'  '.join(f'{v:>5.2f}' for v in velocities)}")
        row("SLW", r_slw, key)
        row("rain", r_rain, key)
        row("hail", r_hail, key)
        row("mixture", r_mix, key)
        row("mix ½hail", r_mix_half, key)
        print()


if __name__ == "__main__":
    main()
