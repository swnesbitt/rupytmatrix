# Tutorials

Fourteen numbered, executable notebooks. Each one runs on every docs
build (cached — unchanged notebooks aren't re-run), so the figures and
printed output you see here are the actual output of the current
`rustmatrix`. Every notebook ships with a twin `.py` script under
[`examples/`](https://github.com/swnesbitt/rustmatrix/tree/main/examples)
if you'd rather run them locally.

| # | Notebook | What it covers |
|---|---|---|
| 01 | [Sphere / Mie parity](01_sphere_mie) | Sanity check — T-matrix at spherical shape reduces to Mie. |
| 02 | [Raindrop $Z_\mathrm{dr}$](02_raindrop_zdr) | Single 2 mm oblate raindrop at C-band. |
| 03 | [Gamma-PSD rain](03_psd_gamma_rain) | Tabulated $S$, $Z$ → $Z_h$, $Z_\mathrm{dr}$, $K_\mathrm{dp}$, $A_i$ vs rain rate. |
| 04 | [Oriented ice](04_oriented_ice) | Columnar ice at W-band with a Gaussian canting PDF. |
| 05 | [Radar band sweep](05_radar_band_sweep) | Same particle across S/C/X/Ku/Ka/W. |
| 06 | [HydroMix](06_hd_mix) | Rain + oriented ice as one combined scatterer. |
| 07 | [Doppler spectrum, rain](07_doppler_spectrum_rain) | Reproduces Kollias et al. 2002 W-band Mie minimum. |
| 08 | [Spectral polarimetry, rain + ice](08_spectral_polarimetry_rain_ice) | Billault-Roux et al. 2023 dual-frequency snow signature. |
| 09 | [Zhu 2023 particle inertia](09_zhu_2023_particle_inertia) | Diameter-dependent turbulence broadening. |
| 10 | [SLW vs snow](10_slw_vs_snow) | Bimodal W-band spectrum from a HydroMix. |
| 11 | [Honeyager 2013 classes](11_honeyager_hydrometeor_classes) | Rain / aggregate / graupel / dense-ice σ_b and DWR. |
| 12 | [Rain + SLW + hail (Lakshmi 2024)](12_spectral_polarimetry_rain_slw_hail) | C-band spectral polarimetry at 500 hPa. |
| 13 | [Wind × turbulence sensitivity](13_wind_turbulence_sensitivity) | Closed-form σ_beam validation sweep. |
| 14 | [Beam pattern × scene](14_beam_pattern_scene) | Gaussian / Airy patterns over convective cells; interactive cell-spacing slider. |

The notebooks build on each other loosely — later tutorials assume
you've seen earlier ones — but any of them run standalone once
`rustmatrix` is installed.

```{toctree}
:hidden:
:maxdepth: 1

01_sphere_mie
02_raindrop_zdr
03_psd_gamma_rain
04_oriented_ice
05_radar_band_sweep
06_hd_mix
07_doppler_spectrum_rain
08_spectral_polarimetry_rain_ice
09_zhu_2023_particle_inertia
10_slw_vs_snow
11_honeyager_hydrometeor_classes
12_spectral_polarimetry_rain_slw_hail
13_wind_turbulence_sensitivity
14_beam_pattern_scene
```
