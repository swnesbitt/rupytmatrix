"""Tutorial 02 — Single-drop polarimetric response at S/C/X bands (D = 0–8 mm).

Physics question
----------------
A falling raindrop is flattened along the vertical. The flattening grows
with diameter (Thurai et al. 2007), which leaves distinctive fingerprints
on polarimetric radar observables:

* **Z_h** rises as D⁶ (Rayleigh) then departs from that trend once the
  drop walks out of Rayleigh — earlier at shorter wavelengths.
* **Z_dr** rises with diameter because oblateness grows with D.
* **K_dp** — the forward-propagation differential phase — scales like
  Re(f_h(0) − f_v(0)) per drop; it also grows with D and with
  oblateness, and it *decreases* with wavelength (hence longer-λ
  radars need more drops to see a given K_dp).
* **LDR** — the linear depolarisation ratio — is sub-dominant in rain
  and is set by drop canting. Here we model a modest σ = 5° Gaussian
  canting distribution to produce realistic rain LDR levels of about
  −30 to −20 dB.

What this script does
---------------------
For each of S (λ ≈ 11.1 cm), C (5.35 cm), and X (3.33 cm) bands, at
10 °C water refractive index, it sweeps drop equivalent diameter
D = 0.1–8 mm using the Thurai 2007 shape model and computes:

1. Z_h [dBZ per drop/m³]
2. Z_dr [dB]
3. K_dp [°/km per drop/m³] — so multiplying by drop concentration N
   [m⁻³] gives the usual K_dp.
4. LDR [dB] with σ = 5° canting.

Non-Rayleigh roll-off at X-band (large drops) and the strong wavelength
dependence of K_dp are the headline takeaways.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import Scatterer, orientation, radar, scatter
from rustmatrix.refractive import m_w_10C
from rustmatrix.tmatrix_aux import (K_w_sqr, dsr_thurai_2007,
                                    geom_horiz_back, geom_horiz_forw,
                                    wl_C, wl_S, wl_X)


BANDS = [("S", wl_S), ("C", wl_C), ("X", wl_X)]
D_GRID = np.linspace(0.1, 8.0, 40)
CANTING_STD_DEG = 5.0


def build_drop(D_mm: float, wl: float) -> Scatterer:
    s = Scatterer(radius=D_mm / 2.0, wavelength=wl, m=m_w_10C[wl],
                  axis_ratio=1.0 / dsr_thurai_2007(D_mm),
                  Kw_sqr=K_w_sqr[wl], ddelt=1e-4, ndgs=2)
    # Flat-lying oblate drop: symmetry axis vertical (β ≈ 0) with small
    # wobble. mean=0° keeps the drop horizontally oriented on average;
    # std=5° represents turbulent wobble that creates a finite LDR.
    s.orient = orientation.orient_averaged_fixed
    s.or_pdf = orientation.gaussian_pdf(std=CANTING_STD_DEG, mean=0.0)
    s.n_alpha = 6
    s.n_beta = 12
    return s


def sweep_band(wl: float) -> dict:
    Zh = np.empty_like(D_GRID)
    Zdr = np.empty_like(D_GRID)
    Kdp = np.empty_like(D_GRID)
    LDR = np.empty_like(D_GRID)
    for i, D in enumerate(D_GRID):
        s = build_drop(D, wl)
        s.set_geometry(geom_horiz_back)
        Zh[i] = 10 * np.log10(max(radar.refl(s, h_pol=True), 1e-30))
        Zdr[i] = 10 * np.log10(max(radar.Zdr(s), 1e-30))
        LDR[i] = 10 * np.log10(max(scatter.ldr(s, h_pol=True), 1e-30))
        s.set_geometry(geom_horiz_forw)
        Kdp[i] = radar.Kdp(s)
    return dict(Zh=Zh, Zdr=Zdr, Kdp=Kdp, LDR=LDR)


def main() -> None:
    print(f"Single-drop polarimetric response (Thurai 2007 shape, "
          f"10 °C water, canting σ = {CANTING_STD_DEG:.0f}°)")
    print(f"N_drop = 1 m⁻³ convention: Z_h [dBZ], K_dp [°/km] both per drop/m³.")
    print()

    data = {name: sweep_band(wl) for name, wl in BANDS}

    # Print a short table at representative diameters.
    rows = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0)
    idx = [int(np.argmin(np.abs(D_GRID - D))) for D in rows]
    for obs_name, fmt in (("Z_h [dBZ]", "{:+7.2f}"),
                          ("Z_dr [dB]", "{:+7.3f}"),
                          ("K_dp [°/km]", "{:+7.3e}"),
                          ("LDR [dB]", "{:+7.2f}")):
        print(f"{obs_name:<14} " + "  ".join(f"D={D:.1f}" for D in rows))
        key = obs_name.split()[0].replace("_", "").replace("LDR", "LDR")
        key = {"Zh": "Zh", "Zdr": "Zdr", "Kdp": "Kdp", "LDR": "LDR"}[key]
        for band_name, _ in BANDS:
            row = data[band_name][key][idx]
            cells = "  ".join(fmt.format(v) for v in row)
            print(f"  {band_name}-band       {cells}")
        print()


if __name__ == "__main__":
    main()
