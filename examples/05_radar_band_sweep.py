"""Tutorial 05 — the same rain PSD across six radar bands.

Physics question
----------------
Modern observational networks run dual-pol radars at S, C, X, Ku, Ka, and
W band. The same drop-size distribution produces very different Z_dr and
K_dp signatures at each wavelength because (a) the complex refractive
index of water is frequency-dependent, and (b) the size parameter x = πD/λ
changes which drops sit in the Rayleigh regime vs. the Mie/resonance
regime.

What this script does
---------------------
1. Loops over the six standard radar bands (``wl_S``…``wl_W``).
2. For each band, rebuilds the scatter table for a fixed PSD using the
   tabulated 20 °C water index and Thurai (2007) drop shapes.
3. Prints Z_h (dBZ), Z_dr (dB), K_dp (°/km), and A_i (dB/km) side by side.

This is the natural jumping-off point for multi-frequency retrieval
studies: swap in your own PSD and refractive-index model and you have a
forward operator across the whole instrument suite.

pytmatrix analogue
------------------
Identical to what you'd write with ``pytmatrix.tmatrix_aux.wl_*``; the
only change is the ``from rustmatrix ...`` imports.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import Scatterer
from rustmatrix import radar, psd as rs_psd
from rustmatrix.tmatrix_aux import (
    K_w_sqr,
    dsr_thurai_2007,
    geom_horiz_back,
    geom_horiz_forw,
    wl_C,
    wl_Ka,
    wl_Ku,
    wl_S,
    wl_W,
    wl_X,
)
from rustmatrix.refractive import m_w_20C


BANDS = [("S", wl_S), ("C", wl_C), ("X", wl_X),
         ("Ku", wl_Ku), ("Ka", wl_Ka), ("W", wl_W)]


def run_band(wavelength_mm: float) -> dict:
    s = Scatterer(
        wavelength=wavelength_mm,
        m=m_w_20C[wavelength_mm],
        Kw_sqr=K_w_sqr[wavelength_mm],
        ddelt=1e-4,
        ndgs=2,
    )
    integ = rs_psd.PSDIntegrator()
    integ.D_max = 8.0
    integ.num_points = 64
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    integ.geometries = (geom_horiz_back, geom_horiz_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)

    # Moderate convective-ish rain: D0 = 1.5 mm, Nw = 8000, mu = 4.
    s.psd = rs_psd.GammaPSD(D0=1.5, Nw=8e3, mu=4)

    s.set_geometry(geom_horiz_back)
    Zh_dBZ = 10 * np.log10(radar.refl(s, h_pol=True))
    Zdr_dB = 10 * np.log10(radar.Zdr(s))

    s.set_geometry(geom_horiz_forw)
    Kdp = radar.Kdp(s)
    Ai = radar.Ai(s, h_pol=True)

    return {"Zh_dBZ": Zh_dBZ, "Zdr_dB": Zdr_dB,
            "Kdp_deg_per_km": Kdp, "Ai_dB_per_km": Ai}


def main() -> None:
    print("Same gamma PSD (D0=1.5 mm, Nw=8e3, mu=4) at six radar bands:")
    print()
    print(f"{'band':<4} {'λ [mm]':>8} {'Z_h':>9} {'Z_dr':>8} "
          f"{'K_dp':>10} {'A_i':>10}")
    print(f"{'':<4} {'':>8} {'[dBZ]':>9} {'[dB]':>8} "
          f"{'[°/km]':>10} {'[dB/km]':>10}")
    print("-" * 57)
    for name, wl in BANDS:
        obs = run_band(wl)
        print(f"{name:<4} {wl:>8.2f} "
              f"{obs['Zh_dBZ']:>9.2f} {obs['Zdr_dB']:>8.3f} "
              f"{obs['Kdp_deg_per_km']:>10.3f} {obs['Ai_dB_per_km']:>10.4f}")


if __name__ == "__main__":
    main()
