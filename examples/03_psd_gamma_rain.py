"""Tutorial 03 — a gamma-distributed rain PSD at C-band.

Physics question
----------------
Real radar volumes contain many drops following some particle-size
distribution (PSD). The observed Z_h, Z_dr, K_dp, and specific
attenuation A_i are the PSD-weighted integrals of the single-drop
quantities across the drop spectrum. A common parameterisation is the
normalised gamma PSD of Bringi & Chandrasekar (2001).

What this script does
---------------------
1. Tabulates ``S(D)`` and ``Z(D)`` once across 0.1–8 mm using a
   ``PSDIntegrator`` (the parallel Rust tabulator — see the
   *Performance* section of the README).
2. Evaluates the PSD-integrated Z_h, Z_dr, K_dp, and A_i for a few
   representative normalised-gamma PSDs that roughly correspond to
   stratiform (D0=1 mm), convective (D0=2 mm), and heavy-convective
   (D0=3 mm) rain.

pytmatrix analogue
------------------
Same ``PSDIntegrator.init_scatter_table`` + ``GammaPSD`` pattern as
pytmatrix; the only operational difference is that rustmatrix runs the
per-diameter solves in parallel with the GIL released.
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
)
from rustmatrix.refractive import m_w_10C


def build_scatterer() -> Scatterer:
    """Scatterer configured for C-band rain drops."""
    s = Scatterer(
        wavelength=wl_C,
        m=m_w_10C[wl_C],
        Kw_sqr=K_w_sqr[wl_C],
        ddelt=1e-4,
        ndgs=2,
    )
    integ = rs_psd.PSDIntegrator()
    integ.D_max = 8.0
    integ.num_points = 64
    # Vary axis ratio with diameter via Thurai 2007 (h/v = 1/dsr).
    integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
    # Tabulate both backscatter and forward geometries: we need forward
    # scattering for K_dp and A_i.
    integ.geometries = (geom_horiz_back, geom_horiz_forw)
    s.psd_integrator = integ
    s.psd_integrator.init_scatter_table(s)
    return s


def observables(s: Scatterer, psd) -> dict:
    s.psd = psd
    s.set_geometry(geom_horiz_back)
    Zh = radar.refl(s, h_pol=True)
    Zdr = radar.Zdr(s)
    s.set_geometry(geom_horiz_forw)
    Kdp = radar.Kdp(s)
    Ai = radar.Ai(s, h_pol=True)
    return {"Zh_dBZ": 10 * np.log10(Zh),
            "Zdr_dB": 10 * np.log10(Zdr),
            "Kdp_deg_per_km": Kdp,
            "Ai_dB_per_km": Ai}


def main() -> None:
    s = build_scatterer()

    psds = {
        "stratiform (D0=1, Nw=8e3, mu=4)":
            rs_psd.GammaPSD(D0=1.0, Nw=8e3, mu=4),
        "convective  (D0=2, Nw=8e3, mu=4)":
            rs_psd.GammaPSD(D0=2.0, Nw=8e3, mu=4),
        "heavy conv. (D0=3, Nw=8e3, mu=4)":
            rs_psd.GammaPSD(D0=3.0, Nw=8e3, mu=4),
    }

    print(f"{'PSD':<34} {'Z_h':>10} {'Z_dr':>8} {'K_dp':>10} {'A_i':>10}")
    print(f"{'':<34} {'[dBZ]':>10} {'[dB]':>8} {'[°/km]':>10} {'[dB/km]':>10}")
    print("-" * 76)
    for name, p in psds.items():
        obs = observables(s, p)
        print(f"{name:<34} "
              f"{obs['Zh_dBZ']:>10.2f} {obs['Zdr_dB']:>8.3f} "
              f"{obs['Kdp_deg_per_km']:>10.3f} {obs['Ai_dB_per_km']:>10.4f}")


if __name__ == "__main__":
    main()
