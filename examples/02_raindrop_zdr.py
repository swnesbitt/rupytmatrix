"""Tutorial 02 — a single oblate raindrop at C-band.

Physics question
----------------
A falling raindrop is squashed along the vertical — the larger the drop,
the more oblate. That shape asymmetry shows up on dual-polarisation radar
as *differential reflectivity* Z_dr (the ratio of horizontal to vertical
backscattered power, in dB). This script computes Z_dr for a single 2 mm
drop and shows how sensitive it is to the axis-ratio model.

What this script does
---------------------
1. Picks an equal-volume diameter D = 2 mm and chooses an axis ratio from
   the Thurai et al. (2007) drop-shape relationship.
2. Builds a ``Scatterer`` for the drop at C-band (λ ≈ 5.35 cm) using the
   tabulated 10 °C water refractive index.
3. Computes Z_h (linear reflectivity for a single drop, times N = 1),
   Z_dr (linear ratio), δ_hv (backscatter differential phase), and the
   linear depolarisation ratio.

pytmatrix analogue
------------------
Mirrors the single-drop demo in Leinonen (2014, §4): same formulas, same
axis-ratio convention (``axis_ratio = horizontal / vertical``, so oblate
drops have ``axis_ratio > 1``).
"""

from __future__ import annotations

import numpy as np

from rustmatrix import Scatterer
from rustmatrix import radar
from rustmatrix.tmatrix_aux import (
    dsr_thurai_2007,
    geom_horiz_back,
    K_w_sqr,
    wl_C,
)
from rustmatrix.refractive import m_w_10C


def main() -> None:
    D_mm = 2.0
    radius_mm = D_mm / 2.0

    # Thurai returns vertical/horizontal; Scatterer expects horizontal/vertical.
    axis_ratio = 1.0 / dsr_thurai_2007(D_mm)

    s = Scatterer(
        radius=radius_mm,
        wavelength=wl_C,
        m=m_w_10C[wl_C],
        axis_ratio=axis_ratio,
        Kw_sqr=K_w_sqr[wl_C],
        ddelt=1e-4,
        ndgs=2,
    )
    s.set_geometry(geom_horiz_back)

    Zh = radar.refl(s, h_pol=True)         # linear reflectivity @ N=1
    Zv = radar.refl(s, h_pol=False)
    Zdr = radar.Zdr(s)                     # linear Z_h / Z_v
    delta_hv = radar.delta_hv(s)           # radians

    print(f"D = {D_mm} mm, axis ratio (h/v) = {axis_ratio:.4f}  [Thurai 2007]")
    print(f"Wavelength = {wl_C} mm (C-band), m = {m_w_10C[wl_C]}")
    print()
    print(f"  Z_h (linear)   = {Zh:.4g} mm⁶·m⁻³ (for N = 1 drop/m³)")
    print(f"  Z_h (dBZ)      = {10 * np.log10(Zh):.2f}")
    print(f"  Z_dr (linear)  = {Zdr:.4f}")
    print(f"  Z_dr (dB)      = {10 * np.log10(Zdr):.3f}")
    print(f"  δ_hv           = {np.degrees(delta_hv):+.4f}°")
    print()
    print("Sensitivity to the drop-shape model:")
    for ratio_label, ar in (("sphere", 1.0),
                            ("Thurai 2007", axis_ratio),
                            ("more oblate (1.25)", 1.25)):
        s.axis_ratio = ar
        print(f"  {ratio_label:<20}  Z_dr = "
              f"{10 * np.log10(radar.Zdr(s)):+.3f} dB")


if __name__ == "__main__":
    main()
