"""Tutorial 04 — an oriented columnar ice crystal at W-band.

Physics question
----------------
Pristine ice crystals (columns, plates, dendrites) fall with a preferred
orientation that wobbles around the horizontal by some canting angle. The
*orientation PDF* captures that wobble, and dual-pol observables depend
strongly on it. This script shows how to feed an orientation PDF into a
``Scatterer`` and compares the two orientation-averaging schemes
implemented in rustmatrix.

What this script does
---------------------
1. Builds a prolate-column ice crystal (``axis_ratio = 0.5``) at W-band
   using the bundled Warren ice-refractive-index interpolator.
2. Sets a Gaussian orientation PDF (mean β = 90°, σ = 20°): crystals
   fall nearly horizontally but wobble.
3. Computes Z_dr with three schemes:
   * ``orient_single`` — no averaging; just α = β = 0.
   * ``orient_averaged_fixed`` — Gaussian quadrature in β (fast,
     accurate to ~5e-3 with a few dozen points).
   * ``orient_averaged_adaptive`` — scipy.dblquad (slow, high accuracy).

pytmatrix analogue
------------------
Same ``orient`` / ``or_pdf`` attributes as pytmatrix; the adaptive path
calls the Rust core through PyO3 rather than Fortran through f2py but
the rules and the results match.
"""

from __future__ import annotations

import time

import numpy as np

from rustmatrix import Scatterer
from rustmatrix import orientation as rs_orient
from rustmatrix import radar
from rustmatrix.tmatrix_aux import geom_horiz_back, wl_W
from rustmatrix.refractive import mi


def main() -> None:
    # ~1 mm prolate ice column, axis ratio 0.5 (length twice the width).
    D_eq_mm = 0.5
    ice_m = mi(wl_W, 0.9)                    # nearly-solid ice at W-band
    print(f"λ = {wl_W} mm (W-band), m_ice(0.9 g/cm³) = {ice_m}")

    base_kwargs = dict(
        radius=D_eq_mm / 2.0,
        wavelength=wl_W,
        m=ice_m,
        axis_ratio=0.5,                      # prolate (vertical > horizontal)
        ddelt=1e-4,
        ndgs=2,
    )

    # Reusable Gaussian canting PDF.
    pdf = rs_orient.gaussian_pdf(std=20.0, mean=90.0)

    schemes = [
        ("orient_single",
         rs_orient.orient_single,
         {"or_pdf": pdf}),
        ("orient_averaged_fixed (n_alpha=6, n_beta=12)",
         rs_orient.orient_averaged_fixed,
         {"or_pdf": pdf, "n_alpha": 6, "n_beta": 12}),
        ("orient_averaged_adaptive",
         rs_orient.orient_averaged_adaptive,
         {"or_pdf": pdf}),
    ]

    print(f"{'scheme':<50} {'Z_dr [dB]':>12} {'time [s]':>10}")
    print("-" * 74)
    for label, orient, extra in schemes:
        s = Scatterer(**base_kwargs, **extra)
        s.orient = orient
        s.set_geometry(geom_horiz_back)
        t0 = time.perf_counter()
        zdr = radar.Zdr(s)
        elapsed = time.perf_counter() - t0
        print(f"{label:<50} {10 * np.log10(zdr):>+12.4f} {elapsed:>10.3f}")

    print()
    print("Takeaway: the 'fixed' scheme is ~100× faster than 'adaptive'")
    print("and matches it to within a few hundredths of a dB in Z_dr for")
    print("smooth PDFs — pick 'fixed' for PSD-integrated work.")


if __name__ == "__main__":
    main()
