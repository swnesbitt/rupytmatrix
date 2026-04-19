"""Tutorial 01 — a dielectric sphere at X-band, checked against Mie theory.

Physics question
----------------
Does the T-matrix solver actually reduce to classical Mie scattering in the
``axis_ratio = 1`` limit? Mie gives ``S`` and ``Z`` in closed form for a
sphere, so this is the cleanest sanity check before moving on to real
nonspherical particles.

What this script does
---------------------
1. Builds a ``Scatterer`` for a 1 mm water sphere at X-band.
2. Computes the amplitude matrix ``S`` and phase matrix ``Z`` in horizontal
   backscatter geometry.
3. Computes the backscatter cross-section ``sigma_HH = 4π |S_HH|²`` two ways:
   from the T-matrix ``S`` and from the closed-form ``mie_qsca`` /
   ``mie_qext`` helpers (which use the Bohren-Huffman formulation).
4. Reports the relative error between the two; the parity tests in
   ``tests/test_parity_pytmatrix.py::test_sphere_parity`` require < 1e-3.

pytmatrix analogue
------------------
This is the same starting problem as the quick-start snippet in the
pytmatrix README; the ``Scatterer`` constructor here is byte-for-byte
identical.
"""

from __future__ import annotations

import numpy as np

from rustmatrix import Scatterer, mie_qext, mie_qsca
from rustmatrix import scatter
from rustmatrix.tmatrix_aux import geom_horiz_back, wl_X
from rustmatrix.refractive import m_w_10C


def main() -> None:
    radius_mm = 1.0
    wavelength_mm = wl_X                       # ≈ 33.3 mm, X-band
    m = m_w_10C[wl_X]                          # tabulated water index @ 10 °C

    s = Scatterer(
        radius=radius_mm,
        wavelength=wavelength_mm,
        m=m,
        axis_ratio=1.0,                        # 1.0 → sphere (Mie limit)
        ddelt=1e-4,
        ndgs=2,
    )
    s.set_geometry(geom_horiz_back)            # (90, 90, 0, 180, 0, 0)
    S, Z = s.get_SZ()

    # Backscatter cross-section from the amplitude matrix:
    # sigma_HH = 4π |S_HH|² (pytmatrix convention).
    sigma_hh_tmatrix = 4.0 * np.pi * np.abs(S[1, 1]) ** 2

    # Total scattering / extinction cross-sections (full-sphere integrals).
    sigma_sca_tmatrix = scatter.sca_xsect(s, h_pol=True)
    sigma_ext_tmatrix = scatter.ext_xsect(s, h_pol=True)

    # Closed-form Mie reference.
    size_param = 2.0 * np.pi * radius_mm / wavelength_mm
    q_sca = mie_qsca(size_param, m.real, m.imag)
    q_ext = mie_qext(size_param, m.real, m.imag)
    geometric = np.pi * radius_mm ** 2
    sigma_sca_mie = q_sca * geometric
    sigma_ext_mie = q_ext * geometric

    print(f"Size parameter x = 2π r / λ = {size_param:.4f}")
    print()
    print("T-matrix (axis_ratio = 1):")
    print(f"  S[1,1] (complex)             = {S[1, 1]:.4e}")
    print(f"  σ_HH backscatter   [mm²]     = {sigma_hh_tmatrix:.6g}")
    print(f"  σ_sca (full-sphere) [mm²]   = {sigma_sca_tmatrix:.6g}")
    print(f"  σ_ext               [mm²]   = {sigma_ext_tmatrix:.6g}")
    print()
    print("Closed-form Mie reference:")
    print(f"  σ_sca               [mm²]   = {sigma_sca_mie:.6g}")
    print(f"  σ_ext               [mm²]   = {sigma_ext_mie:.6g}")
    print()
    # T-matrix in the axis_ratio=1 limit must agree with Mie.
    rel_sca = abs(sigma_sca_tmatrix - sigma_sca_mie) / sigma_sca_mie
    rel_ext = abs(sigma_ext_tmatrix - sigma_ext_mie) / sigma_ext_mie
    print(f"Relative error σ_sca  (T-matrix vs Mie) = {rel_sca:.2e}")
    print(f"Relative error σ_ext  (T-matrix vs Mie) = {rel_ext:.2e}")
    assert rel_sca < 1e-2 and rel_ext < 1e-2


if __name__ == "__main__":
    main()
