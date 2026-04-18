"""Auxiliary constants and drop-shape relationships.

Direct port of ``pytmatrix.tmatrix_aux``. Wavelength presets (radar
bands), typical ``K_w^2`` factors, canned geometries for backscatter /
forward-scatter, and the Thurai/Pruppacher-Beard/Beard-Chuang drop
axis-ratio formulas.
"""

from __future__ import annotations

VERSION = "0.1.0"

# Typical radar wavelengths [mm] at different bands.
wl_S = 111.0
wl_C = 53.5
wl_X = 33.3
wl_Ku = 22.0
wl_Ka = 8.43
wl_W = 3.19

# Typical water dielectric factors |K_w|^2 at the above bands.
K_w_sqr = {
    wl_S: 0.93,
    wl_C: 0.93,
    wl_X: 0.93,
    wl_Ku: 0.93,
    wl_Ka: 0.92,
    wl_W: 0.75,
}

# Preset (thet0, thet, phi0, phi, alpha, beta) geometries.
geom_horiz_back = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0)
geom_horiz_forw = (90.0, 90.0, 0.0, 0.0, 0.0, 0.0)
geom_vert_back = (0.0, 180.0, 0.0, 0.0, 0.0, 0.0)
geom_vert_forw = (180.0, 180.0, 0.0, 0.0, 0.0, 0.0)


# ---------- Drop shape relationships ----------
# All return vertical/horizontal axis ratio; pass 1/dsr(...) to Scatterer
# since Scatterer expects horizontal/vertical.


def dsr_thurai_2007(D_eq: float) -> float:
    """Thurai et al. (2007) drop shape. `D_eq` in mm."""
    if D_eq < 0.7:
        return 1.0
    if D_eq < 1.5:
        return (
            1.173 - 0.5165 * D_eq + 0.4698 * D_eq ** 2
            - 0.1317 * D_eq ** 3 - 8.5e-3 * D_eq ** 4
        )
    return (
        1.065 - 6.25e-2 * D_eq - 3.99e-3 * D_eq ** 2
        + 7.66e-4 * D_eq ** 3 - 4.095e-5 * D_eq ** 4
    )


def dsr_pb(D_eq: float) -> float:
    """Pruppacher and Beard drop shape. `D_eq` in mm."""
    return 1.03 - 0.062 * D_eq


def dsr_bc(D_eq: float) -> float:
    """Beard and Chuang drop shape. `D_eq` in mm."""
    return (
        1.0048 + 5.7e-04 * D_eq - 2.628e-02 * D_eq ** 2
        + 3.682e-03 * D_eq ** 3 - 1.677e-04 * D_eq ** 4
    )
