"""Basic usage example.

Backscattering amplitude for a 1 mm water drop at X-band (lambda = 33.3 mm).
"""

from __future__ import annotations

import numpy as np

from rupytmatrix import Scatterer


def main() -> None:
    s = Scatterer(
        radius=1.0,           # equal-volume-sphere radius, mm
        wavelength=33.3,      # X-band
        m=complex(7.99, 2.21),  # water at 10 GHz, approximate
        axis_ratio=1.0,       # sphere
        ddelt=1e-4,
        ndgs=2,
    )
    # Horizontal back-scatter geometry.
    s.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
    S, Z = s.get_SZ()
    print("S (amplitude matrix, mm):")
    print(S)
    print()
    print("Z (phase matrix, mm^2):")
    print(Z)
    print()
    # Back-scattering cross-section = 4 pi |S_{HH}|^2 for horizontal pol.
    sigma_hh = 4.0 * np.pi * np.abs(S[1, 1]) ** 2
    print(f"sigma_HH (mm^2) = {sigma_hh:.6g}")


if __name__ == "__main__":
    main()
