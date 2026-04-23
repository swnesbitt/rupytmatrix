# Quickstart

Five minutes, three snippets. By the end you'll have built a sphere,
deformed it into a raindrop, and integrated across a gamma PSD to get
real radar observables — the same workflow the tutorials then unpack in
depth.

If you haven't already, [install rustmatrix](install) first. Every
snippet below runs top-to-bottom as-is.

## 1. A sphere (the Mie limit)

The cleanest sanity check: a 1 mm water sphere at X-band. Setting
`axis_ratio=1` forces the T-matrix into its Mie limit, and
`rustmatrix.mie_qsca` gives the closed-form answer to compare against.

```python
import numpy as np
from rustmatrix import Scatterer, mie_qsca, scatter
from rustmatrix.refractive import m_w_10C
from rustmatrix.tmatrix_aux import geom_horiz_back, wl_X

radius_mm = 1.0
s = Scatterer(radius=radius_mm, wavelength=wl_X, m=m_w_10C[wl_X],
              axis_ratio=1.0)
s.set_geometry(geom_horiz_back)

sigma_sca_tmatrix = scatter.sca_xsect(s, h_pol=True)
size_param = 2 * np.pi * radius_mm / wl_X
sigma_sca_mie = mie_qsca(size_param, m_w_10C[wl_X].real,
                         m_w_10C[wl_X].imag) * np.pi * radius_mm ** 2

print(f"σ_sca  T-matrix = {sigma_sca_tmatrix:.4g} mm²")
print(f"σ_sca  Mie      = {sigma_sca_mie:.4g} mm²")
print(f"relative error  = {abs(sigma_sca_tmatrix - sigma_sca_mie) / sigma_sca_mie:.1e}")
```

The two should agree to ~1e-4. This is the parity gate all of rustmatrix
rests on.

## 2. A single raindrop (differential reflectivity)

Real drops are oblate. `dsr_thurai_2007(D)` gives the canonical
horizontal-to-vertical axis ratio for an equivalent-volume diameter
`D` in mm, so the only change from the sphere is `axis_ratio =
1 / dsr_thurai_2007(D)`. Swap to C-band while we're here.

```python
import numpy as np
from rustmatrix import Scatterer, radar
from rustmatrix.refractive import m_w_10C
from rustmatrix.tmatrix_aux import (K_w_sqr, dsr_thurai_2007,
                                    geom_horiz_back, geom_horiz_forw, wl_C)

D = 2.0  # mm, equivalent-volume diameter
s = Scatterer(radius=D / 2.0, wavelength=wl_C, m=m_w_10C[wl_C],
              axis_ratio=1.0 / dsr_thurai_2007(D), Kw_sqr=K_w_sqr[wl_C])

s.set_geometry(geom_horiz_back)
Zh  = 10 * np.log10(radar.refl(s, h_pol=True))
Zdr = 10 * np.log10(radar.Zdr(s))

s.set_geometry(geom_horiz_forw)
Kdp = radar.Kdp(s)

print(f"Z_h  = {Zh:7.2f} dBZ  (per drop / m³)")
print(f"Z_dr = {Zdr:7.3f} dB")
print(f"K_dp = {Kdp:7.4f} °/km (per drop / m³)")
```

Backscatter geometry drives `Z_h` and `Z_dr`; forward geometry drives
`K_dp` and specific attenuation `A_i`. Calling `set_geometry` is cheap —
the T-matrix itself is cached on the `Scatterer`.

## 3. A full PSD (what a radar actually sees)

One drop isn't a radar echo. A PSD is. `PSDIntegrator` tabulates the
amplitude and phase matrices once across the diameter range — that's
where the parallel Rust kernel earns its keep — then any number of
PSD shapes get evaluated from the cached table.

```python
import numpy as np
from rustmatrix import Scatterer, radar, psd
from rustmatrix.refractive import m_w_10C
from rustmatrix.tmatrix_aux import (K_w_sqr, dsr_thurai_2007,
                                    geom_horiz_back, geom_horiz_forw, wl_C)

s = Scatterer(wavelength=wl_C, m=m_w_10C[wl_C], Kw_sqr=K_w_sqr[wl_C])
integ = psd.PSDIntegrator()
integ.D_max = 8.0
integ.num_points = 64
integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
integ.geometries = (geom_horiz_back, geom_horiz_forw)
s.psd_integrator = integ
integ.init_scatter_table(s)                          # one-time Rust sweep

for name, D0 in [("stratiform", 1.0), ("convective", 2.0), ("heavy", 3.0)]:
    s.psd = psd.GammaPSD(D0=D0, Nw=8e3, mu=4)
    s.set_geometry(geom_horiz_back)
    Zh  = 10 * np.log10(radar.refl(s, h_pol=True))
    Zdr = 10 * np.log10(radar.Zdr(s))
    s.set_geometry(geom_horiz_forw)
    Kdp = radar.Kdp(s)
    print(f"{name:>10}  D0={D0} mm  "
          f"Z_h={Zh:6.2f} dBZ  Z_dr={Zdr:5.2f} dB  K_dp={Kdp:5.2f} °/km")
```

Same three observables, now physically meaningful.

## What just happened

| Object | Role |
|---|---|
| `Scatterer` | The T-matrix solver. Give it size, shape, refractive index, wavelength; it caches the T-matrix and reuses it across geometries and PSDs. |
| `radar.*` | Polarimetric observables (`refl`, `Zdr`, `Kdp`, `rho_hv`, `delta_hv`, `Ai`). |
| `psd.PSDIntegrator` | Tabulates `S(D)` and `Z(D)` once; evaluates arbitrary PSDs from the table. Parallel Rust under the hood. |
| `psd.GammaPSD` | Normalised-gamma PSD (Bringi & Chandrasekar). `ExponentialPSD`, `UnnormalizedGammaPSD`, `BinnedPSD` are also available. |
| `tmatrix_aux` | Radar-band wavelengths (`wl_S`, `wl_C`, …, `wl_W`), `\|K_w\|²` values, drop-shape relations, scattering geometries. |
| `refractive` | Tabulated refractive indices for water and ice across the radar bands. |

## Next steps

Pick a tutorial that matches what you're trying to do — they're
numbered roughly in order of difficulty. [Tutorial index →](tutorials/index)
