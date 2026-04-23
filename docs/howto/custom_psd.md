# Plug in your own PSD

`rustmatrix.psd` ships exponential, gamma, normalised-gamma, and
binned PSDs — enough for most radar-meteorology cases. If your work
needs a custom functional form (e.g. a lognormal, an observed
particle size distribution, a mixture of two gammas, an event-
conditioned parameterisation)
subclass `psd.PSD` and you're done.

## The contract

`PSD` is a two-method base class:

```python
class PSD:
    def __call__(self, D):       # N(D) in mm⁻¹ m⁻³ at diameter D in mm
        ...
    def __eq__(self, other):      # used by PSDIntegrator to invalidate caches
        ...
```

Two rules:

1. `__call__` must handle **both scalars and arrays**. The
   `PSDIntegrator` calls it on an `np.ndarray` of quadrature
   diameters; other callers will hit it with Python scalars. Use
   `np.shape(D) == ()` or `np.asarray(D)` to branch.
2. `__eq__` must return `False` against a PSD with different
   parameters. `PSDIntegrator` inspects `sca.psd == old_psd` to
   decide whether the cached observable integrals are still good;
   a `True` return means "same PSD, reuse the cached integrals."

## Example — a lognormal PSD

```python
import numpy as np
from rustmatrix import psd

class LognormalPSD(psd.PSD):
    """N(D) = (N_T / (√(2π) σ D)) · exp(-(ln D - ln D_g)² / (2 σ²))."""

    def __init__(self, NT: float, Dg: float, sigma: float):
        self.NT = float(NT)        # total number concentration [m⁻³]
        self.Dg = float(Dg)        # geometric mean diameter [mm]
        self.sigma = float(sigma)  # geometric std dev of ln D
        # PSDIntegrator reads self.D_max to know where to truncate.
        self.D_max = Dg * np.exp(4 * sigma)

    def __call__(self, D):
        D = np.asarray(D, dtype=float)
        z = (np.log(np.maximum(D, 1e-30)) - np.log(self.Dg)) / self.sigma
        return self.NT / (np.sqrt(2 * np.pi) * self.sigma * D) \
               * np.exp(-0.5 * z * z)

    def __eq__(self, other):
        return (type(self) is type(other)
                and (self.NT, self.Dg, self.sigma)
                == (other.NT, other.Dg, other.sigma))

    def __hash__(self):
        return hash((type(self), self.NT, self.Dg, self.sigma))
```

Assign it as you would any other PSD:

```python
from rustmatrix import Scatterer, radar, psd
from rustmatrix.tmatrix_aux import wl_C, K_w_sqr, dsr_thurai_2007, geom_horiz_back
from rustmatrix.refractive import m_w_10C

s = Scatterer(wavelength=wl_C, m=m_w_10C[wl_C], Kw_sqr=K_w_sqr[wl_C])
integ = psd.PSDIntegrator()
integ.D_max = 8.0
integ.num_points = 64
integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
integ.geometries = (geom_horiz_back,)
s.psd_integrator = integ
integ.init_scatter_table(s)

s.psd = LognormalPSD(NT=1e3, Dg=1.5, sigma=0.35)
s.set_geometry(geom_horiz_back)
print(f"Z_h = {10*np.log10(radar.refl(s)):.2f} dBZ")
```

## Performance note

The PSD is evaluated at whatever quadrature diameters
`PSDIntegrator` chose (`num_points` between `D_min=0.0` and
`D_max`). Expensive `__call__` bodies are fine — the Rust T-matrix
tabulation is what dominates, not the PSD eval. Don't bother
vectorising aggressively unless a profile says otherwise.

## See also

* [`rustmatrix.psd`](../api/psd) — the built-in PSDs you might want
  to subclass instead of starting from `PSD`.
* [Gamma-PSD tutorial](../tutorials/03_psd_gamma_rain) — worked
  example of the integration loop.
