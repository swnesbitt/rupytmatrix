# Use a measured fall-speed curve

`rustmatrix.spectra` ships several analytic $v_t(D)$ relations
(`atlas_srivastava_sekhon_1973`, `brandes_et_al_2002`, `beard_1976`,
Locatelli-Hobbs variants for snow). If your observation instead has
a measured $v_t(D)$ from a disdrometer, videosonde, or wind-tunnel
lab — use it directly.

## The contract

`SpectralIntegrator`'s `fall_speed` parameter is **any callable** that
maps a 1-D `np.ndarray` of diameters in **mm** to a 1-D array of
fall speeds in **m s⁻¹**, same shape. That's the whole interface.

## Example — interpolating a disdrometer table

```python
import numpy as np
from rustmatrix import Scatterer, spectra, psd
from rustmatrix.tmatrix_aux import wl_W, K_w_sqr, geom_vert_back
from rustmatrix.refractive import m_w_10C

# Observed v_t(D) — e.g. from an OTT Parsivel or a 2DVD.
D_obs = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])      # mm
v_obs = np.array([1.20, 2.10, 4.03, 6.49, 8.10, 8.84, 9.20, 9.40])  # m/s

def v_t_measured(D):
    """Piecewise-linear interpolation of a measured v_t curve."""
    return np.interp(D, D_obs, v_obs, left=0.0, right=v_obs[-1])

s = Scatterer(wavelength=wl_W, m=m_w_10C[wl_W], Kw_sqr=K_w_sqr[wl_W])
# ... PSDIntegrator setup as in the quickstart ...
s.psd = psd.ExponentialPSD(N0=8000, Lambda=2.0)
s.set_geometry(geom_vert_back)

specint = spectra.SpectralIntegrator(
    v_bins=np.linspace(-2, 12, 281),
    fall_speed=v_t_measured,        # <-- your callable
    sigma_t=0.3,
)
result = specint.compute(s)
```

## Extrapolation

Pick a policy for $D$ outside the measured range:

* **`np.interp(..., left=0.0, right=v_max)`** — clamp. Fine if your
  PSD tail is already small there.
* **Power-law splice** — match one of the analytic relations
  (Atlas-Srivastava, Brandes) smoothly at the boundary.
* **Reject** — raise if `D > D_obs.max()` so you can't accidentally
  integrate into noise.

The right answer depends on whether your drop-size distribution
extends beyond where the measurement is trustworthy; clamp-to-last
is usually safe enough for rain PSDs.

## Habit-dependent fall speeds (mixed phases)

For a `HydroMix`, each `MixtureComponent` gets its own fall-speed
kernel — pass a different callable per species:

```python
mix = spectra.SpectralIntegrator(
    v_bins=v_bins,
    fall_speed={"rain": v_t_measured, "snow": snow_v_t_curve},
    sigma_t=0.3,
)
```

See the [SLW-vs-snow tutorial](../tutorials/10_slw_vs_snow) for a
worked example mixing liquid-cloud and snow fall-speed models.

## See also

* [`rustmatrix.spectra`](../api/spectra) — the fall-speed presets
  shipped with the package.
* [The spectra background page](../background/spectra.md) for the
  Doppler model.
