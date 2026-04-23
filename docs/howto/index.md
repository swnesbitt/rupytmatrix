# How-to recipes

Short, task-focused pages — each answers one question, with a
runnable snippet. Unlike the [tutorials](../tutorials/index), these
don't build on each other.

| Topic | Recipe |
|---|---|
| [Plug in your own PSD](custom_psd) | Subclass `psd.PSD` for a custom analytic form |
| [Use a measured fall-speed curve](custom_fallspeed) | Feed a disdrometer / lab $v_t(D)$ into `SpectralIntegrator` |
| [Wrap a measured antenna pattern](custom_beam) | Import a range-scan CSV as a `TabulatedBeam` |
| [Combine species with HydroMix](hydromix) | Rain + ice + graupel through one Scatterer-shaped object |
| [Profile and benchmark](profiling) | Where the Rust speedups apply; how to measure |

## Related projects

* **[myPSD](https://github.com/swnesbitt/myPSD)** — an interactive
  web frontend for radar simulation that drives `rustmatrix` under
  the hood. Drop in a PSD and see the polarimetric / spectral
  radar response update live; good companion for pedagogy and
  sensitivity exploration.

```{toctree}
:hidden:
:maxdepth: 1

custom_psd
custom_fallspeed
custom_beam
hydromix
profiling
```
