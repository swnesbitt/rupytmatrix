# Combine species with HydroMix

`HydroMix` bundles two or more `Scatterer` objects — typically
different hydrometeor species, each with its own PSD and
orientation distribution — into a single object that quacks like a
`Scatterer`. Every `rustmatrix.radar` helper, every
`rustmatrix.scatter` cross-section, and `SpectralIntegrator`
accept a `HydroMix` wherever they accept a `Scatterer`, so you can
layer species without rewriting your observable code.

## The pattern

```python
from rustmatrix import Scatterer, HydroMix, MixtureComponent, radar
from rustmatrix.tmatrix_aux import (wl_C, K_w_sqr, geom_horiz_back,
                                    dsr_thurai_2007)
from rustmatrix.refractive import m_w_10C, m_i
from rustmatrix import psd

Kw = K_w_sqr[wl_C]

# --- Rain species: oblate water drops, Thurai shape, Gaussian canting
rain = Scatterer(wavelength=wl_C, m=m_w_10C[wl_C], Kw_sqr=Kw)
integ = psd.PSDIntegrator()
integ.D_max, integ.num_points = 8.0, 64
integ.axis_ratio_func = lambda D: 1.0 / dsr_thurai_2007(D)
integ.geometries = (geom_horiz_back,)
rain.psd_integrator = integ
integ.init_scatter_table(rain)

# --- Ice species: columnar, low density, wide canting
ice = Scatterer(wavelength=wl_C, m=m_i[wl_C], Kw_sqr=Kw)
integ_i = psd.PSDIntegrator()
integ_i.D_max, integ_i.num_points = 12.0, 64
integ_i.axis_ratio_func = lambda D: 0.6       # fixed for a column
integ_i.geometries = (geom_horiz_back,)
ice.psd_integrator = integ_i
integ_i.init_scatter_table(ice)

# --- Combine
mix = HydroMix(
    components=[
        MixtureComponent(scatterer=rain,
                         psd=psd.GammaPSD(D0=1.5, Nw=4e3, mu=2)),
        MixtureComponent(scatterer=ice,
                         psd=psd.ExponentialPSD(N0=3000, Lambda=2.0)),
    ],
    Kw_sqr=Kw,
)

mix.set_geometry(geom_horiz_back)
print(f"Z_h  = {10 * np.log10(radar.refl(mix)):6.2f} dBZ")
print(f"Z_dr = {10 * np.log10(radar.Zdr(mix)):6.2f} dB")
```

The mixture evaluates each component against the shared geometry,
sums the scattering / phase matrices linearly in $N(D)$, and
presents the result through the same `get_S` / `get_Z` / `get_SZ`
interface the bulk `radar.*` helpers expect.

## Why linear summation is correct

Polarimetric observables depend on integrals of $|S|^2$ or $\Re(S)$
weighted by $N(D)$. For disjoint species with distributions
$N_a(D)$ and $N_b(D)$,

$$
\int \! |S_i|^2 \left[ N_a + N_b \right] dD
= \int \! |S_i|^2 N_a \, dD + \int \! |S_i|^2 N_b \, dD,
$$

i.e. the mixture's polarimetric bulk is the arithmetic sum of
per-species bulks. The same holds for spectra — `S_spec` and
`Z_spec` are linear in $N(D)$, so bimodal SLW + snow Doppler
spectra (e.g. [tutorial 10](../tutorials/10_slw_vs_snow)) drop out
automatically.

## Gotchas

* **Geometries must match.** Each component's `PSDIntegrator` must
  have been `init_scatter_table`-d on **the same geometry** you
  query at mix time. The HydroMix does not build its own table.
* **Scale $N(D)$, not amplitudes.** `Nw` / `N0` on each component's
  PSD expresses that species' volumetric share. Don't post-scale
  observables.
* **Kw matches the detection convention.** `Kw_sqr` on the mix is
  what `radar.refl` uses to turn $|S|^2$ into mm⁶ m⁻³ — almost
  always water at the radar band, even when the species is ice.
  (This is the radar-meteorology convention for "reflectivity
  factor.")

## Beyond two species

Three, four, or more components are fine — the Rust kernel doesn't
care about cardinality, and each component is just another term in
the linear sum. A rain + graupel + hail melting-layer case is
straight assembly of three `MixtureComponent`s with their own
T-matrix tables.

## See also

* [`rustmatrix.hd_mix`](../api/hd_mix) — the public API.
* [HydroMix tutorial](../tutorials/06_hd_mix) — rain + oriented ice
  at C-band.
* [SLW-vs-snow tutorial](../tutorials/10_slw_vs_snow) — bimodal
  Doppler spectrum from a HydroMix.
