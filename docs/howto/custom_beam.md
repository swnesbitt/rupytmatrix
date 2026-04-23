# Wrap a measured antenna pattern

`rustmatrix.spectra.beam.GaussianBeam` and `AiryBeam` cover the two
canonical analytic patterns. If you have a measured range-azimuth
pattern from an antenna-range scan or a manufacturer's datasheet,
feed it in as a `TabulatedBeam` instead.

## The contract

`TabulatedBeam(theta, gain)` is a drop-in `BeamPattern`:

* `theta` — 1-D array of off-axis angles in **radians**, monotonic
  increasing, starting at or near 0.
* `gain` — 1-D array of **one-way power gain** (linear, not dB) at
  each angle. Normalise so that the peak is 1.0; the absolute
  level doesn't matter (integration normalises it).

Side-lobe angles past the last table entry get extrapolated to zero.

## Example — import a measured H-plane cut

```python
import numpy as np
from rustmatrix import spectra
from rustmatrix.spectra import beam

# Measured H-plane cut from an antenna-range scan — angle in degrees,
# gain in dB (one-way).
theta_deg, gain_dB = np.loadtxt("antenna_pattern_h.csv",
                                delimiter=",", unpack=True)

# Convert to the BeamPattern convention.
theta_rad = np.deg2rad(theta_deg)
gain_lin  = 10 ** (gain_dB / 10.0)
gain_lin /= gain_lin.max()                # peak-normalise

# Fold about θ=0 if the measurement sweeps ±θ_max: keep only θ ≥ 0.
mask = theta_rad >= 0
pattern = beam.TabulatedBeam(theta_rad[mask], gain_lin[mask])
```

Then pass `pattern` straight into `BeamIntegrator`:

```python
bi = beam.BeamIntegrator(
    pattern=pattern,
    scene=beam.marshall_palmer_scene(Z_dBZ_map, N0=8000),
    v_bins=np.linspace(-2, 12, 281),
    # ... other SpectralIntegrator kwargs
)
spectrum = bi.compute(scatterer)
```

## Gotchas

* **Power, not voltage.** `TabulatedBeam` expects *power* gain. If
  your CSV is in field-strength / voltage units, square it first.
* **One-way, not two-way.** Many antenna-range reports give the
  two-way pattern (what the radar measures in transmit-receive).
  For the beam-integration math in `rustmatrix.spectra.beam`, use
  the one-way pattern — the two-way weight is applied internally as
  $G^2$.
* **θ in radians.** Degrees is a frequent source of silent bugs;
  assertions inside `TabulatedBeam` flag angles > π, but the
  *range* of valid radian values is usually [0, 0.1] for a 5°
  beam.

## E-plane vs H-plane

A real antenna has slightly different patterns in the E-plane and
H-plane. `rustmatrix.spectra.beam` treats the beam as axisymmetric.
If you need the full 2-D pattern, wrap a `BeamPattern` subclass that
takes `(theta, phi)` — `TabulatedBeam` is a reasonable copy-paste
template.

## See also

* [`rustmatrix.spectra.beam`](../api/spectra.beam) — the beam-
  integration API.
* [Beam-pattern × scene tutorial](../tutorials/14_beam_pattern_scene) —
  worked example with `GaussianBeam` over a convective scene.
