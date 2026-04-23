# Profile and benchmark

`rustmatrix` is mostly-Rust under the hood but called from Python.
The speedup vs `pytmatrix` (~6× orientation averaging, ~10× PSD
tabulation, ~430× `orient_averaged_adaptive`) depends on which
call you make. This page tells you where the Rust kernels apply,
where you're paying Python tax, and how to measure.

## Where Rust runs

| Operation | Rust? |
|---|---|
| `Scatterer(...)` construction | Python (cheap) |
| `scatterer.set_geometry(...)` | Python dispatch + **Rust** rotation |
| `scatter.amplitude_matrix`, single geometry | **Rust** T-matrix build + rotation |
| `orientation.orient_averaged_fixed` | **Rust**, GIL released, parallel |
| `orientation.orient_averaged_adaptive` | **Rust**, GIL released, parallel — this is where you see 100× speedups |
| `psd.PSDIntegrator.init_scatter_table` | **Rust**, parallel across diameters |
| PSD integration against the cached table | NumPy on the Python side (cheap) |
| `radar.refl`, `Zdr`, `Kdp`, …  | NumPy / Python (cheap; inputs are the pre-integrated bulks) |
| `spectra.SpectralIntegrator.compute` | Python loop over velocity bins + NumPy; the T-matrix work is already done |

Rule of thumb: **anything that evaluates the T-matrix at many
diameters × orientations** lives in Rust and parallelises across
cores. Everything downstream (polarimetric algebra, PSD weighting,
spectral assembly) is NumPy-speed.

## Quick benchmarks

The `benches/` directory ships a pytmatrix head-to-head:

```bash
uv pip install pytmatrix
python benches/bench_vs_pytmatrix.py
```

That script runs each of the hot operations against the same problem
on both backends and prints wall-time ratios. Don't benchmark on a
cold import — the first T-matrix call pays a one-time JIT-like cost
for the pyo3 bindings.

## Profiling your own code

The usual Python profilers work, but they can't descend into the
Rust side. What they *can* do is tell you whether you're spending
time in the Rust kernel or in your own glue:

```bash
python -m cProfile -s cumulative my_script.py | head -30
```

If the hot function is `rustmatrix._core.calctmat` or
`init_scatter_table`, you're in Rust — any further speedup needs
fewer calls, not faster ones. If the hot function is somewhere in
`spectra.py` or in your own code, you have Python-side work to cut.

## Common speed traps

* **Rebuilding the scatter table per PSD.** `init_scatter_table`
  is expensive; `s.psd = new_psd; integrate` is nearly free.
  Build the table once per (shape, refractive-index, wavelength)
  tuple and swap PSDs against it.
* **Calling `set_geometry` on every drop.** Geometry switches are
  cheap (no T-matrix re-compute) but still Python-side; call once
  per back / forward sweep, not in an inner loop.
* **`num_points` too large.** 64 quadrature diameters is almost
  always enough; 256 rarely changes the answer past the third
  digit and quadruples the Rust cost.
* **Single-geometry calls in a loop instead of one vectorised
  PSD integration.** If you're iterating over raindrop diameters
  in Python and calling `Scatterer` per drop, stop — build a
  `PSDIntegrator` and let the Rust kernel batch.

## When to reach for `cargo bench`

The Rust crate has its own microbenchmarks (`cargo bench`). You
want them when:

* you're modifying the Rust kernels themselves;
* you want to isolate Rust-only cost from Python overhead;
* you care about per-diameter T-matrix timings, not end-to-end
  polarimetrics.

For application-level work ("does my spectrum run fast enough?"),
the Python benchmarks in `benches/` are the right altitude.
