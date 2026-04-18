<p align="center">
  <img src="assets/icon.svg" alt="rupytmatrix icon" width="160" height="160">
</p>

# rupytmatrix

**Rust-backed T-matrix scattering for nonspherical particles.**

A port of the numerical core of [pytmatrix](https://github.com/jleinonen/pytmatrix) —
itself a Python wrapper around M. I. Mishchenko's Fortran T-matrix code —
with the Fortran replaced by pure Rust behind a PyO3 extension module.
Targets **Python 3.9–3.13** via ABI3.

> **Status: alpha.** The core T-matrix solver is numerically verified against
> the original pytmatrix (Fortran backend) for spheres, prolate/oblate
> spheroids, and finite cylinders. All 7 parity tests pass. See [Status](#status)
> below for specifics.

## Why?

* Replace a Fortran dependency with a pure Rust dependency that cross-compiles
  cleanly to every platform Python 3.13 cares about (including Apple Silicon
  and Windows, where the original pytmatrix has historically been awkward).
* Avoid the `numpy.f2py` / `distutils` build path, which broke in Python 3.12+.
* Modern build tooling (maturin + PyO3 0.22, abi3 wheels).

## Installation

```bash
# From a checkout:
git clone <your-fork-of-this-repo> rupytmatrix
cd rupytmatrix

# Dev install — builds the Rust extension and puts it on sys.path.
pip install maturin
maturin develop --release

# Or build a wheel:
maturin build --release
pip install target/wheels/rupytmatrix-*.whl
```

Requires a Rust toolchain (`rustup default stable`, 1.75+) and Python 3.9+.

## Usage

The `Scatterer` class is API-compatible with `pytmatrix.tmatrix.Scatterer`:

```python
from rupytmatrix import Scatterer

s = Scatterer(
    radius=1.0,                     # mm, equal-volume-sphere radius
    wavelength=33.3,                # mm, X-band
    m=complex(7.99, 2.21),          # water at 10 GHz
    axis_ratio=1.0,                 # 1.0 = sphere
    ddelt=1e-4,                     # convergence tolerance
    ndgs=2,                         # quadrature density factor
)
s.set_geometry((90.0, 90.0, 0.0, 180.0, 0.0, 0.0))
S, Z = s.get_SZ()
```

Shape constants follow pytmatrix's conventions (`SHAPE_SPHEROID = -1`,
`SHAPE_CYLINDER = -2`, `SHAPE_CHEBYSHEV = 1`).

See `examples/basic_usage.py`.

## Architecture

```
rupytmatrix/
├─ Cargo.toml                  # Rust crate (PyO3 + maturin)
├─ pyproject.toml              # Python build (maturin backend)
├─ src/
│  ├─ lib.rs                   # Module root + Python extension entrypoint
│  ├─ quadrature.rs            # Gauss-Legendre (port of Mishchenko's GAUSS)
│  ├─ special.rs               # spherical Bessel (RJB, RYB, CJB)
│  ├─ wigner.rs                # Wigner d-functions (VIG, VIGAMPL)
│  ├─ shapes.rs                # Particle shapes (RSP1/2/3/4)
│  ├─ mie.rs                   # Closed-form Mie for the sphere limit
│  ├─ tmatrix.rs               # T-matrix solver (CALCTMAT, CONST, VARY, TMATR0, TMATR)
│  ├─ amplitude.rs             # Amplitude + phase matrix (CALCAMPL, AMPL)
│  └─ pybindings.rs            # PyO3 exposure
├─ python/rupytmatrix/
│  ├─ __init__.py              # Public Python API
│  ├─ scatterer.py             # Scatterer class (matches pytmatrix signature)
│  └─ _core.pyi                # Type stubs for the Rust extension
├─ tests/
│  ├─ conftest.py              # pytmatrix-availability fixture
│  ├─ test_mie.py              # Mie unit tests
│  ├─ test_scatterer_api.py    # API smoke tests
│  ├─ test_parity_pytmatrix.py # Parity vs. original pytmatrix (skipped if missing)
│  └─ test_quadrature_python.py
├─ examples/basic_usage.py
├─ benches/                    # (empty — reserved for cargo-bench suites)
└─ .github/workflows/ci.yml
```

Heavy linear algebra is handled by `nalgebra` (complex LU inversion for the
Q matrix). This replaces the `lpd.f` LAPACK routines that ship with the
original pytmatrix Fortran backend.

## Status

**Numerically verified** against pytmatrix (Fortran backend) for all supported
shapes. All 7 parity tests pass at tolerances ≤ 5×10⁻³ for S and Z:

| Shape | Test | Tolerance |
|---|---|---|
| Sphere (`axis_ratio=1`) — 3 cases | ✅ pass | 1×10⁻³ |
| Prolate spheroid (`axis_ratio=0.5`) | ✅ pass | 5×10⁻³ |
| Oblate spheroid (`axis_ratio=2.0`) | ✅ pass | 5×10⁻³ |
| Spheroid (`axis_ratio=1.5`) | ✅ pass | 5×10⁻³ |
| Finite cylinder | ✅ pass | 5×10⁻³ |

Fully implemented and unit-tested (`cargo test --lib`, `pytest`):

* Gauss-Legendre quadrature with endpoint / half-range options.
* Spherical Bessel `j_n`, `y_n` (real argument, up-recurrence).
* Spherical Bessel `j_n` of complex argument (down-recurrence, Mishchenko's CJB).
* Riccati-Bessel wrappers with correct derivative conventions.
* Wigner d-function helpers `VIG` / `VIGAMPL`.
* Shape radii for spheroid, Chebyshev, cylinder, gen-Chebyshev.
* Closed-form Mie scattering (sphere baseline).
* `tmatrix::tmatr0` — `m = 0` azimuthal block of T.
* `tmatrix::tmatr` — `m > 0` azimuthal blocks of T.
* `amplitude::ampl` — amplitude matrix rotation / summation.
* Full PyO3 Python API: `calctmat`, `calcampl`, `Scatterer`.
* Orientation averaging: `orient_single` (default), `orient_averaged_fixed`
  (Gauss quadrature in β, uniform sampling in α), and
  `orient_averaged_adaptive` (scipy `dblquad`). Ported pure-Python from
  pytmatrix and parity-verified against it.
* `gaussian_pdf` / `uniform_pdf` orientation PDFs and the Gautschi-based
  `get_points_and_weights` quadrature helper (used internally by
  `orient_averaged_fixed`).
* Size-distribution integration: `ExponentialPSD`, `UnnormalizedGammaPSD`,
  `GammaPSD`, `BinnedPSD`, and `PSDIntegrator` (tabulate-then-trapezoid
  averaging over an ``N(D)``). All four tabulation paths — single
  orientation, fixed-quadrature orientation averaging, adaptive
  orientation averaging, and single-orient `angular_integration=True`
  (per-diameter `sca_xsect` / `ext_xsect` / `asym`) — are implemented
  in Rust and parallelised across diameters via `rayon` (GIL released).
  Against Fortran pytmatrix: ~4× on single-orient PSD, ~10× on
  orientation-averaged PSD (64 points), ~300× on `angular_integration`
  (64 points), and ~400× on orient-averaged-adaptive PSD. Parity-verified
  against pytmatrix on all four paths.
* Angular-integrated scattering helpers (`sca_intensity`, `ldr`,
  `sca_xsect`, `ext_xsect`, `ssa`, `asym`) and radar-band auxiliary
  constants (`wl_S`..`wl_W`, `K_w_sqr`, geometry presets, Thurai /
  Pruppacher-Beard / Beard-Chuang drop-shape relationships).
* Refractive-index helpers: Maxwell-Garnett and Bruggeman EMAs,
  tabulated water refractive indices at 0/10/20 °C for all six radar
  bands, and an ice/snow interpolator bundled with the Warren ice
  optical-constants table (``rupytmatrix.refractive.mi``).
* Polarimetric radar observables: `radar_xsect`, `refl`/`Zi`, `Zdr`,
  `delta_hv`, `rho_hv`, `Kdp`, `Ai` (direct port of pytmatrix's
  `radar.py`, works on both single orientations and PSD-integrated
  scatterers).

## Performance

Against the Fortran `pytmatrix` backend on the same machine (Apple M-series,
`benches/bench_vs_pytmatrix.py`; positive ratio = rupytmatrix faster):

| Workload | pytmatrix (Fortran) | rupytmatrix (Rust) | Speedup |
|---|---:|---:|---:|
| `calctmat` only (spheroid, ax = 1.5) | 0.22 ms | 0.21 ms | 1.1× |
| Single orientation, cold (fresh `Scatterer`) | 0.23 ms | 0.26 ms | 0.9× (slower) |
| Cached re-evaluation (warm T-matrix) | 0.01 ms | 0.00 ms | 1.6× |
| Orientation-averaged fixed (4 × 8 = 32 orientations) | 4.26 ms | 0.75 ms | **5.7×** |
| PSD `init_scatter_table`, 32 points | 12.3 ms | 2.2 ms | **5.7×** |
| PSD `init_scatter_table`, 64 points | 13.5 ms | 3.4 ms | **4.0×** |
| PSD + orient-avg (4 × 8), 32 points | 13.8 ms | 1.7 ms | **8.2×** |
| PSD + orient-avg (4 × 8), 64 points | 23.4 ms | 2.4 ms | **9.7×** |
| PSD + `angular_integration`, 32 points | 13 345 ms | 52 ms | **258×** |
| PSD + `angular_integration`, 64 points | 26 210 ms | 87 ms | **300×** |
| PSD + orient-avg-adaptive, 4 points | 1 758 ms | 4.1 ms | **433×** |

Headline notes:

* The core T-matrix solve is roughly tied — heavily-optimised Fortran plus
  LAPACK is hard to beat on pure linear algebra, so "comparable" is the
  expected result there.
* The big wins come from moving the outer loops into Rust: orientation
  averaging (~6× on a single particle) and especially PSD tabulation
  (~4× single-orient, ~10× combined with orient averaging), because the
  per-diameter T-matrix solves are independent and `rayon` parallelises
  them across cores with the GIL released.
* The outlier 100×–400× speedups on `angular_integration` and
  `orient_averaged_adaptive` come from replacing scipy's per-diameter
  `dblquad` callbacks with a fixed Gauss-Legendre product grid evaluated
  inside Rust. The callbacks cross the Python/Fortran boundary hundreds
  of times per diameter on pytmatrix; the Rust path amortises the
  T-matrix solve across the whole grid and runs diameters in parallel.
* The ~16% slowdown on the "single orient cold" case is Python/PyO3
  boundary overhead that F2PY avoids. It disappears as soon as the
  T-matrix is reused (cached re-eval, orientation averaging, PSD tabulation).

Reproduce with:

```bash
pip install pytmatrix    # needs gfortran
python benches/bench_vs_pytmatrix.py
```

## Running the tests

```bash
# Rust-only tests (fast, no Python needed):
cargo test --lib

# Full test suite (requires maturin develop first):
pytest -v tests/

# Parity tests against pytmatrix (requires pytmatrix installed):
pip install pytmatrix
pytest -v tests/test_parity_pytmatrix.py
```

CI runs the Rust + Python tests across Python 3.10 / 3.11 / 3.12 / 3.13
on Linux. Parity tests against pytmatrix run locally when the package is
installed; they're not in CI because pytmatrix needs gfortran.

## License

MIT. See [LICENSE](./LICENSE).

Upstream credits:

* Jussi Leinonen — pytmatrix (the Python wrapper and modifications that
  made the Mishchenko code MIT-compatible).
* Michael I. Mishchenko (NASA GISS) — the underlying T-matrix Fortran code.
