# rupytmatrix

**Rust-backed T-matrix scattering for nonspherical particles.**

A port of the numerical core of [pytmatrix](https://github.com/jleinonen/pytmatrix) —
itself a Python wrapper around M. I. Mishchenko's Fortran T-matrix code —
with the Fortran replaced by pure Rust behind a PyO3 extension module.
Targets **Python 3.9–3.13** via ABI3.

> **Status: alpha / work in progress.** The project scaffold, primitives,
> and API surface are in place. The general-spheroid T-matrix assembly
> is a direct Fortran-to-Rust translation that compiles and runs but has
> **not yet been bit-parity verified** against the original pytmatrix.
> The sphere (axis ratio = 1) case reduces to classical Mie theory and
> is verified against closed-form expectations. See [Status](#status)
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

Fully implemented and unit-tested (`cargo test`, `pytest tests/test_mie.py`,
`pytest tests/test_scatterer_api.py`):

* Gauss-Legendre quadrature with endpoint / half-range options.
* Spherical Bessel `j_n`, `y_n` (real argument, up-recurrence).
* Spherical Bessel `j_n` of complex argument (down-recurrence, Mishchenko's CJB).
* Riccati-Bessel wrappers with correct derivative conventions.
* Wigner d-function helpers `VIG` / `VIGAMPL`.
* Shape radii for spheroid, Chebyshev, cylinder, gen-Chebyshev.
* Closed-form Mie scattering (sphere baseline).
* Full PyO3 Python API: `calctmat`, `calcampl`, `Scatterer`.

Implemented but **pending numerical verification** against pytmatrix:

* `tmatrix::tmatr0` — `m = 0` block of T.
* `tmatrix::tmatr` — `m > 0` blocks of T.
* `amplitude::ampl` — amplitude matrix rotation / summation.

These are direct line-by-line Fortran → Rust translations. They compile
and produce finite, well-typed output, but matching pytmatrix to the tight
tolerances parity tests demand is likely to require one debugging pass
over sign / index-convention issues (the Fortran uses 1-indexed arrays,
negative powers of `i`, and several `COMMON` blocks that alias variables
that a translation easily loses track of). The parity tests in
`tests/test_parity_pytmatrix.py` for spheroids and cylinders are
currently marked `xfail` for this reason; remove the marker as each
shape is verified.

Not yet implemented:

* Orientation averaging (single orientation only — the
  `orient_averaged_fixed` / `orient_averaged_adaptive` variants are absent).
* Size distribution integration (`psd_integrator`).
* Radar / PSD / refractive-index helper modules. These are pure Python in
  pytmatrix and could be copied over verbatim once the core is parity-
  verified (they don't touch Fortran).

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
