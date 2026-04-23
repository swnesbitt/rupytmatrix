# Contributing

Issues and pull requests are welcome at
[github.com/swnesbitt/rustmatrix](https://github.com/swnesbitt/rustmatrix).

## Development install

```bash
git clone https://github.com/swnesbitt/rustmatrix.git
cd rustmatrix
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest "numpy>=1.23,<2" "scipy>=1.10,<1.14"
maturin develop --release
pytest tests/
```

`maturin develop --release` compiles the Rust extension into the
active venv. Iterative Rust edits are picked up by re-running that
one command — no need to reinstall.

For the docs site:

```bash
pip install -e ".[docs]"
maturin develop --release        # notebooks import rustmatrix
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

## Tests

* **Python unit tests** — `pytest tests/` (97 tests, ~30 s).
* **Parity tests vs pytmatrix** — `pytest tests/ -m parity`. Needs
  `pytmatrix` installed, which in turn needs a Fortran compiler
  (`gfortran`) on your machine. On macOS: `brew install gcc`.
* **Rust crate tests** — `cargo test --lib --release`.

CI matrix covers Python 3.10–3.13 on Ubuntu; the release workflow
publishes ABI3 wheels for CPython 3.8+ on macOS (arm64, x86_64),
Linux (manylinux x86_64, aarch64), and Windows x86_64.

## Code style

* **Python** — `ruff format` + `ruff check` on every PR. Line length
  100. Targets Python 3.9+.
* **Rust** — `cargo fmt --all` + `cargo clippy --all-targets -- -D
  warnings`. Edition 2021.
* **Docstrings** — NumPy style (`Parameters`, `Returns`, `Notes`,
  `References`, `Examples` sections) — Sphinx/napoleon picks them up
  directly.

## Release flow

1. Bump `version` in both `Cargo.toml` and `pyproject.toml` to the
   new semver tag (they must match).
2. Update `docs/changelog.md` with the user-facing highlights.
3. Commit + tag (`git tag -a vX.Y.Z -m "Release X.Y.Z — …"`) +
   push both.
4. The GitHub Actions release workflow builds wheels for every
   supported platform and uploads to PyPI.
5. `cargo publish` from a clean checkout publishes the crate.
6. docs.rs and Read the Docs pick up the tag automatically; verify
   the new version page builds within ~15 min.

## Where to touch what

| To change | Edit | Rebuild |
|---|---|---|
| T-matrix kernel, orientation averaging | `src/**.rs` | `maturin develop --release` |
| Polarimetric helpers, PSD classes | `python/rustmatrix/*.py` | none (pure Python) |
| Tutorials | `examples/*.py` | `python examples/_build_notebooks.py` → regenerates `.ipynb` |
| Docs | `docs/**.md`, `docs/conf.py` | `sphinx-build docs docs/_build/html` |
| Rust crate metadata | `Cargo.toml` | — |
