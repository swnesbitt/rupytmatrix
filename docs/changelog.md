# Changelog

The authoritative changelog is the
[GitHub Releases page](https://github.com/swnesbitt/rustmatrix/releases).

This page is a Phase 1 stub. Subsequent doc phases will mirror the
release notes here so they are searchable and versioned alongside the
rest of the documentation.

## Latest

* **v2.1.1** — First crates.io publish of the Rust crate
  (`cargo add rustmatrix` / [docs.rs/rustmatrix](https://docs.rs/rustmatrix)).
  New documentation site at
  [rustmatrix.readthedocs.io](https://rustmatrix.readthedocs.io) —
  five background / theory pages, a one-stop conventions reference,
  per-module Python API pages, five task-focused how-to recipes,
  all 14 tutorial notebooks rendered with live figures, and a
  full bibliography. `pyo3/extension-module` is now gated behind
  a feature flag so plain `cargo build` and docs.rs work.
* **v2.1.0** — `spectra.beam` module (Gaussian / Airy / Tabulated beam
  patterns; `BeamIntegrator` for pattern × scene integration), tutorials
  13 and 14.
* **v2.0.1** — Logo and branding refresh.
* **v2.0.0** — Initial public release of the Rust-accelerated rewrite of
  pytmatrix.
