# Changelog

Follows Keep a Changelog + SemVer.

## [Unreleased]

## [0.1.0] - 2026-11-11

### Added

- Planar truss element (`sta132.truss`) and `plot_truss`.
- Planar Euler-Bernoulli beam element (`sta132.beam`) with uniform and midpoint
  member loads and end-force recovery.
- `plot_frame` with Hermite-interpolated deformed shape.
- `code_numbers_from_nodes` and shared `geometry_line_2d`.
- Notebooks 05 (truss assembly) and 06 (frame assembly).
- Unit tests and input validation for elements and plotting.
- `sta132.__version__`.

### Changed (since `cf36c7d`)

- Beam load and end-force functions return global actions, local actions, and
  the transformation matrix; local quantities use a `_local` suffix.
- Notebook 05 pins its Colab functions to commit `cf36c7d`.
