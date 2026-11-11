# Changelog

Follows Keep a Changelog + SemVer.

## [Unreleased]

### Added

- Lecture 6 frame assembly notebook with manual code numbers followed by helper
  generation, member loading, prescribed support motion, and a deformed-shape plot.
- `code_numbers_from_nodes` for full one-based code numbers in endpoint order,
  with separate conversion to zero-based Python indices.
- Planar Euler-Bernoulli beam functions following Lecture 6, with separate
  membrane and bending stiffness contributions, coordinate transformation,
  global element stiffness, and end-force recovery.
- Uniform member loads and midpoint point forces in both local directions,
  including fixed-end actions, equivalent global nodal loads, and loaded
  end-force recovery.
- `plot_frame` with linear axial and cubic Hermite transverse interpolation
  of nodal displacements, with analytical checks for element end results and plotting.

### Changed

- Added finite-and-positive property guards for truss and beam elements.
  Plotting rejects noninteger connectivity before conversion, checks index
  bounds, and rejects non-finite coordinates, displacements, and scale.

- Simplified beam postprocessing to member end actions and a Hermite-interpolated
  deformed shape. Internal-force diagrams are left as a student task. The plot
  omits member-load deformation with zero endpoint motion.

- Removed the separate equivalent-nodal-load helper and standalone examples.
  Assembly uses the negative global fixed-end actions returned by the load
  functions directly.

- Replaced duplicate truss and beam geometry functions with shared
  `geometry_line_2d` in `sta132.geometry`, updating all current callers.
  Notebook 05 pins its Colab teaching functions to the original truss commit
  `cf36c7d`; local execution continues to use the local checkout.

- Beam fixed-end and recovered end-action functions now return global actions,
  local actions, and the transformation matrix, in that order. Local-action
  inputs and teaching variables explicitly include `_local` in their names.

- Made selected Lecture 6 load-transformation and equilibrium steps student
  exercises, retaining the worked assembly and prescribed-displacement solve.

- Unified truss and beam matrix formatting, docstring structure, endpoint and
  displacement validation, and scalar naming.
