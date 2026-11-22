# Scientific plot quality implementation plan

**Date** 2026-11-20
**Approved input** [Four-figure specification](spec.md)

## Summary

Implement Taylor auto-fit first through both public entry points. Then add normalized Taylor comparisons and the companion error table. Deliver scoped journal and operations presets, three additional scientific presentation upgrades, and a reproducible gallery. Python 3.11+, Matplotlib, NumPy, pandas, xarray, Cartopy, and pytest are the existing stack. Use headless Linux CI. No persistent storage or new runtime dependency is planned.

## Constitution check

The checkout has no Spec Kit constitution or extension configuration. Keep existing plot calls, return types, and export defaults compatible. The Taylor domain fix applies by default because silently clipping valid points is the defect. New presentation presets remain opt-in.

## Implementation order and code ownership

1. **Taylor domain (FR-009–010, SC-003/005).** Work in `src/monet_plots/taylordiagram.py` and `src/monet_plots/plots/taylor_diagram.py`. Prototype updating the floating grid limits and correlation ticks in place. Confirm that the host view limits, auxiliary transform, and boundary clipping all track the new domain before adopting this approach. If the available Matplotlib API cannot preserve those invariants, prototype a supported polar axes implementation against the existing entry points first. Keep the same artist handles for earlier points. Allow their marker glyphs to cross the boundary without changing data coordinates. Regenerate managed centered RMS contours when the domain grows, preserving explicit levels, styles, and wrapper labels. Add focused tests to both existing Taylor test modules before implementation.
2. **Normalized Taylor comparisons and error table (FR-011–012, SC-006).** Extend `plots/taylor_diagram.py` to accept explicit comparison groups with reference and model columns. Validate and aggregate finite pairs per model. Feed the resulting dimensionless points through the auto-fit engine. Add an independent opt-in companion table with raw and, when defined, normalized errors. Reuse a single row of computed metrics for the plot and table. Preserve legacy raw Taylor calls. Cover mixed units, missing pairs, constant offsets, zero variance, and repeat plotting in the wrapper tests.
3. **Presentation and export (FR-001/005–006, SC-002/004).** Extend `style.py` and `plots/base.py`. Apply new presets in scoped figure creation, plot drawing, and saving. Test mixed presets, style restoration on exceptions, and save after interleaved plot calls. Preserve explicit overrides. Export PDF text as text and raster with explicit width and DPI. Use a layout that preserves physical page dimensions and verify fit at those dimensions.
4. **Scientific labels and scales (FR-002–005/008, SC-003/004).** Extend `plots/timeseries.py` for metadata and spread labels. Extend `plots/spatial_bias_scatter.py` and reuse `colorbars.py` for stable nonzero limits, fixed limits across comparable panels, and overflow. Extend `plots/reliability_diagram.py` for empty-bin gaps, count cues, and independently verified skill semantics. Use the existing metrics implementation where its result agrees with independent fixtures. Test small bias, zero bias, unknown units, missing data, sparse bins, and long labels.
5. **Gallery and release evidence (FR-007, SC-001–004).** Add deterministic examples in `examples/` and document them in `docs/`. Produce baseline figures from the pinned revision and journal/operations versions from matching inputs. Publish captions and text/data equivalents. Add headless semantic, export, and visual checks to the existing pytest/CI workflow. Use deterministic local map fixtures in CI.

## Planning artifacts

- [Research and decisions](research.md)
- [Figure data and invariants](data-model.md)
- [Proposed Python and export contracts](contracts/plotting.md)
- [Validation quickstart](quickstart.md)
