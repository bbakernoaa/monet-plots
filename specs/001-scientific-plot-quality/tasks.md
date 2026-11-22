# Implementation tasks for scientific plot quality

Inputs are the [specification](spec.md), [approved plan](plan.md), [plotting contract](contracts/plotting.md), and [validation guide](quickstart.md). Paths below are relative to the repository root. Record evidence and check each task after its validation passes.

## P1. Interpret model evaluation figures

- [x] T001 Establish the baseline using the existing headless tests, starting with `tests/test_taylordiagram.py` and `tests/plots/test_taylor_diagram.py`. Record pre-existing failures and render matched baseline figures from the pinned revision with deterministic inputs. Evidence supports SC-001 and SC-004.
- [x] T002 [P1] After T001, add failing direct API cases to `tests/test_taylordiagram.py`. Update `src/monet_plots/taylordiagram.py` to grow radial and angular domains on insertion, including after draw, while preserving previous artists and refreshing managed contours. Verify high spread, correlations −1/0/+1, boundary glyphs, and contour values with headless PNG/PDF renders. Covers FR-009, FR-010, SC-003, and SC-005.
- [x] T003 [P1] After T002, update `src/monet_plots/plots/taylor_diagram.py` and `tests/plots/test_taylor_diagram.py` to preserve the existing wrapper contract and verify fresh and reused diagrams, including shared coordinates, distinct labels, and undefined correlation diagnostics. Verify centered RMS labels. Covers FR-009, FR-010, and SC-005.
- [x] T004 [P1] After T003, add opt-in normalization, explicit comparison groups, and the optional error table in `src/monet_plots/plots/taylor_diagram.py`. In `tests/plots/test_taylor_diagram.py` verify same-pair calculations, cross-unit scaling invariance, divisor-N RMS decomposition, constant offsets, zero-spread diagnostics with valid raw errors, no finite pairs, and distinct table rows for coincident markers. Covers FR-011, FR-012, and SC-006.
- [x] T005 [P1] After T007, update `src/monet_plots/plots/timeseries.py` and `tests/test_timeseries_stats.py` to verify available units, time basis, spread band labels, missing metadata, and independently calculated means and spread. Covers FR-002, FR-008, and SC-003.
- [x] T006 [P1] After T007, update `src/monet_plots/plots/spatial_bias_scatter.py`, `src/monet_plots/colorbars.py` as needed, and `tests/test_colorbars_scaling.py` plus a plot-level check in `tests/plots/test_spatial_bias_scatter.py`. Verify sign, units, zero-centered scales, shared limits, overflow, missing versus zero, and zero or sub-unit spread. Covers FR-003, FR-008, and SC-003.
- [x] T008 [P1] After T007, update `src/monet_plots/plots/reliability_diagram.py` and `tests/plots/test_reliability_diagram.py`. Verify frequencies and counts against a hand-calculated sample, preserve empty-bin gaps, and validate any retained skill shading against climatology. Covers FR-004, FR-008, and SC-003.

## P2. Export journal figures and briefing graphics

- [x] T007 [P2] After T004, add scoped journal and operations presets in `src/monet_plots/style.py`, `src/monet_plots/plots/base.py`, and the four pilot entry points. Verify existing style and save behavior, overrides, mixed figures, exception restoration, physical PDF width, selectable PDF text, and raster dimensions and resolution in `tests/test_style_activation.py` and `tests/test_base.py`. Covers FR-001, FR-006, SC-002, and SC-004.
- [x] T009 [P2] After T005, T006, and T008, review dense layouts, accessible color and independent visual cues across the four plot modules. Render journal and operations examples at declared sizes and inspect grayscale and color vision simulations, long labels, empty inputs, and clipped elements. Covers FR-005, FR-008, and SC-002.

## P3. Reproduce and verify figures

- [x] T010 [P3] After T009, deliver reproducible inputs and export instructions in `examples/plot_scientific_quality.py` and `docs/examples/scientific-plot-quality.md`. Document all four baselines and eight journal/operations figures, including captions, text/data equivalents, provenance, and export settings. Cover normalized Taylor comparisons and the error table. Covers FR-005, FR-007, and SC-001.
- [x] T011 [P3] After T010, add headless render and semantic checks in `tests/plots/test_scientific_plot_quality.py` and wire them into `.github/workflows/ci.yml` if needed. Verify visual tolerance on deterministic local data, exports, and legacy calls. Run the focused and full suites and record any failures with evidence. Covers SC-002, SC-003, and SC-004.

## Execution order

T001 → T002 → T003 → T004 → T007 → T005/T006/T008 → T009 → T010 → T011. T005, T006, and T008 may proceed independently after T007.

## Implementation evidence

- Baseline Taylor tests passed on the pinned revision (6 passed). All four baseline figures rendered from that revision.
- Taylor regressions reproduced fixed bounds and missing normalization before the changes. The updated wrapper and direct API exercised both signed correlation and radial growth, preservation of an earlier marker, centered RMS contours, pairing, and undefined correlation.
- The gallery exported four journal PDFs at 468 × 288 pt with extractable text, four 600 dpi journal PNGs, and four 150 dpi operations PNGs, with matched synthetic inputs.
- Implementation choice. Use Matplotlib polar axes with public setters for radial/angular limits. This preserves existing sample artist handles while the coordinate domain grows. The Taylor wrapper refreshes its managed contour labels after expansion.
- Reviewed the gallery in color, grayscale, and protanopia/deuteranopia simulations. Added shape cues for station-bias sign and a regression for a long Taylor legend in journal output. Empty reliability inputs give an explicit diagnostic. Empty time series and all-missing station bias render headlessly.
- Final headless pytest run finished with 204 passed and 1 skipped. The gallery regression verifies pixel-identical repeated operations renders. All tasks are checked.
