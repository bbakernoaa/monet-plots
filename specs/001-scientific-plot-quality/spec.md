# Scientific plot quality for MONET Plots

**Created** 2026-11-20
**Status** Draft for approval
**Baseline** `aa919945296ab839d7ab9f6f9aa18a3667cbb6e8`

## Scope

Improve four static figures used in model evaluation. Start with time series, station bias maps, forecast reliability diagrams, and Taylor diagrams. These cover time, geography, probabilistic verification, and model comparison. Offer opt-in presentation settings for journal figures and routine NOAA/NWS review through the existing plotting entry points. Use one shared set of presentation conventions across the four figures. Defer migration of other plot families and changes to interactive rendering.

A style-only pass would improve typography and layout quickly. A shared style plus four pilot figures also lets reviewers verify the meaning of uncertainty bands, bias colors, and skill annotations. A library-wide rollout would cover more products but require a substantially larger verification matrix. This draft proposes the four-figure pilot.

Custom skill-score contours, rank-based coordinates, and separate Target or Solar diagrams are deferred.

## User scenarios and acceptance

### P1. Interpret model evaluation figures

An evaluator needs to identify what the data show without relying on undocumented plot conventions.

1. Given known timestamps and repeated measurements, the time series labels the variable and supplied units, identifies the time basis, and names the interval used for any shaded band. A fixture distinguishes a standard deviation band from a confidence interval.
2. Given stations with negative, zero, positive, and out-of-range bias, the map labels the difference direction and units, shows zero at the center, and identifies values outside the scale. Identical values map to identical colors when panels use shared limits.
3. Given forecast probabilities with empty bins and a hand-calculated observed event frequency, the reliability diagram shows the reference diagonal, preserves empty-bin gaps, and exposes counts when available. Any skill shading agrees with an independently calculated reference for the selected baseline.

4. Given models with spread above the initial radial limit and correlations of −1, 0, and +1, the Taylor diagram shows every valid marker in the displayed domain. Adding a model to an existing diagram expands the domain as needed and preserves earlier markers, their labels, and any centered RMS difference contours.

5. Given paired model and reference series for variables with different units, normalized mode places each reference at unit radius and zero angle. Rescaling a pair by the same positive factor preserves its normalized point and centered RMS difference. The plot identifies each model and its reference group.
6. Given a model equal to the reference plus a constant offset, its Taylor marker coincides with the reference. The companion table lists the model separately and shows its signed bias, total RMSE, and zero centered RMS difference within numerical tolerance.

### P2. Export a paper figure or an operational briefing graphic

An author or analyst needs the same scientific content in media with different reading sizes.

1. Given a representative input for each figure, the author can export a PDF with vector text and a high-resolution raster at the requested publication width. Figure text, legends, and colorbars fit within the exported bounds.
2. Given the same input, the analyst can produce an operational PNG in a headless run. Titles, series, reference marks, and units remain legible at the intended display size.
3. Given a figure with multiple series, a reader can distinguish the primary series and references under grayscale and common color vision deficiency simulations. Each reference gallery figure has a plain-language description and a data or summary equivalent.

### P3. Reproduce figures in unattended workflows

A maintainer needs repeatable output and evidence that existing usage still works.

1. Given the same data and settings in the same test environment, repeated renders agree within a documented visual tolerance.
2. Given an existing call without the new options, the call continues to work with the same defaults, return type, and export path behavior.
3. Given two figures with different new presets in a single process, saving either figure preserves its selected presentation and later plots retain their intended settings.

## Functional requirements

- **FR-001** Provide opt-in journal and operations presentation settings for all four pilot figures. Preserve existing call signatures and default presentation.
- **FR-002** Present available units and time context consistently. Where metadata is unavailable, omit the unit or explicitly identify it as unknown. Identify any uncertainty or spread band according to the statistic it depicts.
- **FR-003** Give signed bias maps a zero-centered scale. Support explicit shared limits, distinguish missing data from zero, show overflow, and state the sign convention.
- **FR-004** Keep forecast reliability axes comparable on the 0–1 scale. Explain sample support when available and validate any displayed claim of skill against its named baseline.
- **FR-005** Distinguish key series and reference marks through labels, markers, or line patterns as well as color. Use ordered colors for magnitudes and a diverging scale for signed bias. Provide text equivalents for the example gallery.
- **FR-006** Export configurable physical dimensions and raster resolution. Preserve vector text in PDF. Support headless exports through the normal save workflow and avoid cross-figure style leakage for the new settings.
- **FR-007** Supply paired before-and-after reference figures and reproducible inputs for each pilot figure. Record the input data source or fixture identifier, units, relevant time or event definition, and settings needed to reproduce each example.
- **FR-008** Exercise empty and all-missing data, constant or sub-unit bias, sparse reliability bins, missing metadata, and long labels. Each case must yield either a scientifically interpretable figure or a clear diagnostic.

- **FR-009** Automatically fit the reference and all valid model points in Taylor diagrams, including samples added to an existing diagram. Treat the configured radial scale as an initial extent. Allow negative correlations where needed. Keep markers visible at domain boundaries and identify coincident samples while preserving their coordinates.
- **FR-010** Label Taylor contours as centered RMS difference. Report models with undefined or invalid Taylor coordinates explicitly. For zero-variance inputs, where correlation is undefined, identify the affected model or reference without inventing a correlation. Keep other valid points visible.

- **FR-011** Offer opt-in Taylor normalization for multiple variables or regimes on one diagram. Divide each model standard deviation and centered RMS difference by its paired reference standard deviation. Use Pearson correlation and dimensionless axis and contour labels. Identify reference groups and report zero or undefined reference spread explicitly.
- **FR-012** Offer a companion Taylor table with model and reference identifiers, signed mean bias (model minus reference), centered RMS difference, total RMSE, and paired sample count. Show raw errors with units where known and dimensionless equivalents scaled by reference standard deviation in normalized mode. Compute each row from the same finite pairs. Use mean squared deviations with divisor N for the companion RMS decomposition and normalized Taylor geometry so total RMSE squared equals centered RMS difference squared plus bias squared. Preserve legacy defaults for callers who use the existing raw diagram.

## Success criteria

- **SC-001** Produce eight reference outputs covering each of the four figures in both presentation settings, plus before examples for all four.
- **SC-002** Review all eight outputs at their declared target size. Find zero clipped required labels, legends, or colorbars. Check publication PDF width against the chosen journal width within 1 mm. Check raster exports against the declared output dimensions and resolution.
- **SC-003** Match independently specified fixture values for time-series mean and spread, station bias sign and scale, reliability bin frequency and count, and Taylor coordinates and centered RMS difference. Assess any reliability skill shading against the named baseline before including it in the gallery.
- **SC-004** Pass automated semantic checks, export smoke checks, and visual comparisons for all four pilot plots in headless CI. Verify representative legacy calls using existing tests or focused regression checks.

- **SC-005** Verify both the direct Taylor diagram API and the plot wrapper with a model above the radial limit, a negative correlation, a score exactly on a boundary, and an outlier added after the initial render. Check exported marker visibility and preservation of previous samples. Verify zero-variance diagnostics and coincident labels.

- **SC-006** Verify unit reference radius and invariance under positive rescaling using two variables with different units and spreads. Compare table values to hand calculations for constant offset and partially missing pairs. Verify RMS decomposition within numerical tolerance and explicit reporting for zero reference spread. Report valid raw table metrics when normalization is undefined.

## Baseline findings

- `src/monet_plots/style.py` provides `paper`, `wiley`, and other presets. `BasePlot` applies styles globally and delegates saving to the figure. The cartographic helper also has a separate export path with its own DPI defaults.
- `src/monet_plots/plots/timeseries.py` computes a mean plus or minus one standard deviation for tabular input and can display `(None)` as a unit.
- `src/monet_plots/plots/spatial_bias_scatter.py` rounds the 95th percentile of absolute bias before scaling markers. Small or zero differences can therefore produce zero scale limits.
- `src/monet_plots/plots/reliability_diagram.py` shades between climatology and the perfect-reliability diagonal under a positive-skill description. Verify that annotation against the stated baseline before treating it as a scientific claim.

- `src/monet_plots/taylordiagram.py` bounds the plot at 1.5 times the reference standard deviation by default and draws only the nonnegative correlation quadrant. `TaylorDiagramPlot` creates its axes and contours before adding model samples.

## Assumptions

- AMS figure guidance is the proposed publication reference. Physical width and raster resolution should follow the target journal and artwork type. [AMS figure guidance](https://www.ametsoc.org/ams/publications/author-information/figure-information-for-authors/)
- NOAA/NWS use means internal model-evaluation and briefing workflows for this pilot. Agency branding, mandated product palettes, and formal accessibility conformance require their own review if requested. The examples follow federal guidance on redundant visual cues and text alternatives. [Section 508 color guidance](https://www.section508.gov/create/making-color-usage-accessible/), [Section 508 alt-text guidance](https://www.section508.gov/create/alternative-text/)
- Scientific calculations remain compatible with existing workflows. Any confirmed error in a scientific annotation needs a documented correction and independent fixture expectations.
