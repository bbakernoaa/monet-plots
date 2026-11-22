# Proposed plotting contracts

- Keep current constructors, return values, `add_sample`, `add_contours`, and `save` available. Add the opt-in `journal` and `operations` values to the existing plot style option and document their sizing defaults.
- Treat Taylor `scale` as the initial radial extent. Valid inputs to `add_sample(stddev, corrcoef, ...)` fit automatically. Preserve earlier artist handles and kwargs. Tolerate only roundoff outside the correlation domain. Reject invalid direct coordinates with a clear error. The wrapper reports undefined models individually and renders the valid models.
- Existing save kwargs keep precedence over new preset choices. An explicit filename extension chooses the format unless the caller supplies an explicit format. Journal examples include a PDF and print-resolution raster. Operations examples include PNG.
- Caption or figure text identifies applicable variable, unit, time basis, event/baseline, or bias sign. Text equivalents accompany gallery examples.

## Taylor comparison options

- Propose keyword-only `normalize=False` and `show_error_table=False` options on `TaylorDiagramPlot`. Support existing single-reference inputs and explicit `comparison_groups` in normalized mode. Each group supplies a label, reference column, model columns, and optional common raw unit. Retain the existing axes return value.
- Render optional companion rows with a clear unit or dimensionless label for each applicable value. Keep table rows for undefined geometry. Refresh the table when comparisons are added to the same diagram and retain distinct rows for coincident markers.
