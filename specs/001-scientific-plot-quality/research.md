# Research and decisions

## Taylor geometry

The constructor uses a floating axes grid with a fixed radial maximum and a quarter-circle correlation domain. The wrapper draws contours before it adds model points. Expand only when the requested sample exceeds current coverage. A pre-scan in the wrapper alone would leave the direct incremental API broken. Retain the positive quadrant until negative correlation requires a semicircle.

Plot the model standard deviation as radius and `acos(correlation)` as angle. Centered RMS difference is the Cartesian distance from the reference point. Computing it as `hypot(sigma_model * cos(theta) - sigma_reference, sigma_model * sin(theta))` avoids squaring large values and reduces cancellation near perfect agreement. A constant offset can increase ordinary RMSE without moving a Taylor point.

Validate a genuinely rendered figure after expansion. Updating numerical attributes alone does not update a floating axes clip path or repaint a contour. The exact Matplotlib mechanism remains subject to the prototype described in the plan.

## Exports and scientific annotations

The existing global style setter and direct save forwarding show why opt-in preset state needs to follow the figure. A tight export bounding box can change physical size. Validate actual PDF page dimensions and raster pixels. Use the approved AMS and Section 508 references in the spec for figure size and accessible encodings.

Use hand-calculated fixtures for plotted statistics. For reliability shading, verify the conditional Brier score improvement against climatology or omit the shading and retain labeled reference lines. For bias, use symmetric limits around zero and report overflow explicitly. Use a small positive data-informed fallback for an all-zero scale and document it.

## Normalized comparisons and error decomposition

Build each comparison from that model and reference's finite paired rows. Use divisor N for both standard deviations in the new mode and for the mean squared errors. Compute signed bias as mean(model − reference), total RMSE from the differences, and centered RMS difference from those same differences after removing their mean. Numerically compare their squared decomposition and compare normalized centered RMS difference with the Taylor geometry.

Use unit reference radius for all normalized groups. Label groups because their matched sample subsets can differ. When reference spread is zero or fewer than two finite pairs remain, show any defined raw errors and sample counts with a diagnostic for unavailable coordinates. A constant model also has undefined Pearson correlation. With no finite pairs, show count zero and unavailable error values. Preserve the existing raw plot's standard deviation convention and document it separately from the table's divisor-N convention.
