# Validation quickstart

From the checkout root, install the project and its test dependencies using the CI environment setup. Run the focused Taylor checks first.

```sh
MPLBACKEND=Agg python -m pytest tests/test_taylordiagram.py tests/plots/test_taylor_diagram.py
```

Use a fixed nonconstant reference series. Test a model with ten times its spread, an anticorrelated model, an identical model, and a constant model. Add another high-spread sample after rendering the first figure and drawing contours. Check both API paths, marker and label visibility in saved figures, contour growth, and earlier artist identity. Include a constant-offset model to distinguish centered RMS difference from ordinary RMSE.

Run the full headless suite after integrating the other pilots.

```sh
MPLBACKEND=Agg python -m pytest
```

Review the four baseline figures and eight proposed preset figures at their labeled sizes. Check PDF page width to within 1 mm, selectable vector text, raster dimensions and resolution, and clipping. Check grayscale and color vision simulations, text equivalents, fixture agreement, and matching rerenders within the documented tolerance.

## Normalization and companion table checks

Use reference values [0, 2] and model values [3, 5]. Verify unit radius, Pearson correlation +1, centered RMS difference 0, bias +3, total RMSE 3, and paired count 2. The reference standard deviation with divisor N is 1. Compare on the same diagram with a second group scaled by a positive factor and labeled in distinct units. Check that their dimensionless values coincide while raw errors scale and both model identifiers remain visible.

Use reference values [0, 2, 4, missing] and model values [1, missing, 7, 9]. The two retained differences are [1, 3]. Expect bias 2, centered RMS difference 1, total RMSE sqrt(5), and count 2. Test zero reference spread and undefined model correlation separately and inspect the exported table for diagnostics and clipping.
