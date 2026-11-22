# Scientific figure gallery

Use `style="journal"` for physical-size PDF and 600 dpi raster output, or `style="operations"` for a briefing PNG. Set `figsize=(width, height)` in inches when you have a specific target. Journal defaults are 6.5 × 4 inches. Operations defaults are 10 × 6 inches at 150 dpi. Check a journal's figure dimensions and resolution for your artwork before submission.

From this checkout, run:

```sh
MPLBACKEND=Agg python examples/plot_scientific_quality.py ./previews/journal journal
MPLBACKEND=Agg python examples/plot_scientific_quality.py ./previews/operations operations
```

To reproduce the baseline, run the same script with the `baseline` option and import the library from revision `aa919945296ab839d7ab9f6f9aa18a3667cbb6e8`. The script writes synthetic input CSV files next to the outputs.

| Figure | Caption and text equivalent |
| --- | --- |
| Time series | Daily synthetic station temperature, Jan 1–14, 2024 UTC. The daily mean ranges from about 282 to 288 K. Three stations contribute each day and the shading shows mean ± one sample standard deviation (1 K). See `timeseries.csv`. |
| Station bias | Forecast minus observed temperature across six synthetic stations. Errors range from −2 K in the west to +4 K in the east. The scale spans ±3 K and marks overflow. Downward triangles, circles, and upward triangles distinguish negative, zero, and positive bias. See `station_bias.csv`. |
| Reliability diagram | Five forecast probability bins for a synthetic temperature exceedance event. The 0.5 bin has no forecasts, so there is a break in the curve. The forecast is close to the diagonal on either side of the gap. The baseline event frequency is 37/79. See `reliability.csv`. |
| Taylor diagram | The offset model has the same spread and correlation as the reference but a +3 K bias. The high-spread model has four times the reference spread. The inverted model has correlation −1. See `taylor.csv` and `taylor_statistics.csv`. |

Taylor diagrams expand to fit high spread and negative correlations. Pass `normalize=True` to compare dimensionless spread and centered RMS difference. Pass `show_error_table=True` to display paired bias, centered RMS difference, total RMSE, and sample count. Use `comparison_groups` to provide named reference/model column groups and units for a normalized comparison. A constant series has undefined Pearson correlation. The table retains valid error values for that series. In raw mode with the table enabled, markers and table values use the same complete timestamps across the selected models and divisor-N spread. Normalized comparisons use finite pairs for each model.

## More forecast verification diagrams

Run the bundled example for ensemble ranks, ROC, and event performance.

```sh
MPLBACKEND=Agg python examples/plot_verification_presets.py ./previews/journal
MPLBACKEND=Agg python examples/plot_verification_presets.py ./previews/operations --style operations
```

Use the same `plot()` and `save()` calls with your own data. For rank histograms, `facet_col` makes comparable panels, and `n_members` includes empty ranks at the ends. For ROC curves, `label_col` identifies curves and the legend reports AUC. For performance diagrams, pass contingency counts and a grouping column. The plot distinguishes configurations by marker shape and reports CSI for groups with one point.

## Grouped scorecards and weather displays

Pass `row_group` and `col_group` to `ScorecardPlot.plot()` to divide a long-form table into scientific groups. Supply one value per grouped row and column combination. The journal and operations presets show signed values, neutral dashes for missing cells, and a horizontal scale centered on the reference value. Pass `baseline` to name the reference for precomputed differences, `units` for the scale, and `lower_is_better` to identify the direction of improvement. Give several scorecards the same `vmin` and `vmax` to compare them directly. An arrow on the scale marks values beyond the selected limits, and cell labels retain the actual values.

Supply a Boolean `sig_col` to mark cells with an asterisk. The plot uses that field as provided. The synthetic example uses a pointwise paired t-test at 0.05 with no multiple-testing adjustment.

```sh
MPLBACKEND=Agg python examples/plot_grouped_scorecard.py ./previews/scorecard
```

Meteograms share a compact time axis across variables. Curtain plots use the same labels for datetime coordinates. Wind arrows and barbs keep small grids intact and thin dense grids to at most 16 samples per dimension by default. Pass a positive `skip` to set an explicit stride. Diurnal bias plots use a zero-centered scale and label the reference, which can be set with `reference_label`.

### Scorecards over valid time

Use `TemporalScorecardPlot` when each column is an actual verification period. Set `y_col="region"` for one row per region. Set `y_col="variable", row_group="region"` for a block of variables within each region. Provide one precomputed score per row and timestamp. Set `frequency="D"` for daily scores or `frequency="6h"` for six-hour periods. The plot sorts timestamps, inserts missing periods, and labels the time zone when supplied. For mixed variables, provide comparable score units, such as percent change from a control.

```sh
MPLBACKEND=Agg python examples/plot_temporal_scorecard.py ./previews/temporal-scorecard
```
