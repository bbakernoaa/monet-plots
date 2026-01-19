# Fingerprint Plot

Displays a variable as a heatmap across two different temporal scales, such as hour of day vs. day of year, to reveal periodic patterns.

## Example

```python
from monet_plots.plots import FingerprintPlot

plot = FingerprintPlot(df, val_col='concentration', x_scale='hour', y_scale='dayofyear')
plot.plot()
```
