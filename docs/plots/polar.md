# Bivariate Polar Plot

Shows how a variable varies with wind speed and wind direction using polar coordinates.

## Example

```python
from monet_plots.plots import BivariatePolarPlot

plot = BivariatePolarPlot(df, ws_col='ws', wd_col='wd', val_col='conc')
plot.plot()
```
