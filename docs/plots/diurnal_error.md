# Diurnal Error Heat Map

Visualizes model error (bias) as a function of the hour of day and another temporal dimension (e.g., month, day of week).

## Example

```python
from monet_plots.plots import DiurnalErrorPlot

plot = DiurnalErrorPlot(df, obs_col='obs', mod_col='mod', second_dim='month')
plot.plot()
```
