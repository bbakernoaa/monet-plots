# Conditional Quantile Plot

Plots the distribution (quantiles) of modeled values as a function of binned observed values.

## Example

```python
from monet_plots.plots import ConditionalQuantilePlot

plot = ConditionalQuantilePlot(df, obs_col='obs', mod_col='mod')
plot.plot(show_points=True)
```
