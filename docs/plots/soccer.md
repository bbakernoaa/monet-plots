# Soccer Plot

The Soccer Plot is a model evaluation tool that plots model bias against error. It typically shows Mean Fractional Bias (MFB) or Normalized Mean Bias (NMB) on the x-axis and Mean Fractional Error (MFE) or Normalized Mean Error (NME) on the y-axis.

## Example

```python
from monet_plots.plots import SoccerPlot
import pandas as pd

df = pd.DataFrame({
    'obs': [10, 20, 30],
    'mod': [12, 18, 35]
})

plot = SoccerPlot(df, obs_col='obs', mod_col='mod')
plot.plot()
```

## Parameters

- `data`: Input data (DataFrame, DataArray, etc.)
- `obs_col`: Column name for observations.
- `mod_col`: Column name for model values.
- `goal`: Dictionary with 'bias' and 'error' thresholds for the goal zone.
- `criteria`: Dictionary with 'bias' and 'error' thresholds for the criteria zone.
