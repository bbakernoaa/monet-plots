# Meteogram

A Meteogram is a time series plot that displays multiple meteorological variables (e.g., temperature, pressure, humidity, wind) for a single geographical location.

## Overview

Meteograms are essential for visualizing the evolution of local weather conditions. They typically feature stacked subplots sharing a common time axis, allowing for easy correlation between different atmospheric parameters.

## Usage

```python
from monet_plots.plots.meteogram import Meteogram
import pandas as pd

# Data should be a DataFrame with a datetime index
df = pd.DataFrame(...)
plot = Meteogram(df=df, variables=["temp", "rh", "pres"])
plot.plot()
```

## Classes

::: monet_plots.plots.meteogram.Meteogram
