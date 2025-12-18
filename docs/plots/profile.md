# Profile Plots

## ProfilePlot

::: monet_plots.plots.profile.ProfilePlot

## VerticalSlice

::: monet_plots.plots.profile.VerticalSlice

### Example

```python
import numpy as np
from monet_plots.plots import VerticalSlice

# Create sample data
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create the plot
plot = VerticalSlice(x=X, y=Y, z=Z)
plot.plot()
```

## StickPlot

::: monet_plots.plots.profile.StickPlot

### Example

```python
import numpy as np
from monet_plots.plots import StickPlot

# Create sample data
u = np.random.rand(10)
v = np.random.rand(10)
y = np.arange(10)

# Create the plot
plot = StickPlot(u, v, y)
plot.plot()
```
