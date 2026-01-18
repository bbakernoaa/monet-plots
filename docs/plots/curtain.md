# Vertical Curtain Plot

Vertical curtain plots show a 2D cross-section of data, typically with time or distance on the x-axis and altitude or pressure on the y-axis.

## Example

```python
from monet_plots.plots import CurtainPlot
import xarray as xr

# da is a 2D DataArray with dimensions (level, time)
plot = CurtainPlot(da)
plot.plot(kind='pcolormesh')
```
