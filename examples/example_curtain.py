import xarray as xr
import numpy as np
import pandas as pd
from monet_plots.plots.curtain import CurtainPlot

# Create dummy 2D data
times = pd.date_range("2023-01-01", periods=24, freq="h")
levels = np.linspace(1000, 100, 10)
data = np.exp(-((np.arange(24) - 12)**2) / 20)[:, np.newaxis] * np.exp(-((np.arange(10) - 5)**2) / 10)
da = xr.DataArray(data.T, coords=[levels, times], dims=["level", "time"], name="concentration")

# Initialize and plot
plot = CurtainPlot(da)
plot.plot(kind='contourf', cmap='viridis')
plot.save('curtain_example.png')
print("Curtain plot saved to curtain_example.png")
