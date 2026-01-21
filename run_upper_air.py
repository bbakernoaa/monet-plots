import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.upper_air import UpperAir

# 1. Prepare sample data
lats = np.linspace(30, 50, 20)
lons = np.linspace(-125, -70, 30)
hgt = np.random.uniform(5000, 6000, (20, 30))
u = np.random.uniform(-20, 20, (20, 30))
v = np.random.uniform(-20, 20, (20, 30))

# 2. Initialize and plot
plot = UpperAir(lat=lats, lon=lons, hgt=hgt, u=u, v=v, figsize=(10, 8))
plot.plot()

plt.title("Upper Air Example (500 hPa)")
plt.savefig("upper_air_plot.png")
print("Saved upper_air_plot.png")
