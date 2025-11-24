# Categorical Plot

The `categorical_plot` function creates various types of categorical plots, such as bar plots or violin plots, using `seaborn.catplot`. It is designed to work with `xarray.Dataset` or `xarray.DataArray` objects.

This plot is useful for visualizing the distribution of a quantitative variable across different categories.

```python
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from src.monet_plots.plots.categorical import categorical_plot

# Generate sample data
np.random.seed(42)
n_samples = 100
sites = ['Site A', 'Site B', 'Site C']
categories = ['Category 1', 'Category 2', 'Category 3']

data_values = np.random.rand(n_samples) * 100
site_labels = np.random.choice(sites, n_samples)
category_labels = np.random.choice(categories, n_samples)

df = pd.DataFrame({
    'value': data_values,
    'site': site_labels,
    'category': category_labels
})

# Convert to xarray.Dataset
ds = xr.Dataset.from_dataframe(df)

# Create the plot
fig, ax = categorical_plot(
    ds,
    x='category',
    y='value',
    hue='site',
    kind='bar',
    col_wrap=2,
    title='Categorical Plot Example',
    figsize=(10, 6)
)

# Display the plot (in a real script, you might save it)
plt.tight_layout()
plt.show()
```

![Categorical Plot](categorical_plot.png)
