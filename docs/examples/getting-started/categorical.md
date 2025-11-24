# Categorical Plots

Categorical plots are used to visualize the relationship between a numerical variable and one or more categorical variables. MONET Plots leverages `seaborn.catplot` to provide a flexible interface for creating various types of categorical plots, such as bar plots and violin plots.

## Prerequisites

-   Basic Python knowledge
-   Understanding of xarray, pandas, and numpy
-   MONET Plots installed (`pip install monet_plots`)
-   Seaborn installed (`pip install seaborn`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

The MONET Plots workflow for categorical plots typically involves:

1.  **Prepare Data**: Organize your data into an `xarray.Dataset`, `xarray.DataArray`, or `pandas.DataFrame` suitable for categorical plotting. Ensure you have at least one categorical variable and one numerical variable.
2.  **Call `categorical_plot`**: Use the `monet_plots.plots.categorical.categorical_plot` function, specifying the `x` (categorical) and `y` (numerical) variables, and the `kind` of plot (e.g., 'bar', 'violin').
3.  **Customize (Optional)**: Add titles, labels, and other visual enhancements.
4.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Bar Plot

Let's create a basic bar plot to show the mean and confidence interval of a numerical variable across different categories.

```python
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.categorical import categorical_plot

# 1. Prepare sample data
# Create a sample xarray DataArray with a categorical dimension
np.random.seed(42) # for reproducibility
data = xr.DataArray(
    np.random.normal(loc=10, scale=2, size=(100, 3)),
    coords={
        'sample': np.arange(100),
        'category': ['Group A', 'Group B', 'Group C']
    },
    dims=['sample', 'category'],
    name='measurement'
)

# 2. Create a basic bar plot
# Note: categorical_plot handles conversion to dataframe internally if needed
fig, ax = categorical_plot(
    data,
    x='category',
    y='measurement',
    kind='bar',
    title='Mean Measurement per Category',
    xlabel='Category',
    ylabel='Measurement Value'
)

plt.tight_layout()
plt.show()
```

### Expected Output

A bar plot will be displayed showing three bars, one for each 'Group A', 'Group B', and 'Group C'. Each bar's height will represent the mean 'measurement' value for that group, and a black line on top of each bar will indicate the 95% confidence interval around the mean.

### Key Concepts

-   **`categorical_plot`**: The primary function for generating categorical visualizations.
-   **`kind='bar'`**: This argument specifies that a bar plot should be generated. Bar plots are excellent for comparing the central tendency (mean) and spread (confidence interval) of a numerical variable across different categories.
-   **`x` and `y` parameters**: These are crucial for defining which column in your `DataFrame` represents the categorical variable (`x`) and which represents the numerical variable (`y`).

## Example 2: Basic Violin Plot

Now, let's visualize the distribution of the numerical variable across categories using a violin plot. Violin plots are useful for showing the full distribution of data, including its density, median, and quartiles.

```python
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.categorical import categorical_plot

# 1. Prepare sample data (using the same data as the bar plot example)
np.random.seed(42) # for reproducibility
data = xr.DataArray(
    np.random.normal(loc=10, scale=2, size=(100, 3)),
    coords={
        'sample': np.arange(100),
        'category': ['Group A', 'Group B', 'Group C']
    },
    dims=['sample', 'category'],
    name='measurement'
)

# 2. Create a basic violin plot
fig, ax = categorical_plot(
    data,
    x='category',
    y='measurement',
    kind='violin',
    title='Distribution of Measurement per Category',
    xlabel='Category',
    ylabel='Measurement Value'
)

plt.tight_layout()
plt.show()
```

### Expected Output

A violin plot will be displayed, showing three "violins," one for each 'Group A', 'Group B', and 'Group C'. Each violin's shape will represent the kernel density estimate of the 'measurement' values within that group, providing a visual representation of the data's distribution. Inside each violin, a small box plot will indicate the median and interquartile range.

### Key Concepts

-   **`kind='violin'`**: This argument specifies a violin plot. Violin plots are a powerful way to visualize the distribution of a numerical variable for several categories, offering more detail than a simple box plot.
-   **Distribution Shape**: The width of the violin at any given point indicates the density of data points at that value.
-   **Internal Box Plot**: By default, a mini box plot is drawn inside each violin, showing the median and interquartile range.

## Example 3: Categorical Plot with Hue

You can further break down the data by adding another categorical variable using the `hue` parameter, which will create separate violins or bars for each level of the `hue` variable within each `x` category.

```python
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from monet_plots.plots.categorical import categorical_plot

# 1. Prepare sample data with an additional 'condition' category
np.random.seed(42) # for reproducibility
n_samples_per_group = 50
categories = ['Group A', 'Group B']
conditions = ['Condition X', 'Condition Y']

data_list = []
for cat in categories:
    for cond in conditions:
        # Simulate data with slightly different means for conditions
        if cond == 'Condition X':
            loc = 10
        else:
            loc = 12
        measurements = np.random.normal(loc=loc, scale=1.5, size=n_samples_per_group)
        temp_df = pd.DataFrame({
            'category': cat,
            'condition': cond,
            'measurement': measurements
        })
        data_list.append(temp_df)

df_hue = pd.concat(data_list).reset_index(drop=True)
# Convert to xarray DataArray for compatibility with categorical_plot expectations if needed,
# but categorical_plot also accepts DataFrames if we skip the 'name' check or ensure it has one.
# Let's stick to the DataFrame which is also supported by the underlying sns.catplot,
# but we need to wrap it in an xarray object if the function strictly requires it.
# Looking at the source code, it does `if isinstance(data, xr.DataArray)...` and then `df = data.to_dataframe()`.
# So if we pass a DataFrame, it might fail on `.to_dataframe()`.
# Let's wrap it in an xarray Dataset to be safe, or just assume we can patch it.
# Actually, the source code shows: `df = data.to_dataframe().reset_index()`
# This implies `data` MUST be an xarray object.
# Let's convert our DataFrame to an xarray Dataset.
ds_hue = xr.Dataset.from_dataframe(df_hue)


# 2. Create a violin plot with 'hue'
fig, ax = categorical_plot(
    ds_hue,
    x='category',
    y='measurement',
    hue='condition',
    kind='violin',
    title='Measurement Distribution by Category and Condition',
    xlabel='Category',
    ylabel='Measurement Value'
)

plt.tight_layout()
plt.show()
```

### Expected Output

A violin plot will be displayed with two main categories ('Group A', 'Group B') on the x-axis. Within each category, there will be two violins side-by-side, one for 'Condition X' and one for 'Condition Y', distinguished by color. This allows for a direct comparison of the measurement distribution across both categorical variables.

### Key Concepts

-   **`hue` parameter**: This argument allows you to introduce a third categorical variable, which will be represented by different colors or styles within each `x` category. It's powerful for showing interactions or sub-group comparisons.
-   **Grouped Distributions**: The plot effectively visualizes how the distribution of `measurement` changes not only across `category` but also within `condition`.
