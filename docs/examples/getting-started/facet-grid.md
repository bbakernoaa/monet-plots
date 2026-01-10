# Facet Grid Plots

Facet grids (also known as "trellis plots" or "small multiples") are powerful visualizations that display the distribution of one or more variables, or the relationship between multiple variables, separately for each level of one or more categorical variables. MONET Plots provides the `FacetGridPlot` class, which wraps Seaborn's `FacetGrid` to simplify the creation of these complex, multi-panel plots while maintaining consistent styling.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Seaborn installed (`pip install seaborn`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create a facet grid plot with `monet_plots`:

1.  **Prepare Data**: Ensure your data is in a `pandas.DataFrame`, `xarray.Dataset`, or `xarray.DataArray` format, with clear categorical variables for faceting, and numerical variables for plotting within each facet.
2.  **Initialize `FacetGridPlot`**: Create an instance of `FacetGridPlot`, specifying the `data` and the categorical variables for `row`, `col`, and/or `hue` to define the grid's layout and coloring.
3.  **Map a Plotting Function**: Use the `grid.map()` or `grid.map_dataframe()` method to apply a Matplotlib or Seaborn plotting function (e.g., `plt.scatter`, `sns.histplot`) to each facet, specifying the `x` and `y` variables to be plotted.
4.  **Customize (Optional)**: Adjust titles, labels, or other aesthetic properties of the grid.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic Facet Grid with Scatter Plots

Let's create a facet grid to explore the relationship between two numerical variables (`x_data`, `y_data`) across different levels of a categorical variable (`category`).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from monet_plots.plots.facet_grid import FacetGridPlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility
n_samples_per_category = 100
categories = ['Group A', 'Group B', 'Group C']

data_list = []
for i, cat in enumerate(categories):
    x = np.random.normal(loc=i * 5, scale=2, size=n_samples_per_category)
    y = 0.5 * x + np.random.normal(loc=0, scale=1, size=n_samples_per_category) + i * 3
    temp_df = pd.DataFrame({
        'x_data': x,
        'y_data': y,
        'category': cat
    })
    data_list.append(temp_df)

df = pd.concat(data_list).reset_index(drop=True)

# 2. Initialize FacetGridPlot
# We'll create a column for each 'category'
fg_plot = FacetGridPlot(data=df, col='category', height=4, aspect=1.2)

# 3. Map a scatter plot to each facet
fg_plot.grid.map(plt.scatter, 'x_data', 'y_data', alpha=0.7, s=50, edgecolor='w')

# 4. Set titles and labels
fg_plot.set_titles(col_template="Category: {col_name}")
fg_plot.grid.set_xlabels("X-axis Data")
fg_plot.grid.set_ylabels("Y-axis Data")
fg_plot.fig.suptitle("Scatter Plot across Categories", y=1.02) # y adjusts title position

plt.tight_layout()
plt.show()
```

### Expected Output

You will see a figure with three subplots arranged horizontally. Each subplot will correspond to one category ('Group A', 'Group B', 'Group C') and will contain a scatter plot showing the relationship between `x_data` and `y_data` for that specific category. The titles above each subplot will clearly indicate the category name.

### Key Concepts

-   **`FacetGridPlot`**: The class used to orchestrate the creation of multi-panel plots.
-   **`col='category'`**: This argument tells `FacetGridPlot` to create a separate column of plots for each unique value in the 'category' column of your DataFrame. You could also use `row='category'` for rows, or both `row` and `col` for a grid.
-   **`height` and `aspect`**: Control the size and aspect ratio of individual facets.
-   **`fg_plot.grid.map(plt.scatter, 'x_data', 'y_data')`**: This is how you apply a plotting function (here, Matplotlib's `scatter`) to each facet. The strings 'x\_data' and 'y\_data' refer to column names in the input DataFrame.
-   **`fg_plot.set_titles()`**: Used to customize the titles of individual facets.

## Example 2: Facet Grid with `row`, `col`, `hue`, and `col_wrap`

This example demonstrates a more complex facet grid layout, using `row` and `col` for grid layout, `hue` for color-coding within each subplot, and `col_wrap` to arrange the columns onto multiple rows if there are too many.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from monet_plots.plots.facet_grid import FacetGridPlot

# 1. Prepare sample data with two categorical variables and a hue variable
np.random.seed(29) # for reproducibility
n_obs = 50
scenarios = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4']
times = ['Morning', 'Evening']
conditions = ['Low', 'High']

data_list = []
for scenario in scenarios:
    for time in times:
        for cond in conditions:
            # Simulate data with distinct characteristics for each combination
            base_val = np.random.uniform(0, 10)
            x = np.random.normal(loc=base_val, scale=2, size=n_obs)
            y = 0.8 * x + np.random.normal(loc=0, scale=1, size=n_obs) + (5 if cond == 'High' else 0)

            temp_df = pd.DataFrame({
                'feature_x': x,
                'feature_y': y,
                'scenario': scenario,
                'time_of_day': time,
                'condition': cond
            })
            data_list.append(temp_df)

df_complex = pd.concat(data_list).reset_index(drop=True)

# 2. Initialize FacetGridPlot with row, col, hue, and col_wrap
# Plots will be arranged by 'scenario' in columns, 'time_of_day' in rows,
# and colored by 'condition'. 'col_wrap=2' will wrap columns after every 2 scenarios.
fg_plot_complex = FacetGridPlot(data=df_complex,
                   row='time_of_day',
                   col='scenario',
                   hue='condition',
                   col_wrap=2, # Wrap columns after 2 scenarios
                   height=3.5,
                   aspect=1.3
                   )

# 3. Map a seaborn kdeplot (Kernel Density Estimate) to each facet
fg_plot_complex.grid.map_dataframe(sns.kdeplot, x='feature_x', y='feature_y', fill=True, alpha=0.5, cbar=True)

# 4. Set titles, labels, and add a legend
fg_plot_complex.set_titles(row_template="{row_name}", col_template="{col_name}")
fg_plot_complex.grid.set_xlabels("Feature X Value")
fg_plot_complex.grid.set_ylabels("Feature Y Value")
fg_plot_complex.grid.add_legend(title="Condition")
fg_plot_complex.fig.suptitle("Relationship by Scenario, Time, and Condition", y=1.02)

plt.tight_layout()
plt.show()
```

### Expected Output

A grid of plots will appear, arranged in rows by 'time\_of\_day' (Morning, Evening) and up to two columns by 'scenario'. Within each subplot, you will see kernel density estimates for 'feature\_x' and 'feature\_y', with colors distinguishing 'Low' and 'High' conditions. A color bar might also be present for the KDE. This complex layout efficiently displays conditional relationships across multiple categorical variables.

### Key Concepts

-   **`row` and `col`**: Used together, these arguments create a grid with rows defined by one variable and columns by another.
-   **`hue`**: Adds another layer of categorization by coloring plot elements within each facet based on the 'condition' variable. Automatically generates a legend.
-   **`col_wrap`**: Specifies the maximum number of columns in the facet grid, wrapping to the next row if exceeded. Very useful for managing layouts with many `col` levels.
-   **`fg_plot.grid.map_dataframe()`**: A versatile method to map plotting functions that expect a DataFrame and column names as arguments.
