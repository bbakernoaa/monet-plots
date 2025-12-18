
# monet_plots/plots/xarray_integration_spec.md

## Objective

Refactor plotting classes to natively support `xarray.Dataset` and `xarray.DataArray` inputs, removing the mandatory conversion to `pandas.DataFrame`. This will simplify the data pipeline, improve performance, and leverage xarray's rich metadata for more intuitive plotting.

### Key Benefits

-   **Performance Boost**: Eliminates the overhead of converting large xarray objects to pandas DataFrames.
-   **Enhanced Usability**: Allows users to work directly with their preferred data format.
-   **Metadata Utilization**: Preserves valuable metadata from xarray objects, which can be used for automatic labeling of plots.

---

## Phase 1: Core Utility and Pilot Refactoring

### 1. Data Normalization Utility

A new utility function, `_normalize_data`, will be introduced in [`src/monet_plots/plot_utils.py`](src/monet_plots/plot_utils.py) to handle various data types gracefully.

```python
# In src/monet_plots/plot_utils.py

import pandas as pd
import xarray as xr

def _normalize_data(data):
    """
    Inspects the input data and prepares it for plotting.

    If the input is a pandas DataFrame, it is returned as is.
    If the input is an xarray DataArray, it is converted to a Dataset.
    If the input is an xarray Dataset, it is returned as is.

    This ensures that downstream plotting functions can expect a consistent
    data structure (either a DataFrame or a Dataset).

    TDD Anchor:
    - test_normalize_dataframe_passthrough: Verify a DataFrame is returned unchanged.
    - test_normalize_dataarray_to_dataset: Ensure a DataArray is converted to a Dataset.
    - test_normalize_dataset_passthrough: Check that a Dataset is returned unchanged.
    - test_normalize_invalid_type: Confirm a TypeError is raised for unsupported types.
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, xr.DataArray):
        return data.to_dataset()
    elif isinstance(data, xr.Dataset):
        return data
    else:
        raise TypeError("Unsupported data type. Please provide a pandas DataFrame or an xarray object.")

```

### 2. Refactor `TimeSeriesPlot`

The [`TimeSeriesPlot`](src/monet_plots/plots/timeseries.py) class will be updated to use `_normalize_data` and handle both pandas and xarray inputs.

```python
# In src/monet_plots/plots/timeseries.py

from monet_plots.plot_utils import _normalize_data
from monet_plots.plots.base import BasePlot

class TimeSeriesPlot(BasePlot):
    """A class for creating time series plots."""

    def __init__(self, data, **kwargs):
        """
        Initializes the plot with data, which can be a pandas DataFrame
        or an xarray Dataset/DataArray.
        """
        self.data = _normalize_data(data)
        # ... other initializations

    def _get_plot_data(self, variables):
        """
        Retrieves specified variables from the stored data structure.
        This internal method will handle data access for both pandas and xarray.
        """
        # TDD Anchor:
        # - test_get_plot_data_from_dataframe: Verify data retrieval from a DataFrame.
        # - test_get_plot_data_from_dataset: Ensure correct data slicing from an xarray Dataset.
        if isinstance(self.data, pd.DataFrame):
            return self.data[variables]
        else: # xarray.Dataset
            return self.data[variables]

    def plot(self, variables, **kwargs):
        """
        Generates the time series plot.
        The core plotting logic will now be able to handle xarray objects directly.
        """
        plot_data = self._get_plot_data(variables)

        # The plotting library (e.g., matplotlib, seaborn) might have different
        # calling conventions for pandas vs. xarray. This is where the
        # adaptation happens.
        if isinstance(plot_data, pd.DataFrame):
            # Existing pandas-based plotting logic
            # e.g., ax.plot(plot_data.index, plot_data[var], ...)
            pass
        else: # xarray.Dataset
            # New xarray-based plotting logic
            # e.g., plot_data[var].plot(ax=ax, ...)
            # This leverages xarray's built-in plotting capabilities.
            pass

        # TDD Anchor:
        # - test_plot_with_xarray_input: Create a plot with an xarray Dataset and verify the output.
        # - test_plot_with_pandas_input: Ensure backward compatibility by plotting with a DataFrame.

```

---

## Phase 2: Broader Implementation and Testing

### 3. Update Other Plotting Classes

All other plotting classes in [`src/monet_plots/plots/`](src/monet_plots/plots/) will be refactored following the pattern established with `TimeSeriesPlot`. This includes, but is not limited to:

-   [`ScatterPlot`](src/monet_plots/plots/scatter.py)
-   [`SpatialPlot`](src/monet_plots/plots/spatial.py)
-   [`ProfilePlot`](src/monet_plots/plots/profile.py)

Each class will incorporate the `_normalize_data` utility and adapt its internal logic to handle both pandas and xarray objects.

### 4. Comprehensive Testing Strategy

A dedicated test file, [`tests/plots/test_xarray_integration.py`](tests/plots/test_xarray_integration.py), will be created to house all tests related to this refactoring.

```python
# In tests/plots/test_xarray_integration.py

import pytest
import pandas as pd
import xarray as xr
import numpy as np

from monet_plots.plots.timeseries import TimeSeriesPlot
from monet_plots.plot_utils import _normalize_data

# TDD: Test the normalization utility
def test_normalize_dataframe_passthrough():
    df = pd.DataFrame({'a': [1, 2]})
    assert _normalize_data(df) is df

def test_normalize_dataarray_to_dataset():
    da = xr.DataArray(np.random.rand(3), name='test_var')
    ds = _normalize_data(da)
    assert isinstance(ds, xr.Dataset)
    assert 'test_var' in ds.data_vars

# TDD: Test the refactored TimeSeriesPlot
@pytest.fixture
def sample_xarray_dataset():
    # Create a sample xarray Dataset for testing
    time = pd.to_datetime(['2023-01-01', '2023-01-02'])
    data = np.random.rand(2, 3)
    return xr.Dataset(
        {'temperature': (('time', 'location'), data)},
        coords={'time': time, 'location': ['A', 'B', 'C']}
    )

def test_timeseries_plot_with_xarray(sample_xarray_dataset):
    """
    Verify that TimeSeriesPlot can be instantiated and can plot
    directly from an xarray Dataset.
    """
    plot = TimeSeriesPlot(sample_xarray_dataset)
    # This test would check if the plot object is created correctly
    # and if the plot() method executes without errors.
    # More advanced checks could involve inspecting the plot artifacts.
    assert isinstance(plot.data, xr.Dataset)
    # Further implementation would mock the plotting backend to verify calls.

def test_backward_compatibility_with_pandas():
    """
    Ensure that the refactored class still works perfectly with pandas DataFrames.
    """
    df = pd.DataFrame({'temperature': [10, 12]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
    plot = TimeSeriesPlot(df)
    assert isinstance(plot.data, pd.DataFrame)
    # Add plotting assertions here.

```
