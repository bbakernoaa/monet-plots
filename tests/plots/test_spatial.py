import pytest
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from monet_plots.plots.spatial import SpatialPlot, SpatialTrack


@pytest.fixture
def clear_figures():
    """Clear all existing figures before and after a test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def sample_da():
    """Create a sample DataArray for testing."""
    return xr.DataArray(
        np.random.rand(10, 10),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.arange(30, 40),
            "longitude": np.arange(-100, -90),
        },
    )


def test_spatial_plot_init(clear_figures, sample_da):
    """Test SpatialPlot initialization."""
    plot = SpatialPlot()
    assert plot is not None


def test_spatial_plot_plot(clear_figures, sample_da):
    """Test SpatialPlot plot method."""
    plot = SpatialPlot()
    # ax = plot.plot()  # SpatialPlot has no plot method
    # assert ax is not None
    assert plot.ax is not None


def test_SpatialTrack_plot(clear_figures):
    """Test SpatialTrack plot method."""
    lon = np.linspace(-120, -80, 10)
    lat = np.linspace(30, 40, 10)
    data = np.random.rand(10)
    plot = SpatialTrack(lon, lat, data)
    plot.plot()


def test_spatial_plot_draw_features_data_driven(clear_figures):
    """Test the data-driven feature drawing mechanism.

    This test validates that when keyword arguments (e.g., `states=True`)
    are passed to the SpatialPlot constructor, the corresponding cartopy
    features are correctly added to the GeoAxes object.
    """
    # 1. The Logic (Implementation)
    # Instantiate the plot with several features enabled.
    plot = SpatialPlot(
        states=True,
        coastlines=True,
        countries=True,
        land=True,
        ocean=True,
        resolution="110m",
    )
    initial_collections = len(plot.ax.collections)

    # 2. The Proof (Validation)
    # The add_features method is called to render the map elements.
    plot.add_features()
    final_collections = len(plot.ax.collections)

    # Assert that the number of collections on the axes has increased,
    # which confirms that cartopy features have been added. The exact number
    # can vary, so we check for a significant increase.
    assert final_collections > initial_collections
    assert final_collections >= 5  # Expect at least 5 features to be added


def test_spatial_plot_feature_styling(clear_figures):
    """Test that feature styling kwargs are correctly applied.

    This test ensures that when a style dictionary is passed for a feature
    (e.g., `states={"linewidth": 2, "edgecolor": "red"}`), the style is
    applied to the corresponding cartopy feature.
    """
    # 1. The Logic (Implementation)
    # Define a custom style for the 'states' feature.
    custom_style = {"linewidth": 2, "edgecolor": "red"}
    plot = SpatialPlot(states=custom_style, resolution="110m")

    # 2. The Proof (Validation)
    # Draw the features to apply the styling.
    plot.add_features()

    # The most reliable way to check is to inspect the LineCollection
    # created by the feature. We look for one with our custom style.
    found_match = False
    for collection in plot.ax.collections:
        # Check if the collection's properties match our custom style.
        # This is a bit of an internal detail, but it's a robust check.
        if (
            collection.get_linewidth()[0] == custom_style["linewidth"]
            and collection.get_edgecolor()[0][0] == 1.0
        ):  # Red channel
            found_match = True
            break

    assert found_match, "Failed to find a feature with the specified custom style."


def test_spatial_plot_draw_map_docstring_example(clear_figures):
    """Test the example from the SpatialPlot.draw_map docstring.

    This ensures the documented example works as expected.
    """
    # 1. The Logic (Implementation from Docstring)
    from cartopy.mpl.geoaxes import GeoAxes

    ax = SpatialPlot.draw_map(states=True, extent=[-125, -70, 25, 50])

    # 2. The Proof (Validation)
    # Assert that the axes is a GeoAxes object
    assert isinstance(ax, GeoAxes)

    # Assert that the 'states' feature was added.
    # We check that at least one collection (the states) has been added.
    assert len(ax.collections) > 0

    # 3. The UI (Visualization)
    # The plot is implicitly created and would be shown with plt.show()
    # No need to save it in a test.


def test_spatialplot_create_map(clear_figures):
    """Test the SpatialPlot.create_map factory method.

    This test ensures that the new, recommended factory method correctly
    creates a SpatialPlot instance and adds the specified features.
    """
    # 1. The Logic (Implementation)
    # Use the factory to create a plot with states and an extent.
    plot = SpatialPlot.create_map(states=True, extent=[-125, -70, 25, 50])

    # 2. The Proof (Validation)
    # Assert that the returned object is a SpatialPlot instance.
    assert isinstance(plot, SpatialPlot)

    # Assert that the underlying axes is a GeoAxes object.
    from cartopy.mpl.geoaxes import GeoAxes
    assert isinstance(plot.ax, GeoAxes)

    # Assert that the 'states' feature was added.
    # The number of collections should be greater than zero.
    assert len(plot.ax.collections) > 0
