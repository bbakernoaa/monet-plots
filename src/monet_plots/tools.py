import numpy as np
import pandas as pd

from .plot_utils import compute


def split_by_threshold(data_list, alt_list, threshold_list):
    """
    Splits data into bins based on altitude thresholds.

    Args:
        data_list (list): List of data values.
        alt_list (list): List of altitude values corresponding to the data.
        threshold_list (list): List of altitude thresholds to bin the data.

    Returns:
        list: A list of arrays, where each array contains the data values
              within an altitude bin.
    """
    df = pd.DataFrame(data={"data": data_list, "alt": alt_list})
    output_list = []
    for i in range(1, len(threshold_list)):
        df_here = df.data.loc[
            (df.alt > threshold_list[i - 1]) & (df.alt <= threshold_list[i])
        ]
        output_list.append(df_here.values)
    return output_list


def wind_grid_slice(shape, skip=None):
    """Keep small wind grids intact and limit dense grids to 16 samples per axis."""
    if skip is not None:
        if not isinstance(skip, (int, np.integer)) or skip < 1:
            raise ValueError("skip must be a positive integer.")
        return (slice(None, None, skip),) * 2
    return tuple(slice(None, None, max(1, (size + 15) // 16)) for size in shape)


def sample_wind_grid(ws, wdir, grid, skip):
    """Load only the coordinates and wind vectors selected for display."""
    shape = np.broadcast_shapes(ws.shape, wdir.shape)
    selection = wind_grid_slice(shape, skip)
    wind = [
        (np.broadcast_to(data, shape) if data.shape != shape else data)[selection]
        for data in (ws, wdir)
    ]
    lon = grid.variables["LON"][0, 0, *selection]
    lat = grid.variables["LAT"][0, 0, *selection]
    ws, wdir, lon, lat = compute(*wind, lon, lat)
    u, v = wsdir2uv(np.asarray(ws), np.asarray(wdir))
    return lon, lat, u, v


def wsdir2uv(ws, wdir):
    """Converts wind speed and direction to u and v components.

    Args:
        ws (numpy.ndarray): The wind speed.
        wdir (numpy.ndarray): The wind direction.

    Returns:
        tuple: A tuple containing the u and v components of the wind.
    """
    rad = np.pi / 180.0
    u = -ws * np.sin(wdir * rad)
    v = -ws * np.cos(wdir * rad)
    return u, v


def uv2wsdir(u, v):
    """Converts u and v components to wind speed and direction.

    Args:
        u (numpy.ndarray): The u component of the wind.
        v (numpy.ndarray): The v component of the wind.

    Returns:
        tuple: A tuple containing the wind speed and direction.
    """
    ws = np.sqrt(u**2 + v**2)
    wdir = 180 + (180 / np.pi) * np.arctan2(u, v)
    return ws, wdir
