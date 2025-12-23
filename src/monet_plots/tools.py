import numpy as np
import pandas as pd


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
        df_here = df.data.loc[(df.alt > threshold_list[i - 1]) & (df.alt <= threshold_list[i])]
        output_list.append(df_here.values)
    return output_list


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
