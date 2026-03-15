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
        df_here = df.data.loc[
            (df.alt > threshold_list[i - 1]) & (df.alt <= threshold_list[i])
        ]
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


def calc_24hr_ave(df, col=None):
    """Calculates 24-hour averages for regulatory analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing 'siteid' and 'time_local' columns.
    col : str
        Column name to average.

    Returns
    -------
    pandas.DataFrame
        Averaged data merged back to original sites and daily times.
    """
    df = df.copy()
    if "time_local" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["time_local"] = df.index
        else:
            raise ValueError("Data must contain 'time_local' column or DatetimeIndex")

    df.index = df.time_local
    # select sites with nobs >=18, 75% completeness
    df_24hr_ave = (
        (
            df.groupby("siteid")[col].resample("D").sum(min_count=18, numeric_only=True)
            / df.groupby("siteid")[col].resample("D").count()
        )
        .reset_index()
        .dropna()
    )
    df_24hr_ave = df_24hr_ave.rename(columns={col: f"{col}_reg"})
    df = df.reset_index(drop=True)
    return df.merge(df_24hr_ave, on=["siteid", "time_local"])


def calc_8hr_rolling_max(df, col=None, window=8):
    """Calculates 8-hour rolling average daily maximum.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing 'siteid' and 'time_local' columns.
    col : str
        Column name to calculate.
    window : int
        Rolling window size in hours, by default 8.

    Returns
    -------
    pandas.DataFrame
        Rolling max data merged back to original sites and daily times.
    """
    df = df.copy()
    if "time_local" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["time_local"] = df.index
        else:
            raise ValueError("Data must contain 'time_local' column or DatetimeIndex")

    df.index = df.time_local
    df_rolling = (
        df.groupby("siteid")[col]
        .rolling(window, min_periods=6, center=True, win_type="boxcar")
        .mean(numeric_only=True)
        .reset_index()
        .dropna()
    )
    # select sites with nobs >=18, 75% completeness
    df_rolling.index = df_rolling.time_local
    df_rolling_max = (
        df_rolling.groupby("siteid")
        .resample("D")
        .max(min_count=18, numeric_only=True)
        .reset_index()
        .dropna()
    )
    df_rolling_max = df_rolling_max.rename(columns={col: f"{col}_reg"})
    df = df.reset_index(drop=True)
    return df.merge(df_rolling_max, on=["siteid", "time_local"])
