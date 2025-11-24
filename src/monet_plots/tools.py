# src/monet_plots/tools.py
import numpy as np

def wsdir2uv(ws, wdir):
    """Converts wind speed and direction to u and v components.

    Args:
        ws (numpy.ndarray): The wind speed.
        wdir (numpy.ndarray): The wind direction.

    Returns:
        tuple: A tuple containing the u and v components of the wind.
    """
    rad = np.pi / 180.
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
    wdir = 180 + (180/np.pi) * np.arctan2(u, v)
    return ws, wdir