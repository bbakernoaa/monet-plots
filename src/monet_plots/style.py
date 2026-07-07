from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

# Colorblind-friendly palette (Okabe-Ito)
CB_COLORS = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
]

# Visually distinct markers
CB_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]

# Define individual style dictionaries
_wiley_style = {
    # Font settings
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Roboto",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
        "sans-serif",
    ],
    "font.size": 10,
    # Axes settings
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linestyle": ":",
    "grid.color": "lightgray",
    "grid.alpha": 0.7,
    # Tick settings
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Line settings
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    # Legend settings
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.borderpad": 0.4,
    # Figure settings
    "figure.figsize": (6, 4),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "tiff",
    "savefig.bbox": "tight",
}

_presentation_style = {
    "figure.figsize": (12, 8),
    "figure.dpi": 100,
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.size": 14,
    "axes.grid": False,
    "savefig.dpi": 150,
    "savefig.format": "png",
    "savefig.bbox": "tight",
}

_paper_style = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Roboto",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
        "sans-serif",
    ],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (8, 6),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "tiff",
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.color": "lightgray",
    "axes.spines.right": False,
    "axes.spines.top": False,
}

_web_style = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Roboto",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
        "sans-serif",
    ],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.color": "#e0e0e0",
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
}

_pivotal_weather_style = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial Narrow",
        "Roboto Condensed",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "sans-serif",
    ],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.grid": False,
    "axes.edgecolor": "#2f3842",
    "axes.linewidth": 0.8,
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "figure.figsize": (11, 8),
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "coastlines": True,
    "coastline.width": 0.7,
    "coastline.color": "#2f3842",
    "states": {"linewidth": 0.35, "edgecolor": "#69737d", "facecolor": "none"},
    "states.width": 0.35,
    "states.color": "#69737d",
    "borders": {"linewidth": 0.6, "edgecolor": "#3f4852", "facecolor": "none"},
    "borders.width": 0.6,
    "borders.color": "#3f4852",
    "counties": False,
    "natural_earth": True,
    "land": {"facecolor": "#f5f4ef", "edgecolor": "none"},
    "ocean": {"facecolor": "#d7e7f5", "edgecolor": "none"},
    "lakes": {"facecolor": "#d7e7f5", "edgecolor": "none"},
    "rivers": {"edgecolor": "#a7c7e1", "linewidth": 0.25},
    "gridlines": False,
    "resolution": "50m",
    "cbar.orientation": "horizontal",
    "cbar.location": "bottom",
    "cbar.size": "3.5%",
    "cbar.pad": 0.08,
    "cbar.labelsize": 10,
    "cbar.tick.labelsize": 9,
}

_weathermesh_style = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Inter",
        "Roboto",
        "Arial",
        "Helvetica",
        "sans-serif",
    ],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 16,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.bottom": False,
    "ytick.left": False,
    "figure.figsize": (10, 8),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


# Dictionary to map context names to style dictionaries
_styles = {
    "wiley": _wiley_style,
    "presentation": _presentation_style,
    "paper": _paper_style,
    "web": _web_style,
    "pivotal_weather": _pivotal_weather_style,
    "weathermesh": _weathermesh_style,
    "default": {},  # Matplotlib default style
}

_current_style_name = "wiley"


def get_available_styles() -> list[str]:
    """
    Returns a list of available style context names.

    Returns
    -------
    list[str]
        List of style names.
    """
    return list(_styles.keys())


def set_style(context: str = "wiley"):
    """
    Set the plotting style based on a predefined context.

    Parameters
    ----------
    context : str, optional
        The name of the style context to apply.
        Available contexts: "wiley", "presentation", "paper", "web", "pivotal_weather", "default".
        Defaults to "wiley".

    Raises
    ------
    ValueError
        If an unknown context name is provided.
    """
    global _current_style_name

    if context not in _styles:
        raise ValueError(
            f"Unknown style context: '{context}'. "
            f"Available contexts are: {', '.join(_styles.keys())}"
        )

    style_dict = _styles[context]

    # Separate standard rcParams from custom ones
    standard_rc = {k: v for k, v in style_dict.items() if k in plt.rcParams}

    if context == "default":
        plt.style.use("default")
    else:
        plt.style.use(standard_rc)

    _current_style_name = context


def get_style_setting(key: str, default: Any = None) -> Any:
    """
    Retrieves a style setting from the currently active style.
    Looks in both standard rcParams and custom style settings.

    Parameters
    ----------
    key : str
        The name of the style setting.
    default : Any, optional
        The default value if the key is not found, by default None.

    Returns
    -------
    Any
        The style setting value.
    """
    # First check current style's dictionary (includes custom keys)
    style_dict = _styles.get(_current_style_name, {})
    if key in style_dict:
        return style_dict[key]

    # Fallback to general rcParams
    return plt.rcParams.get(key, default)


# Expose wiley_style for direct import if needed for backward compatibility
wiley_style = _wiley_style
