import matplotlib.pyplot as plt

# Define individual style dictionaries
_wiley_style = {
    # Font settings
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    # Axes settings
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.color": "gray",
    # Line settings
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    # Legend settings
    "legend.fontsize": 9,
    "legend.frameon": False,
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
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
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
    "axes.spines": ["bottom", "left"],
    "axes.spines.right": False,
    "axes.spines.top": False,
}

_web_style = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
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

# Dictionary to map context names to style dictionaries
_styles = {
    "wiley": _wiley_style,
    "presentation": _presentation_style,
    "paper": _paper_style,
    "web": _web_style,
    "default": {},  # Matplotlib default style
}


def set_style(context="default"):
    """
    Set the plotting style based on a predefined context.

    Parameters
    ----------
    context : str, optional
        The name of the style context to apply.
        Available contexts: "wiley", "presentation", "paper", "web", "default".
        Defaults to "default" (Matplotlib's default style).

    Raises
    ------
    ValueError
        If an unknown context name is provided.
    """
    if context not in _styles:
        raise ValueError(f"Unknown style context: '{context}'. " f"Available contexts are: {', '.join(_styles.keys())}")

    plt.style.use(_styles[context])


# Expose wiley_style for direct import if needed for backward compatibility
wiley_style = _wiley_style
