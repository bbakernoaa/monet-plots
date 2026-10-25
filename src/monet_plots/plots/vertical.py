"""Shared coordinate helpers for vertical plots."""


def configure_vertical_axis(ax, coordinate, mode="auto", yscale=None):
    """Apply conventional pressure or height axis orientation."""
    if mode not in {"auto", "height", "pressure"}:
        raise ValueError("vertical_axis must be 'auto', 'height', or 'pressure'")
    units = getattr(coordinate, "attrs", {}).get("units", "").lower()
    name = getattr(coordinate, "name", "") or ""
    pressure = mode == "pressure" or (
        mode == "auto"
        and ("pressure" in name.lower() or units in {"pa", "hpa", "mb", "mbar"})
    )
    if mode != "auto" or pressure:
        if pressure != ax.yaxis_inverted():
            ax.invert_yaxis()
    if yscale:
        ax.set_yscale(yscale)
    return ax
