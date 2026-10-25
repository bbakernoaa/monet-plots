"""Native sounding display using a skewed Matplotlib temperature axis."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from matplotlib.transforms import Affine2D
from scipy.integrate import solve_ivp

from .base import BasePlot


class SkewTAxes(Axes):
    name = "monet_skewt"

    def __init__(self, *args, rotation=30, **kwargs):
        self.rotation = rotation
        super().__init__(*args, **kwargs)

    def _set_lim_and_transforms(self):
        super()._set_lim_and_transforms()
        self.transData = (
            self.transData
            + self.transAxes.inverted()
            + Affine2D().skew_deg(self.rotation, 0)
            + self.transAxes
        )


register_projection(SkewTAxes)


def _units(values, explicit, default, kind):
    unit = (explicit or getattr(values, "attrs", {}).get("units", default)).lower()
    array = np.asarray(values, dtype=float)
    if kind == "pressure":
        if unit == "pa":
            return array / 100
        if unit in {"hpa", "mb", "mbar"}:
            return array
    elif kind == "temperature":
        if unit in {"k", "kelvin"}:
            return array - 273.15
        if unit in {"c", "degc", "°c", "celsius"}:
            return array
    elif kind == "wind":
        if unit in {"knots", "kt", "knot"}:
            return array
        if unit in {"m/s", "m s-1"}:
            return array * 1.943844
    raise ValueError(f"Unsupported {kind} units: {unit}")


def _moist(pressure, temperature):
    """Liquid-water pseudoadiabat, pressure in hPa and temperature in K."""

    def derivative(p, t):
        es = 6.112 * np.exp(17.67 * (t[0] - 273.15) / (t[0] - 29.65))
        ratio = 0.622 * es / (p - es)
        return [
            (287.05 * t[0] + 2.5e6 * ratio)
            / (p * (1004.0 + 2.5e6**2 * ratio * 0.622 / (287.05 * t[0] ** 2)))
        ]

    return solve_ivp(
        derivative,
        (pressure[0], pressure[-1]),
        [temperature],
        t_eval=pressure,
        rtol=1e-7,
        atol=1e-7,
    ).y[0]


class SkewTPlot(BasePlot):
    """Plot soundings. Unlabeled arrays default to hPa, Celsius, and m/s."""

    def __init__(self, data=None, *, fig=None, ax=None, rotation=30, **kwargs):
        if ax is not None and not isinstance(ax, SkewTAxes):
            raise ValueError(
                "Use subplot_kw={'projection': 'monet_skewt'} for supplied axes"
            )
        if ax is None:
            import matplotlib.pyplot as plt

            fig = fig if fig is not None else plt.figure()
            ax = fig.add_subplot(projection="monet_skewt", rotation=rotation)
        super().__init__(fig=fig, ax=ax, **kwargs)
        self.data = data
        self.overlays = []
        self.ax.set_yscale("log")
        self.ax.set_ylim(1050, 100)
        self.ax.set_xlim(-40, 50)
        self.ax.set_xlabel("Temperature (°C)")
        self.ax.set_ylabel("Pressure (hPa)")
        p = np.geomspace(1050, 100, 150)
        for t in np.arange(240, 441, 20):
            self.ax.plot(
                t * (p / 1000) ** (287.05 / 1004) - 273.15,
                p,
                color="orange",
                alpha=0.25,
                linewidth=0.6,
            )
        for t in np.arange(273.15, 314, 10):
            self.ax.plot(
                _moist(p, t) - 273.15, p, color="green", alpha=0.25, linewidth=0.6
            )
        self.ax.grid(alpha=0.3)

    def plot(
        self,
        pressure,
        temperature,
        dewpoint=None,
        *,
        u=None,
        v=None,
        label=None,
        parcel=False,
        pressure_units=None,
        temperature_units=None,
        wind_units=None,
        **kwargs,
    ):
        def resolve(value):
            return self.data[value] if isinstance(value, str) else value

        p = _units(resolve(pressure), pressure_units, "hpa", "pressure")
        t = _units(resolve(temperature), temperature_units, "degc", "temperature")
        if (
            p.ndim != 1
            or t.shape != p.shape
            or not np.all(np.isfinite(p) & (p > 0))
            or len(np.unique(p)) != len(p)
        ):
            raise ValueError(
                "Soundings require matching 1D values and unique finite positive pressures"
            )
        order = np.argsort(-p)
        p, t = p[order], t[order]
        self.overlays.extend(self.ax.plot(t, p, label=label, **kwargs))
        td = None
        if dewpoint is not None:
            td = _units(resolve(dewpoint), temperature_units, "degc", "temperature")
            if td.shape != p.shape:
                raise ValueError("dewpoint must match pressure")
            td = td[order]
            self.overlays.extend(
                self.ax.plot(
                    td, p, linestyle="--", label=f"{label or 'Sounding'} dewpoint"
                )
            )
        if (u is None) != (v is None):
            raise ValueError("Both wind components are required")
        if u is not None:
            winds = [_units(resolve(a), wind_units, "m/s", "wind") for a in (u, v)]
            if any(a.shape != p.shape for a in winds):
                raise ValueError("Winds must match pressure")
            self.overlays.append(
                self.ax.barbs(
                    np.full(p.shape, 0.96),
                    p,
                    winds[0][order],
                    winds[1][order],
                    transform=self.ax.get_yaxis_transform(),
                )
            )
        if parcel:
            if td is None:
                raise ValueError("Parcel ascent requires dewpoint")
            valid = np.flatnonzero(np.isfinite(t) & np.isfinite(td))
            if not len(valid):
                raise ValueError("No valid parcel starting level")
            i = valid[0]
            tk, dk = t[i] + 273.15, td[i] + 273.15
            if dk > tk:
                raise ValueError("Parcel starting dewpoint exceeds temperature")
            tl = 1 / (1 / (dk - 56) + np.log(tk / dk) / 800) + 56
            pl = p[i] * (tl / tk) ** (1004 / 287.05)
            dry = np.geomspace(p[i], pl, 50)
            moist = np.geomspace(pl, min(100, pl * 0.99), 100)
            self.overlays.extend(
                self.ax.plot(
                    np.r_[tk * (dry / p[i]) ** (287.05 / 1004), _moist(moist, tl)]
                    - 273.15,
                    np.r_[dry, moist],
                    color="black",
                )
            )
            self.overlays.extend(self.ax.plot(tl - 273.15, pl, "ko"))
        if label:
            self.ax.legend()
        return self.ax
