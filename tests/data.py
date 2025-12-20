
import pooch
import xarray as xr


def _get_weather_data():
    """Download and return the path to the xarray weather data."""
    return pooch.retrieve(
        url="https://github.com/pydata/xarray-data/raw/master/air_temperature.nc",
        known_hash="a0d81652c0651123122c09a03ce0324a2a632b31b3b78edc6d04e5213372a693",
    )


def _get_roms_data():
    """Download and return the path to the ROMS ocean data."""
    return pooch.retrieve(
        url="https://github.com/pydata/xarray-data/raw/master/ocean_model.nc",
        known_hash="a9b233e734b6c618533f343365f157704b8c8c8d6ac750e2b34f5e30b0a811a3",
    )


def open_weather_data():
    """Open and return the xarray weather dataset."""
    return xr.open_dataset(_get_weather_data())


def open_roms_data():
    """Open and return the ROMS ocean dataset."""
    return xr.open_dataset(_get_roms_data())
