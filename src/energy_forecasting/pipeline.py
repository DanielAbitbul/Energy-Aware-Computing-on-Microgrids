import pandas as pd
from windpowerlib import WindTurbine

from energy_forecasting.wind import forecast_wind_power
from energy_forecasting.energy import power_to_energy
from energy_forecasting.weather.dmi_latest_forecast import OUTPUT_FILE, load_weather_dataframe


def run_energy_forecast(
    weather: pd.DataFrame | None = None,
    turbine: WindTurbine | None = None,
    interval_minutes: int = 60,
) -> pd.Series:
    """Forecast wind energy production in Wh per interval."""

    if weather is None:
        weather = load_weather_dataframe(OUTPUT_FILE)

    power = forecast_wind_power(
        weather=weather,
        turbine=turbine,
    )

    energy = power_to_energy(
        power=power,
        interval_minutes=interval_minutes,
    )

    energy.name = "wind_energy_wh"

    return energy