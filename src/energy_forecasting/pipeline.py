import pandas as pd
from windpowerlib import WindTurbine

from energy_forecasting.wind import forecast_wind_power
from energy_forecasting.energy import power_to_energy


def run_energy_forecast(
    weather: pd.DataFrame,
    turbine: WindTurbine,
    interval_minutes: int = 60,
) -> pd.Series:
    """Forecast wind energy production in Wh per interval."""

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