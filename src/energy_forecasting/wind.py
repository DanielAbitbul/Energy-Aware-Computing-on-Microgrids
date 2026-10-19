import pandas as pd
from windpowerlib import ModelChain, WindTurbine

from energy_forecasting.config import create_wind_turbine


def forecast_wind_power(
    weather: pd.DataFrame,
    turbine: WindTurbine | None = None,
) -> pd.Series:
    if turbine is None:
        turbine = create_wind_turbine()

    model_chain = ModelChain(
        turbine,
        wind_speed_model="logarithmic",
        density_model="ideal_gas",
        power_output_model="power_curve",
    )

    model_chain.run_model(weather)

    return model_chain.power_output