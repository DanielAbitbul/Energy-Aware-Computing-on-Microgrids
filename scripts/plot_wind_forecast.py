import matplotlib.pyplot as plt

from energy_forecasting.weather.mock import generate_mock_weather
from energy_forecasting.wind import forecast_wind_power
from energy_forecasting.config import create_skystream_turbine


weather = generate_mock_weather(
    start="2026-06-01",
    periods=168,
    seed=42,
)

turbine = create_skystream_turbine()
forecast = forecast_wind_power(weather, turbine)

plt.figure(figsize=(12, 5))
plt.plot(forecast.index, forecast / 1000)

plt.title("Skystream 3.7 — Simulated Wind Power Forecast")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.grid(True)
plt.tight_layout()
plt.show()