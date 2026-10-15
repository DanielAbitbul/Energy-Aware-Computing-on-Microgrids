
import pandas as pd
import pytest

from energy_forecasting.weather.mock import generate_mock_weather
from energy_forecasting.wind import forecast_wind_power
from energy_forecasting.config import create_skystream_turbine
from energy_forecasting.energy import power_to_energy


def test_mock_weather_generation():
    weather = generate_mock_weather(
        start="2026-06-01",
        periods=24,
        seed=42,
    )

    # Exactly 24 hourly observations
    assert len(weather) == 24

    # Datetime index with timezone
    assert isinstance(weather.index, pd.DatetimeIndex)
    assert weather.index.tz is not None

    # Required windpowerlib weather variables
    assert ("wind_speed", 10) in weather.columns
    assert ("temperature", 2) in weather.columns
    assert ("pressure", 0) in weather.columns

    # Wind speeds cannot be negative
    assert (weather[("wind_speed", 10)] >= 0).all()

    # Same seed should produce identical data
    repeated = generate_mock_weather(
        start="2026-06-01",
        periods=24,
        seed=42,
    )

    pd.testing.assert_frame_equal(weather, repeated)


def test_wind_power_forecast():
    weather = generate_mock_weather(
        start="2026-06-01",
        periods=24,
        seed=42,
    )

    forecast = forecast_wind_power(weather)

    assert len(forecast) == 24
    assert (forecast >= 0).all()
    assert (forecast <= 5000).all()


def test_wind_power_at_known_speed():
    weather = generate_mock_weather(
        start="2026-06-01",
        periods=1,
        seed=42,
    )

    weather[("wind_speed", 10)] = 10.0

    forecast = forecast_wind_power(weather)

    assert forecast.iloc[0] == pytest.approx(3000, abs=1)


def test_skystream_power_forecast():
    weather = generate_mock_weather(
        start="2026-06-01",
        periods=24,
        seed=42,
    )

    turbine = create_skystream_turbine()
    forecast = forecast_wind_power(weather, turbine)

    assert len(forecast) == 24
    assert (forecast >= 0).all()
    assert (forecast <= 2400).all()


def test_power_to_energy():
    power = pd.Series([500, 1200, 2000])

    energy = power_to_energy(power, interval_minutes=30)

    expected = pd.Series([250.0, 600.0, 1000.0])

    pd.testing.assert_series_equal(energy, expected)


@pytest.mark.parametrize("interval_minutes", [0, -1, -30, -60])
def test_power_to_energy_rejects_invalid_interval(interval_minutes):
    power = pd.Series([500, 1200])

    with pytest.raises(ValueError):
        power_to_energy(power, interval_minutes=interval_minutes)