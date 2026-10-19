import numpy as np
import pandas as pd


def generate_mock_weather(
    start: str,
    periods: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start=start,
        periods=periods,
        freq="h",
        tz="UTC",
    )

    weather = pd.DataFrame(
        {
            ("wind_speed", 10): rng.uniform(2, 15, periods),
            ("temperature", 2): rng.uniform(273, 303, periods),
            ("pressure", 0): rng.uniform(98000, 103000, periods),
        },
        index=timestamps,
    )

    return weather