import pandas as pd


def power_to_energy(
    power: pd.Series,
    interval_minutes: int = 60,
) -> pd.Series:
    if interval_minutes <= 0:
        raise ValueError("Interval must be positive")

    interval_hours = interval_minutes / 60

    return power * interval_hours