from windpowerlib import WindTurbine


def create_wind_turbine() -> WindTurbine:
    """Original hypothetical 5 kW turbine used in our tests."""
    return WindTurbine(
        nominal_power=5000,
        hub_height=10,
        power_curve={
            "wind_speed": [0, 3, 5, 8, 10, 12, 15, 25, 30],
            "value": [0, 0, 300, 1500, 3000, 5000, 5000, 0, 0],
        },
    )


def create_skystream_turbine() -> WindTurbine:
    """Skystream 3.7 with provisional installation assumptions."""
    return WindTurbine(
        nominal_power=2400,
        hub_height=10,  # ASSUMPTION: AAU installation height unknown
        power_curve={
            # ASSUMPTION: Illustrative curve, NOT manufacturer data
            "wind_speed": [0, 3.5, 5, 7, 9, 11, 13, 20, 25],
            "value": [0, 0, 150, 500, 1100, 1800, 2400, 2400, 0],
        },
    )