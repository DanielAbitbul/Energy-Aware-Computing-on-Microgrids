# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

7th semester AAU project: energy-aware computing on microgrids. Two components are implemented: wind energy forecasting (`src/energy_forecasting`) and LLM workload runtime prediction (`src/workload`). `src/scheduler/` and `src/simulation/` are empty placeholders (`.gitkeep`).

## Commands

Python >=3.11. Install in editable mode with dev extras (pytest):

```
pip install -e ".[dev]"
```

- Run all tests: `pytest`
- Run a single test: `pytest tests/energy_forecasting/test_energy_forecasting.py::test_mock_weather_generation`
- Train and evaluate the workload predictor end to end: `python src/workload/main.py`
- Inspect the raw parquet splits: `python scripts/inspect_dataset.py`
- Fetch the latest DMI forecast into `data/dmi_latest_forecast.csv`: `python src/energy_forecasting/weather/dmi_latest_forecast.py` (`--loop`, `--interval-hours` (min 1), `--retries`, `--retry-delay-seconds`)
- Plot a simulated forecast with mock weather: `python scripts/plot_wind_forecast.py`
- Clean the raw dataset CSV (expects `scripts/merged.csv` by default): `python scripts/CleanData.py --help`

pytest is configured with `pythonpath = ["src"]`, so tests import `energy_forecasting` and `workload` directly. Tests live in `tests/<package>/`.

## Architecture

### Energy forecasting (`src/energy_forecasting/`)

A chain of small modules:

1. **Weather source** (`weather/`): either `dmi_latest_forecast.py` (fetches the DMI HARMONIE open-data API for a hardcoded point, writes a flat CSV to `data/`, and `load_weather_dataframe` reads it back) or `mock.py` (seeded random hourly weather).
2. **Wind power** (`wind.py`): `forecast_wind_power` runs a windpowerlib `ModelChain` (logarithmic wind-speed extrapolation, ideal-gas density, power-curve output) and returns power in W.
3. **Energy** (`energy.py`): `power_to_energy` converts power to Wh per interval.
4. **Pipeline** (`pipeline.py`): `run_energy_forecast` ties these together; defaults to the DMI CSV when no weather is passed.

Turbines are defined in `config.py` (`create_wind_turbine`, `create_skystream_turbine`). The Skystream curve and hub height are marked ASSUMPTION in the code, not manufacturer data.

#### Weather DataFrame contract

windpowerlib requires a DataFrame with a DatetimeIndex (tz-aware UTC) and MultiIndex columns `("wind_speed", 10)`, `("temperature", 2)` (Kelvin) and `("pressure", 0)` (Pa). Both the mock generator and `load_weather_dataframe` produce this shape; any new weather source must as well. The DMI CSV columns are `timestamp, wind_speed_10m_mps, temperature_2m_k, pressure_surface_pa`.

### Workload prediction (`src/workload/`)

Predicts LLM inference runtime (seconds) from a prompt as a Gaussian (mean and std), trained with Gaussian NLL. `main.py` orchestrates the flow, driven by the frozen `ExperimentConfig` dataclass (paths and hyperparameters; there is no CLI):

1. **Data** (`dataset.py`): `WorkloadInput` and `WorkloadMeasurements` dataclasses, plus loaders for CSV and parquet. Raw splits are expected at `data/raw/{train,val,test}.parquet`.
2. **Preprocessing** (`preprocessing.py`): prompts are embedded with `sentence-transformers/all-MiniLM-L6-v2`, then combined with the task type mapping and normalized token count and complexity. `PreprocessingConfig` holds the statistics fitted on the training split only.
3. **Dataset builder** (`dataset_builder.py`): `create_training_dataset_from_splits` writes a cached tensor dict (`X_*`, `y_*`, `metadata`) to `data/processed/workload_dataset.pt`. It is rebuilt only if the file is missing or `rebuild_dataset=True`, so stale caches persist after preprocessing changes.
4. **Model and training** (`model.py`, `trainer.py`): an MLP with shared layers and separate mean and std heads (softplus std), trained with early stopping.
5. **Inference and evaluation** (`inference.py`, `visualization.py`): predictions are produced in normalized space and must be passed through `denormalize_predictions` and `denormalize_targets` (using `metadata["normalization"]["targets"]`) before computing metrics or plots.

Outputs go to `models/`, `reports/results.json` and `reports/figures/`.

### Notes

- `data/`, `models/` and `reports/` are gitignored in full, so generated files and the raw parquet data are never tracked. The DMI API rate-limits aggressively (429 handling suggests waiting ~3 hours), so avoid frequent fetches.
- `scripts/` are standalone utilities, not part of the installed package.
