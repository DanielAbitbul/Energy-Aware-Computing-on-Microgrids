
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class WorkloadInput:
    prompt: str
    input_token_count: int | None
    task_type: str
    complexity: str


@dataclass(frozen=True)
class WorkloadMeasurements:
    output_token_count: int
    runtime_seconds: float
    energy_joules: float


REQUIRED_COLUMNS = {
    "prompt",
    "task_type",
    "complexity",
    "output_token_count",
    "runtime_seconds",
    "energy_joules",
}


def load_dataset(path: str | Path):
    """Load and validate a workload dataset from a CSV file."""

    examples = []

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        # Validate that all required columns are present.
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing columns.")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"CSV file is missing columns: {sorted(missing_columns)}"
            )

        for row in reader:
            # Input token count is optional.
            input_token_count = row.get("input_token_count")

            if input_token_count is not None and input_token_count.strip():
                input_token_count = int(input_token_count)
            else:
                input_token_count = None

            # Complexity supports both categorical and numerical values.
            complexity = row["complexity"]

            try:
                complexity = float(complexity)
            except ValueError:
                pass

            # Convert measurements to their appropriate numerical types.
            output_token_count = int(row["output_token_count"])
            runtime_seconds = float(row["runtime_seconds"])
            energy_joules = float(row["energy_joules"])

            # Measurements cannot be negative.
            if (
                output_token_count < 0
                or runtime_seconds < 0
                or energy_joules < 0
            ):
                raise ValueError("Measurements cannot be negative.")

            workload_input = WorkloadInput(
                prompt=row["prompt"],
                input_token_count=input_token_count,
                task_type=row["task_type"],
                complexity=complexity,
            )

            measurements = WorkloadMeasurements(
                output_token_count=output_token_count,
                runtime_seconds=runtime_seconds,
                energy_joules=energy_joules,
            )

            examples.append((workload_input, measurements))

    return examples


def load_parquet_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a workload dataset from a Parquet file."""

    required_columns = {
        "prompt_text",
        "formatted_input_tokens",
        "task_type_1",
        "prompt_complexity_score",
        "output_tokens",
        "elapsed_s",
        "energy_j",
        "energy_valid",
    }

    dataframe = pd.read_parquet(path)

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Only retain examples with valid energy measurements.
    dataframe = dataframe.loc[
        dataframe["energy_valid"].eq(True)
    ].copy()

    if dataframe.empty:
        raise ValueError("No valid examples found in the dataset.")

    # Keep only the columns needed by our prediction pipeline.
    dataframe = dataframe[
        [
            "prompt_text",
            "formatted_input_tokens",
            "task_type_1",
            "prompt_complexity_score",
            "output_tokens",
            "elapsed_s",
            "energy_j",
        ]
    ]

    # Give the columns consistent names for our pipeline.
    dataframe = dataframe.rename(columns={
        "prompt_text": "prompt",
        "formatted_input_tokens": "input_token_count",
        "task_type_1": "task_type",
        "prompt_complexity_score": "complexity_score",
        "output_tokens": "output_token_count",
        "elapsed_s": "runtime_seconds",
        "energy_j": "energy_joules",
    })

    return dataframe.reset_index(drop=True)
