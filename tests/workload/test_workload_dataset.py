import pandas as pd
import pytest

from workload.dataset import (
    WorkloadInput,
    WorkloadMeasurements,
    load_dataset,
    load_parquet_dataset,
)


def test_load_dataset_reads_valid_csv(tmp_path):
    path = tmp_path / "measurements.csv"

    path.write_text(
        "prompt,input_token_count,task_type,complexity,"
        "output_token_count,runtime_seconds,energy_joules\n"
        '"Explain binary search.",5,explanation,low,120,2.5,45.0\n',
        encoding="utf-8",
    )

    examples = load_dataset(path)

    assert len(examples) == 1

    workload, measurements = examples[0]

    assert workload.prompt == "Explain binary search."
    assert workload.input_token_count == 5
    assert workload.task_type == "explanation"
    assert workload.complexity == "low"

    assert measurements.output_token_count == 120
    assert measurements.runtime_seconds == 2.5
    assert measurements.energy_joules == 45.0


def test_load_dataset_allows_missing_input_token_count(tmp_path):
    path = tmp_path / "measurements.csv"

    path.write_text(
        "prompt,task_type,complexity,"
        "output_token_count,runtime_seconds,energy_joules\n"
        '"What is 2 + 2?",question,low,8,0.5,4.0\n',
        encoding="utf-8",
    )

    examples = load_dataset(path)

    assert examples[0][0].input_token_count is None


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "measurements.csv"

    path.write_text(
        "prompt,task_type\n"
        '"Explain binary search.",explanation\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing columns"):
        load_dataset(path)


def test_load_dataset_rejects_negative_measurements(tmp_path):
    path = tmp_path / "measurements.csv"

    path.write_text(
        "prompt,task_type,complexity,"
        "output_token_count,runtime_seconds,energy_joules\n"
        '"What is 2 + 2?",question,low,8,-0.5,4.0\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        load_dataset(path)


def test_load_parquet_dataset_returns_required_columns(tmp_path):
    data = pd.DataFrame({
        "prompt_text": ["What is Python?"],
        "formatted_input_tokens": [25],
        "task_type_1": ["Open QA"],
        "prompt_complexity_score": [0.15],
        "output_tokens": [50],
        "elapsed_s": [2.5],
        "energy_j": [30.0],
        "energy_valid": [True],
    })

    path = tmp_path / "dataset.parquet"
    data.to_parquet(path)

    result = load_parquet_dataset(path)

    assert list(result.columns) == [
        "prompt",
        "input_token_count",
        "task_type",
        "complexity_score",
        "output_token_count",
        "runtime_seconds",
        "energy_joules",
    ]

    assert len(result) == 1


def test_load_parquet_dataset_excludes_invalid_energy_rows(tmp_path):
    data = pd.DataFrame({
        "prompt_text": ["Valid prompt", "Invalid prompt"],
        "formatted_input_tokens": [25, 30],
        "task_type_1": ["Open QA", "Closed QA"],
        "prompt_complexity_score": [0.15, 0.20],
        "output_tokens": [50, 60],
        "elapsed_s": [2.5, 3.0],
        "energy_j": [30.0, 40.0],
        "energy_valid": [True, False],
    })

    path = tmp_path / "dataset.parquet"
    data.to_parquet(path)

    result = load_parquet_dataset(path)

    assert len(result) == 1
    assert result.iloc[0]["prompt"] == "Valid prompt"


def test_load_parquet_dataset_rejects_missing_columns(tmp_path):
    data = pd.DataFrame({
        "prompt_text": ["What is Python?"],
        "formatted_input_tokens": [25],
    })

    path = tmp_path / "dataset.parquet"
    data.to_parquet(path)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_parquet_dataset(path)


def test_load_parquet_dataset_rejects_empty_dataset(tmp_path):
    data = pd.DataFrame({
        "prompt_text": [],
        "formatted_input_tokens": [],
        "task_type_1": [],
        "prompt_complexity_score": [],
        "output_tokens": [],
        "elapsed_s": [],
        "energy_j": [],
        "energy_valid": [],
    })

    path = tmp_path / "dataset.parquet"
    data.to_parquet(path)

    with pytest.raises(ValueError, match="No valid examples"):
        load_parquet_dataset(path)


def test_workload_input_stores_prompt_features():
    workload = WorkloadInput(
        prompt="Explain binary search.",
        input_token_count=5,
        task_type="explanation",
        complexity="low",
    )

    assert workload.prompt == "Explain binary search."
    assert workload.input_token_count == 5
    assert workload.task_type == "explanation"
    assert workload.complexity == "low"


def test_workload_measurements_stores_observed_targets():
    measurements = WorkloadMeasurements(
        output_token_count=120,
        runtime_seconds=2.5,
        energy_joules=45.0,
    )

    assert measurements.output_token_count == 120
    assert measurements.runtime_seconds == 2.5
    assert measurements.energy_joules == 45.0