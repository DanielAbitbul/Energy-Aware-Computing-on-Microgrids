
from pathlib import Path

import torch

from workload.dataset import (
    WorkloadInput,
    WorkloadMeasurements,
    load_dataset,
    load_parquet_dataset,
)
from workload.preprocessing import (
    MODEL_NAME,
    fit_preprocessing_config,
    transform_inputs,
)


TARGET_NAMES = ["runtime_seconds"]


def create_training_dataset(
    input_path: str | Path,
    output_path: str | Path,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> None:
    examples = load_dataset(input_path)

    if len(examples) < 3:
        raise ValueError("At least three examples are required.")

    if (
        test_size <= 0
        or validation_size <= 0
        or test_size + validation_size >= 1
    ):
        raise ValueError(
            "Test and validation sizes must be positive "
            "and their sum must be less than 1."
        )

    workloads: list[WorkloadInput] = [
        example[0] for example in examples
    ]
    measurements: list[WorkloadMeasurements] = [
        example[1] for example in examples
    ]

    if any(workload.input_token_count is None for workload in workloads):
        raise ValueError(
            "All examples must have an input token count."
        )

    # Generate a reproducible random split.
    generator = torch.Generator().manual_seed(random_state)
    indices = torch.randperm(len(examples), generator=generator)

    test_count = max(1, round(len(examples) * test_size))
    validation_count = max(1, round(len(examples) * validation_size))
    train_count = len(examples) - test_count - validation_count

    if train_count < 1:
        raise ValueError(
            "Dataset is too small for the requested split sizes."
        )

    train_indices = indices[:train_count]
    validation_indices = indices[
        train_count:train_count + validation_count
    ]
    test_indices = indices[train_count + validation_count:]

    # Fit preprocessing parameters using training examples only.
    train_examples = [
        examples[index] for index in train_indices.tolist()
    ]
    config = fit_preprocessing_config(train_examples)

    # Transform each split using the same fitted configuration.
    X_train = transform_inputs(
        [workloads[index] for index in train_indices.tolist()],
        config,
    )
    X_val = transform_inputs(
        [workloads[index] for index in validation_indices.tolist()],
        config,
    )
    X_test = transform_inputs(
        [workloads[index] for index in test_indices.tolist()],
        config,
    )

    # Prepare and normalize runtime prediction targets.
    targets = torch.tensor(
        [
            [measurement.runtime_seconds]
            for measurement in measurements
        ],
        dtype=torch.float32,
    )

    # Runtime was the second target in the original configuration.
    runtime_mean = config.target_mean[1:2]
    runtime_std = config.target_std[1:2]

    targets = (targets - runtime_mean) / runtime_std

    y_train = targets[train_indices]
    y_val = targets[validation_indices]
    y_test = targets[test_indices]

    output = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": {
            "embedding_model": MODEL_NAME,
            "task_type_mapping": config.task_type_mapping,
            "feature_names": (
                [f"embedding_{i}" for i in range(384)]
                + ["input_token_count"]
                + [
                    f"task_type_{name}"
                    for name in config.task_type_mapping
                ]
                + ["complexity_score"]
            ),
            "target_names": TARGET_NAMES,
            "normalization": {
                "input_token_count": {
                    "mean": config.token_count_mean,
                    "std": config.token_count_std,
                },
                "complexity_score": {
                    "mean": config.complexity_mean,
                    "std": config.complexity_std,
                },
                "targets": {
                    "mean": runtime_mean,
                    "std": runtime_std,
                },
            },
            "split_indices": {
                "train": train_indices,
                "validation": validation_indices,
                "test": test_indices,
            },
            "prompts": [workload.prompt for workload in workloads],
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(output, output_path)


def create_training_dataset_from_splits(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
) -> None:
    # Load the three predefined splits.
    train_df = load_parquet_dataset(train_path)
    validation_df = load_parquet_dataset(validation_path)
    test_df = load_parquet_dataset(test_path)

    def convert_dataframe(dataframe):
        examples = []

        for row in dataframe.itertuples(index=False):
            workload = WorkloadInput(
                prompt=row.prompt,
                input_token_count=row.input_token_count,
                task_type=row.task_type,
                complexity=row.complexity_score,
            )

            measurements = WorkloadMeasurements(
                output_token_count=row.output_token_count,
                runtime_seconds=row.runtime_seconds,
                energy_joules=row.energy_joules,
            )

            examples.append((workload, measurements))

        return examples

    train_examples = convert_dataframe(train_df)
    validation_examples = convert_dataframe(validation_df)
    test_examples = convert_dataframe(test_df)

    # Fit preprocessing parameters using training examples only.
    config = fit_preprocessing_config(train_examples)

    # Transform all splits using the same fitted configuration.
    X_train = transform_inputs(
        [example[0] for example in train_examples],
        config,
    )
    X_val = transform_inputs(
        [example[0] for example in validation_examples],
        config,
    )
    X_test = transform_inputs(
        [example[0] for example in test_examples],
        config,
    )

    def create_targets(examples):
        return torch.tensor(
            [
                [measurement.runtime_seconds]
                for _, measurement in examples
            ],
            dtype=torch.float32,
        )

    # Prepare runtime targets for each split.
    y_train_raw = create_targets(train_examples)
    y_val_raw = create_targets(validation_examples)
    y_test_raw = create_targets(test_examples)

    # Normalize runtime using training-set statistics.
    runtime_mean = config.target_mean[1:2]
    runtime_std = config.target_std[1:2]

    y_train = (y_train_raw - runtime_mean) / runtime_std
    y_val = (y_val_raw - runtime_mean) / runtime_std
    y_test = (y_test_raw - runtime_mean) / runtime_std

    # Preserve the existing output format.
    all_examples = (
        train_examples + validation_examples + test_examples
    )

    output = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": {
            "embedding_model": MODEL_NAME,
            "task_type_mapping": config.task_type_mapping,
            "feature_names": (
                [f"embedding_{i}" for i in range(384)]
                + ["input_token_count"]
                + [
                    f"task_type_{name}"
                    for name in config.task_type_mapping
                ]
                + ["complexity_score"]
            ),
            "target_names": TARGET_NAMES,
            "normalization": {
                "input_token_count": {
                    "mean": config.token_count_mean,
                    "std": config.token_count_std,
                },
                "complexity_score": {
                    "mean": config.complexity_mean,
                    "std": config.complexity_std,
                },
                "targets": {
                    "mean": runtime_mean,
                    "std": runtime_std,
                },
            },
            "prompts": [
                workload.prompt for workload, _ in all_examples
            ],
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(output, output_path)
