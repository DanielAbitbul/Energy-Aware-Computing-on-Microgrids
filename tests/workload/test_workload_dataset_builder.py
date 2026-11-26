
import pandas as pd
import torch

from workload.dataset_builder import (
    create_training_dataset,
    create_training_dataset_from_splits,
)


def create_sample_csv(path):
    rows = [
        (
            f"Explain concept number {i} in detail.",
            10 + i,
            ["explanation", "question", "coding"][i % 3],
            [0.1, 0.2, 0.3][i % 3],
            20 + i,
            1.5 + i,
            5.0 + i,
        )
        for i in range(30)
    ]

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "prompt",
            "input_token_count",
            "task_type",
            "complexity",
            "output_token_count",
            "runtime_seconds",
            "energy_joules",
        ],
    )

    dataframe.to_csv(path, index=False)


def create_sample_parquet(path, num_examples=12):
    rows = [
        {
            "prompt_text": f"Explain concept number {i} in detail.",
            "formatted_input_tokens": 10 + i,
            "task_type_1": ["explanation", "question", "coding"][i % 3],
            "prompt_complexity_score": [0.1, 0.2, 0.3][i % 3],
            "output_tokens": 20 + i,
            "elapsed_s": 1.5 + i,
            "energy_j": 5.0 + i,
            "energy_valid": True,
        }
        for i in range(num_examples)
    ]

    pd.DataFrame(rows).to_parquet(path, index=False)


def test_create_training_dataset_produces_expected_tensors(
    tmp_path,
):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "dataset.pt"

    create_sample_csv(input_path)

    create_training_dataset(
        input_path=input_path,
        output_path=output_path,
    )

    dataset = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    assert dataset["X_train"].shape[1] == 389
    assert dataset["y_train"].shape[1] == 1

    assert dataset["X_val"].shape[1] == 389
    assert dataset["y_val"].shape[1] == 1

    assert dataset["X_test"].shape[1] == 389
    assert dataset["y_test"].shape[1] == 1

    assert dataset["y_train"].dtype == torch.float32


def test_create_training_dataset_splits_all_examples(
    tmp_path,
):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "dataset.pt"

    create_sample_csv(input_path)

    create_training_dataset(
        input_path=input_path,
        output_path=output_path,
    )

    dataset = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    total_examples = (
        len(dataset["X_train"])
        + len(dataset["X_val"])
        + len(dataset["X_test"])
    )

    assert total_examples == 30


def test_create_training_dataset_saves_metadata(
    tmp_path,
):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "dataset.pt"

    create_sample_csv(input_path)

    create_training_dataset(
        input_path=input_path,
        output_path=output_path,
    )

    dataset = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    metadata = dataset["metadata"]

    assert metadata["embedding_model"]
    assert metadata["task_type_mapping"]
    assert len(metadata["feature_names"]) == 389
    assert metadata["target_names"] == ["runtime_seconds"]

    assert metadata["normalization"]["targets"]["mean"].shape == (1,)
    assert metadata["normalization"]["targets"]["std"].shape == (1,)

    assert len(metadata["split_indices"]["train"]) > 0
    assert len(metadata["split_indices"]["validation"]) > 0
    assert len(metadata["split_indices"]["test"]) > 0


def test_create_training_dataset_from_splits_preserves_splits(
    tmp_path,
):
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    test_path = tmp_path / "test.parquet"
    output_path = tmp_path / "dataset.pt"

    create_sample_parquet(train_path, num_examples=18)
    create_sample_parquet(validation_path, num_examples=6)
    create_sample_parquet(test_path, num_examples=6)

    create_training_dataset_from_splits(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        output_path=output_path,
    )

    dataset = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    assert dataset["X_train"].shape == (18, 389)
    assert dataset["y_train"].shape == (18, 1)

    assert dataset["X_val"].shape == (6, 389)
    assert dataset["y_val"].shape == (6, 1)

    assert dataset["X_test"].shape == (6, 389)
    assert dataset["y_test"].shape == (6, 1)

    assert dataset["metadata"]["target_names"] == ["runtime_seconds"]
    assert dataset["metadata"]["normalization"]["targets"]["mean"].shape == (1,)
    assert dataset["metadata"]["normalization"]["targets"]["std"].shape == (1,)
