import torch
import pytest

from workload.preprocessing import embed_prompts, fit_preprocessing_config, transform_inputs
from workload.dataset import WorkloadInput, WorkloadMeasurements


def test_embed_prompts_returns_tensor():
    prompts = [
        "Explain binary search.",
        "What is Python?",
    ]

    embeddings = embed_prompts(prompts)

    assert isinstance(embeddings, torch.Tensor)


def test_embed_prompts_returns_correct_dimensions():
    prompts = [
        "Explain binary search.",
        "What is Python?",
    ]

    embeddings = embed_prompts(prompts)

    assert embeddings.shape == (2, 384)


def test_embed_prompts_returns_valid_values():
    prompts = [
        "Explain binary search.",
        "What is Python?",
    ]

    embeddings = embed_prompts(prompts)

    assert torch.isfinite(embeddings).all()


def test_fit_preprocessing_config_calculates_training_statistics():
    examples = [
        (
            WorkloadInput("Prompt 1", 100, "question", 0.1),
            WorkloadMeasurements(50, 2.0, 10.0),
        ),
        (
            WorkloadInput("Prompt 2", 200, "coding", 0.2),
            WorkloadMeasurements(100, 4.0, 20.0),
        ),
        (
            WorkloadInput("Prompt 3", 300, "question", 0.3),
            WorkloadMeasurements(150, 6.0, 30.0),
        ),
    ]

    config = fit_preprocessing_config(examples)

    assert config.task_type_mapping == {
        "coding": 0,
        "question": 1,
    }

    assert config.complexity_mean == pytest.approx(0.2)
    assert config.complexity_std == pytest.approx(0.08164966)
    assert config.token_count_mean == 200
    assert config.token_count_std > 0

    assert torch.allclose(
        config.target_mean,
        torch.tensor([100.0, 4.0, 20.0]),
    )


def test_transform_inputs_produces_expected_features():
    examples = [
        (
            WorkloadInput("Prompt 1", 100, "question", 0.1),
            WorkloadMeasurements(50, 2.0, 10.0),
        ),
        (
            WorkloadInput("Prompt 2", 200, "coding", 0.2),
            WorkloadMeasurements(100, 4.0, 20.0),
        ),
        (
            WorkloadInput("Prompt 3", 300, "question", 0.3),
            WorkloadMeasurements(150, 6.0, 30.0),
        ),
    ]

    config = fit_preprocessing_config(examples)

    new_input = [
        WorkloadInput(
            "Explain how a binary search tree works.",
            200,
            "question",
            0.2,
        )
    ]

    features = transform_inputs(new_input, config)

    assert features.shape == (1, 388)
    assert features.dtype == torch.float32

    # Token count of 200 equals the training mean.
    assert features[0, 384].item() == 0.0

    # Task type "question" has index 1.
    assert torch.equal(
        features[0, 385:387],
        torch.tensor([0.0, 1.0]),
    )

    # Complexity score of 0.2 equals the training mean.
    assert features[0, 387].item() == pytest.approx(0.0)