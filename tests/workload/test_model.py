
import pytest
import torch

from workload.model import WorkloadModelConfig, WorkloadPredictor


def test_model_produces_expected_output_shapes():
    model = WorkloadPredictor()

    x = torch.randn(8, 396)

    mean, std = model(x)

    assert mean.shape == (8, 1)
    assert std.shape == (8, 1)


def test_model_supports_custom_configuration():
    config = WorkloadModelConfig(
        input_size=20,
        hidden_sizes=(64, 32),
    )

    model = WorkloadPredictor(config)

    x = torch.randn(4, 20)

    mean, std = model(x)

    assert mean.shape == (4, 1)
    assert std.shape == (4, 1)


def test_model_produces_positive_standard_deviation():
    model = WorkloadPredictor()

    x = torch.randn(16, 396)

    _, std = model(x)

    assert torch.all(std > 0)


def test_model_outputs_are_differentiable():
    model = WorkloadPredictor()

    x = torch.randn(4, 396)

    mean, std = model(x)
    loss = mean.mean() + std.mean()

    loss.backward()

    assert model.mean_head.weight.grad is not None
    assert model.std_head.weight.grad is not None


def test_model_config_rejects_invalid_input_size():
    with pytest.raises(ValueError):
        WorkloadModelConfig(input_size=0)


def test_model_config_rejects_invalid_hidden_sizes():
    with pytest.raises(ValueError):
        WorkloadModelConfig(hidden_sizes=(256, 0))


def test_model_config_rejects_empty_hidden_sizes():
    with pytest.raises(ValueError):
        WorkloadModelConfig(hidden_sizes=())


def test_model_config_rejects_invalid_min_std():
    with pytest.raises(ValueError):
        WorkloadModelConfig(min_std=0)
