
import pytest
import torch

from workload.inference import (
    predict,
    denormalize_predictions,
    denormalize_targets,
    evaluate_predictions,
)
from workload.model import WorkloadPredictor


def test_predict_returns_expected_shapes():
    model = WorkloadPredictor()
    inputs = torch.randn(8, 396)

    mean, std = predict(model, inputs)

    assert mean.shape == (8, 1)
    assert std.shape == (8, 1)


def test_predict_returns_positive_standard_deviation():
    model = WorkloadPredictor()
    inputs = torch.randn(8, 396)

    _, std = predict(model, inputs)

    assert torch.all(std > 0)


def test_predict_does_not_update_model_parameters():
    model = WorkloadPredictor()
    inputs = torch.randn(8, 396)

    original_parameters = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
    }

    predict(model, inputs)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original_parameters[name])


def test_denormalize_predictions():
    mean = torch.tensor([[0.0], [1.0]])
    std = torch.tensor([[1.0], [2.0]])

    metadata = {
        "normalization": {
            "targets": {
                "mean": torch.tensor([10.0]),
                "std": torch.tensor([2.0]),
            }
        }
    }

    actual_mean, actual_std = denormalize_predictions(
        mean, std, metadata
    )

    assert torch.allclose(
        actual_mean,
        torch.tensor([[10.0], [12.0]])
    )
    assert torch.allclose(
        actual_std,
        torch.tensor([[2.0], [4.0]])
    )


def test_denormalize_targets():
    targets = torch.tensor([[0.0], [1.0], [-1.0]])

    metadata = {
        "normalization": {
            "targets": {
                "mean": torch.tensor([10.0]),
                "std": torch.tensor([2.0]),
            }
        }
    }

    actual_targets = denormalize_targets(targets, metadata)

    assert torch.allclose(
        actual_targets,
        torch.tensor([[10.0], [12.0], [8.0]])
    )


def test_evaluate_predictions_calculates_metrics():
    actual = torch.tensor([[10.0], [20.0]])
    mean = torch.tensor([[12.0], [18.0]])
    std = torch.tensor([[2.0], [1.0]])

    metrics = evaluate_predictions(actual, mean, std)

    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(2.0)
    assert metrics["one_std_coverage"] == pytest.approx(0.5)
    assert metrics["mean_interval_width"] == pytest.approx(3.0)
