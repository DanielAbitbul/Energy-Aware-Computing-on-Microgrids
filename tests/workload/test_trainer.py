
import pytest
import torch

from workload.model import WorkloadModelConfig, WorkloadPredictor
from workload.trainer import (
    TrainingConfig,
    evaluate_model,
    get_device,
    train_model,
)


def create_sample_data(input_size=10, num_samples=32):
    inputs = torch.randn(num_samples, input_size)

    targets = (
        inputs[:, 0:1] * 2
        + inputs[:, 1:2]
        + torch.randn(num_samples, 1) * 0.1
    )

    return inputs, targets


def test_training_config_has_expected_defaults():
    config = TrainingConfig()

    assert config.epochs == 100
    assert config.batch_size == 64
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.patience == 10
    assert config.device == "auto"


def test_training_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)

    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)

    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=0)

    with pytest.raises(ValueError):
        TrainingConfig(weight_decay=-1)

    with pytest.raises(ValueError):
        TrainingConfig(patience=0)

    with pytest.raises(ValueError):
        TrainingConfig(min_delta=-1)


def test_get_device_returns_valid_device():
    device = get_device("cpu")

    assert device == torch.device("cpu")


def test_evaluate_model_returns_finite_loss():
    model = WorkloadPredictor(
        WorkloadModelConfig(
            input_size=10,
            hidden_sizes=(16, 8),
        )
    )

    inputs, targets = create_sample_data()

    loss = evaluate_model(model, inputs, targets)

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_train_model_updates_parameters():
    model = WorkloadPredictor(
        WorkloadModelConfig(
            input_size=10,
            hidden_sizes=(16, 8),
        )
    )

    train_inputs, train_targets = create_sample_data()
    validation_inputs, validation_targets = create_sample_data(
        num_samples=8
    )

    initial_weights = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }

    config = TrainingConfig(
        epochs=3,
        batch_size=8,
        learning_rate=1e-2,
        patience=3,
        device="cpu",
    )

    train_model(
        model,
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        config,
    )

    weights_changed = any(
        not torch.equal(initial_weights[key], value)
        for key, value in model.state_dict().items()
    )

    assert weights_changed


def test_train_model_returns_training_history():
    model = WorkloadPredictor(
        WorkloadModelConfig(
            input_size=10,
            hidden_sizes=(16, 8),
        )
    )

    train_inputs, train_targets = create_sample_data()
    validation_inputs, validation_targets = create_sample_data(
        num_samples=8
    )

    config = TrainingConfig(
        epochs=3,
        batch_size=8,
        patience=3,
        device="cpu",
    )

    history = train_model(
        model,
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        config,
    )

    assert len(history["train_loss"]) > 0
    assert len(history["validation_loss"]) > 0

    assert history["best_epoch"] >= 1
    assert history["best_epoch"] <= len(history["validation_loss"])


def test_train_model_restores_best_weights():
    model = WorkloadPredictor(
        WorkloadModelConfig(
            input_size=10,
            hidden_sizes=(16, 8),
        )
    )

    train_inputs, train_targets = create_sample_data()
    validation_inputs, validation_targets = create_sample_data(
        num_samples=8
    )

    config = TrainingConfig(
        epochs=10,
        batch_size=8,
        patience=3,
        device="cpu",
    )

    history = train_model(
        model,
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        config,
    )

    final_validation_loss = evaluate_model(
        model,
        validation_inputs,
        validation_targets,
    )

    best_validation_loss = min(history["validation_loss"])

    assert final_validation_loss == pytest.approx(
        best_validation_loss,
        rel=1e-5,
    )
