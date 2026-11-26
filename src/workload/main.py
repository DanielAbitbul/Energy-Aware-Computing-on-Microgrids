
from dataclasses import dataclass, field
from pathlib import Path

import torch

from workload.dataset_builder import create_training_dataset_from_splits
from workload.inference import (
    predict,
    denormalize_predictions,
    denormalize_targets,
    evaluate_predictions,
)
from workload.model import WorkloadModelConfig, WorkloadPredictor
from workload.trainer import TrainingConfig, train_model, evaluate_model
from workload.visualization import create_all_plots


@dataclass(frozen=True)
class ExperimentConfig:
    train_path: Path = Path("data/raw/train.parquet")
    validation_path: Path = Path("data/raw/val.parquet")
    test_path: Path = Path("data/raw/test.parquet")

    dataset_path: Path = Path("data/processed/workload_dataset.pt")
    model_path: Path = Path("models/workload_predictor.pt")
    results_path: Path = Path("reports/results.json")
    figures_path: Path = Path("reports/figures")

    rebuild_dataset: bool = False

    model: WorkloadModelConfig = field(
        default_factory=WorkloadModelConfig
    )
    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )


def prepare_dataset(config: ExperimentConfig) -> dict:
    """Build the processed dataset if necessary and load it."""
    if config.rebuild_dataset or not config.dataset_path.exists():
        create_training_dataset_from_splits(
            train_path=config.train_path,
            validation_path=config.validation_path,
            test_path=config.test_path,
            output_path=config.dataset_path,
        )

    return torch.load(
        config.dataset_path,
        map_location="cpu",
        weights_only=False,
    )


def save_model(
    model: WorkloadPredictor,
    config: ExperimentConfig,
    best_epoch: int,
) -> None:
    """Save the trained model and its configuration."""
    config.model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.model,
            "training_config": config.training,
            "best_epoch": best_epoch,
        },
        config.model_path,
    )


def save_results(
    metrics: dict,
    history: dict,
    test_loss: float,
    config: ExperimentConfig,
) -> None:
    """Save evaluation metrics and training history."""
    import json

    config.results_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "test_loss": test_loss,
        "metrics": metrics,
        "best_epoch": history["best_epoch"],
        "validation_loss": history["validation_loss"][
            history["best_epoch"] - 1
        ],
        "training_history": history,
    }

    with config.results_path.open("w") as file:
        json.dump(results, file, indent=4)


def main():
    config = ExperimentConfig()

    # 1. Prepare dataset
    dataset = prepare_dataset(config)

    # 2. Initialize model
    model_config = WorkloadModelConfig(
        input_size=dataset["X_train"].shape[1],
        hidden_sizes=config.model.hidden_sizes,
        min_std=config.model.min_std,
    )

    model = WorkloadPredictor(model_config)

    # 3. Train model
    history = train_model(
        model,
        dataset["X_train"],
        dataset["y_train"],
        dataset["X_val"],
        dataset["y_val"],
        config.training,
    )

    # 4. Evaluate test loss
    test_loss = evaluate_model(
        model,
        dataset["X_test"],
        dataset["y_test"],
    )

    # 5. Run inference
    predicted_mean, predicted_std = predict(
        model,
        dataset["X_test"],
    )

    actual = denormalize_targets(
        dataset["y_test"],
        dataset["metadata"],
    )

    predicted_mean, predicted_std = denormalize_predictions(
        predicted_mean,
        predicted_std,
        dataset["metadata"],
    )

    # 6. Calculate metrics
    metrics = evaluate_predictions(
        actual,
        predicted_mean,
        predicted_std,
    )

    print("\nTest results")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"Test Gaussian NLL (normalized): {test_loss:.4f}")

    # 7. Generate plots
    create_all_plots(
        history,
        actual,
        predicted_mean,
        predicted_std,
        config.figures_path,
    )

    # 8. Save model and results
    save_model(model, config, history["best_epoch"])
    save_results(metrics, history, test_loss, config)

    print(f"\nModel saved to: {config.model_path}")
    print(f"Results saved to: {config.results_path}")
    print(f"Plots saved to: {config.figures_path}")


if __name__ == "__main__":
    main()
