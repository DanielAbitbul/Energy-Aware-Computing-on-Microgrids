
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_training_history(
    history: dict,
    output_path: str | Path,
) -> None:
    """Plot training and validation loss over epochs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["validation_loss"], label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Gaussian NLL loss")
    plt.title("Training history")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_predictions(
    actual: torch.Tensor,
    predicted_mean: torch.Tensor,
    output_path: str | Path,
) -> None:
    """Plot predicted runtime against actual runtime."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual = torch.as_tensor(actual).detach().cpu().numpy().flatten()
    predicted_mean = (
        torch.as_tensor(predicted_mean).detach().cpu().numpy().flatten()
    )

    lower = min(actual.min(), predicted_mean.min())
    upper = max(actual.max(), predicted_mean.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted_mean, alpha=0.6)
    plt.plot([lower, upper], [lower, upper], "--", label="Perfect prediction")

    plt.xlabel("Actual runtime (seconds)")
    plt.ylabel("Predicted runtime (seconds)")
    plt.title("Actual vs. predicted runtime")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residuals(
    actual: torch.Tensor,
    predicted_mean: torch.Tensor,
    output_path: str | Path,
) -> None:
    """Plot the distribution of prediction residuals."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual = torch.as_tensor(actual).detach().cpu().numpy().flatten()
    predicted_mean = (
        torch.as_tensor(predicted_mean).detach().cpu().numpy().flatten()
    )

    residuals = predicted_mean - actual

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.75)
    plt.axvline(0, linestyle="--", label="Zero error")

    plt.xlabel("Prediction error (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of prediction errors")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_uncertainty(
    actual: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_std: torch.Tensor,
    output_path: str | Path,
) -> None:
    """Plot predictions with one-standard-deviation uncertainty intervals."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual = torch.as_tensor(actual).detach().cpu().numpy().flatten()
    predicted_mean = (
        torch.as_tensor(predicted_mean).detach().cpu().numpy().flatten()
    )
    predicted_std = (
        torch.as_tensor(predicted_std).detach().cpu().numpy().flatten()
    )

    # Sort by predicted runtime to make the plot easier to interpret.
    indices = predicted_mean.argsort()
    actual = actual[indices]
    predicted_mean = predicted_mean[indices]
    predicted_std = predicted_std[indices]

    x = range(len(actual))

    plt.figure(figsize=(12, 5))
    plt.scatter(x, actual, label="Actual runtime", alpha=0.7, s=20)
    plt.plot(x, predicted_mean, label="Predicted mean")
    plt.fill_between(
        x,
        predicted_mean - predicted_std,
        predicted_mean + predicted_std,
        alpha=0.2,
        label="±1 standard deviation",
    )

    plt.xlabel("Test samples (sorted by predicted runtime)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime predictions and uncertainty")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_all_plots(
    history: dict,
    actual: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_std: torch.Tensor,
    output_dir: str | Path,
) -> None:
    """Generate and save all model evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(
        history,
        output_dir / "training_loss.png",
    )

    plot_predictions(
        actual,
        predicted_mean,
        output_dir / "predictions.png",
    )

    plot_residuals(
        actual,
        predicted_mean,
        output_dir / "residuals.png",
    )

    plot_uncertainty(
        actual,
        predicted_mean,
        predicted_std,
        output_dir / "uncertainty.png",
    )
