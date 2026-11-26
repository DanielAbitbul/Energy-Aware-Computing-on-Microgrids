
import torch

from workload.model import WorkloadPredictor


def predict(
    model: WorkloadPredictor,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate runtime predictions using a trained model."""
    model.eval()

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        mean, std = model(inputs)

    return mean.cpu(), std.cpu()


def denormalize_predictions(
    mean: torch.Tensor,
    std: torch.Tensor,
    metadata: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized predicted means and standard deviations to seconds."""
    target_stats = metadata["normalization"]["targets"]

    target_mean = torch.as_tensor(target_stats["mean"]).float().reshape(-1)[0]
    target_std = torch.as_tensor(target_stats["std"]).float().reshape(-1)[0]

    actual_mean = mean * target_std + target_mean
    actual_std = std * target_std

    return actual_mean, actual_std


def denormalize_targets(
    targets: torch.Tensor,
    metadata: dict,
) -> torch.Tensor:
    """Convert normalized target values back to seconds."""
    target_stats = metadata["normalization"]["targets"]

    target_mean = torch.as_tensor(target_stats["mean"]).float().reshape(-1)[0]
    target_std = torch.as_tensor(target_stats["std"]).float().reshape(-1)[0]

    return targets * target_std + target_mean


def evaluate_predictions(
    actual: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, float]:
    """Calculate point-prediction and uncertainty metrics."""
    actual = actual.float().reshape(-1)
    mean = mean.float().reshape(-1)
    std = std.float().reshape(-1)

    errors = mean - actual

    mae = torch.mean(torch.abs(errors))
    rmse = torch.sqrt(torch.mean(errors.square()))

    lower_bound = mean - std
    upper_bound = mean + std

    coverage = torch.mean(
        ((actual >= lower_bound) & (actual <= upper_bound)).float()
    )

    interval_width = torch.mean(upper_bound - lower_bound)

    return {
        "mae": mae.item(),
        "rmse": rmse.item(),
        "one_std_coverage": coverage.item(),
        "mean_interval_width": interval_width.item(),
    }
