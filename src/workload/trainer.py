
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from workload.model import WorkloadPredictor


# -----------------------------
# CONFIG
# -----------------------------

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 0.0
    device: str = "auto"

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")

        if self.patience <= 0:
            raise ValueError("patience must be positive")

        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")


# -----------------------------
# DEVICE
# -----------------------------

def get_device(device: str = "auto") -> torch.device:

    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# -----------------------------
# TRAINING
# -----------------------------

def train_epoch(
    model: WorkloadPredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:

    model.train()

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        optimizer.zero_grad()

        mean, std = model(inputs)

        loss = criterion(mean, targets, std.square())

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# -----------------------------
# EVALUATION
# -----------------------------

def evaluate_model(
    model: WorkloadPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device | None = None,
) -> float:

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    criterion = nn.GaussianNLLLoss(full=True)

    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        mean, std = model(inputs)
        loss = criterion(mean, targets, std.square())

    return loss.item()


# -----------------------------
# MODEL TRAINING
# -----------------------------

def train_model(
    model: WorkloadPredictor,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    config: TrainingConfig | None = None,
) -> dict:

    config = config or TrainingConfig()

    device = get_device(config.device)
    model.to(device)

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)

    train_dataset = TensorDataset(train_inputs, train_targets)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    criterion = nn.GaussianNLLLoss(full=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
        "best_epoch": 0,
    }

    best_validation_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(config.epochs):

        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
        )

        validation_loss = evaluate_model(
            model,
            validation_inputs,
            validation_targets,
            device,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        if validation_loss < best_validation_loss - config.min_delta:

            best_validation_loss = validation_loss

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            history["best_epoch"] = epoch + 1
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history
