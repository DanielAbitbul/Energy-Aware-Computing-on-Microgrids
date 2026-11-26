
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# CONFIG
# -----------------------------

@dataclass(frozen=True)
class WorkloadModelConfig:
    input_size: int = 396
    hidden_sizes: tuple[int, ...] = (256, 128)
    min_std: float = 1e-6

    def __post_init__(self):
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")

        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")

        if self.min_std <= 0:
            raise ValueError("min_std must be positive")


# -----------------------------
# MODEL
# -----------------------------

class WorkloadPredictor(nn.Module):

    def __init__(self, config: WorkloadModelConfig | None = None):
        super().__init__()

        self.config = config or WorkloadModelConfig()

        layers = []
        previous_size = self.config.input_size

        for hidden_size in self.config.hidden_sizes:
            layers.extend([
                nn.Linear(previous_size, hidden_size),
                nn.ReLU(),
            ])
            previous_size = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        self.mean_head = nn.Linear(previous_size, 1)
        self.std_head = nn.Linear(previous_size, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.shared_layers(x)

        mean = self.mean_head(features)

        std = (
            F.softplus(self.std_head(features))
            + self.config.min_std
        )

        return mean, std
