import torch
from torch import nn


class QNetwork(nn.Module):
    """Neural network that estimates one Q-value for each traffic-light action."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)