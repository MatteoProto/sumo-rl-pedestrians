from collections import deque
import random
from typing import Deque, NamedTuple

import numpy as np
import torch


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Stores transitions and samples random mini-batches for DQN training."""

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        seed: int = 42,
    ) -> None:
        self.memory: Deque[Experience] = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.random = random.Random(seed)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        experience = Experience(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        )

        self.memory.append(experience)

    def sample(
        self,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(self.memory) < self.batch_size:
            raise ValueError(
                "Not enough experiences in replay buffer "
                f"({len(self.memory)}/{self.batch_size})."
            )

        experiences = self.random.sample(
            list(self.memory),
            k=self.batch_size,
        )

        states = torch.as_tensor(
            np.stack([experience.state for experience in experiences]),
            dtype=torch.float32,
            device=device,
        )

        actions = torch.as_tensor(
            [experience.action for experience in experiences],
            dtype=torch.int64,
            device=device,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            [experience.reward for experience in experiences],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        next_states = torch.as_tensor(
            np.stack([experience.next_state for experience in experiences]),
            dtype=torch.float32,
            device=device,
        )

        dones = torch.as_tensor(
            [experience.done for experience in experiences],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)