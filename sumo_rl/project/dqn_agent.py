import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from sumo_rl.project.networks import QNetwork
from sumo_rl.project.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_frequency: int = 500,
        seed: int = 42,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.random = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.policy_network = QNetwork(
            state_size=state_size,
            action_size=action_size,
        ).to(device)

        self.target_network = QNetwork(
            state_size=state_size,
            action_size=action_size,
        ).to(device)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            batch_size=batch_size,
            seed=seed,
        )

        self.training_steps = 0

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float,
    ) -> int:
        if self.random.random() < epsilon:
            return self.random.randrange(self.action_size)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.device)

        current_q_values = self.policy_network(states).gather(
            1,
            actions,
        )

        with torch.no_grad():
            next_q_values = self.target_network(
                next_states
            ).max(dim=1, keepdim=True).values

            target_q_values = (
                rewards
                + self.gamma
                * next_q_values
                * (1.0 - dones)
            )

        loss = self.loss_function(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "policy_network": self.policy_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(
            filepath,
            map_location=self.device,
        )

        self.policy_network.load_state_dict(
            checkpoint["policy_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.training_steps = checkpoint["training_steps"]