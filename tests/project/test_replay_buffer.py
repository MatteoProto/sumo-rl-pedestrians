import numpy as np
import torch

from sumo_rl.project.replay_buffer import ReplayBuffer


def test_replay_buffer():
    buffer = ReplayBuffer(
        capacity=100,
        batch_size=4,
        seed=42,
    )

    for index in range(10):
        state = np.full(29, index, dtype=np.float32)
        next_state = np.full(29, index + 1, dtype=np.float32)

        buffer.add(
            state=state,
            action=index % 4,
            reward=float(index),
            next_state=next_state,
            done=False,
        )

    assert len(buffer) == 10

    batch = buffer.sample(torch.device("cpu"))

    states, actions, rewards, next_states, dones = batch

    assert states.shape == (4, 29)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, 29)
    assert dones.shape == (4, 1)


if __name__ == "__main__":
    test_replay_buffer()
    print("ReplayBuffer test superato.")