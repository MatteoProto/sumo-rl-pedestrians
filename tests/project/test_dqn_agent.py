import numpy as np
import torch

from sumo_rl.project.dqn_agent import DQNAgent


def test_dqn_agent():
    agent = DQNAgent(
        state_size=29,
        action_size=4,
        device=torch.device("cpu"),
        batch_size=4,
        buffer_size=100,
        target_update_frequency=10,
    )

    state = np.zeros(29, dtype=np.float32)

    action = agent.select_action(
        state=state,
        epsilon=0.0,
    )

    assert 0 <= action < 4

    for index in range(10):
        next_state = np.full(
            29,
            index + 1,
            dtype=np.float32,
        )

        agent.store_transition(
            state=state,
            action=index % 4,
            reward=float(index),
            next_state=next_state,
            done=False,
        )

    loss = agent.train_step()

    assert loss is not None
    assert loss >= 0.0


if __name__ == "__main__":
    test_dqn_agent()
    print("DQNAgent test superato.")