import torch

from sumo_rl.project.networks import QNetwork


def test_q_network_output_shape():
    network = QNetwork(
        state_size=29,
        action_size=4,
    )

    fake_state = torch.zeros(1, 29)

    output = network(fake_state)

    assert output.shape == (1, 4)


if __name__ == "__main__":
    test_q_network_output_shape()
    print("QNetwork test superato.")