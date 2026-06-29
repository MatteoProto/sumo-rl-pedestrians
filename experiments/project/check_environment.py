from pathlib import Path

import gymnasium as gym
import sumo_rl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_DIR = (
    PROJECT_ROOT
    / "sumo_rl"
    / "nets"
    / "project_intersection_v1"
)


def main():
    env = gym.make(
        "sumo-rl-v0",
        net_file=str(SCENARIO_DIR / "project_intersection.net.xml"),
        route_file=str(SCENARIO_DIR / "scenario_medium.rou.xml"),
        use_gui=False,
        num_seconds=300,
        fixed_ts=False,
        single_agent=True,
    )

    observation, info = env.reset(seed=42)

    print("\n--- AMBIENTE SUMO-RL ---")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Numero azioni:", env.action_space.n)
    print("Osservazione iniziale:", observation)
    print("Forma osservazione:", observation.shape)
    print("Dimensione stato:", observation.shape[0])

    action = env.action_space.sample()

    next_observation, reward, terminated, truncated, info = env.step(action)

    print("\n--- PRIMO STEP ---")
    print("Azione scelta:", action)
    print("Nuova osservazione:", next_observation)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    env.close()


if __name__ == "__main__":
    main()