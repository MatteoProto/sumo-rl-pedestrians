from pathlib import Path
import argparse
import csv

import gymnasium as gym
import numpy as np
import torch
import sumo_rl

from sumo_rl.project.dqn_agent import DQNAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_DIR = (
    PROJECT_ROOT
    / "sumo_rl"
    / "nets"
    / "project_intersection_v1"
)

NET_FILE = SCENARIO_DIR / "project_intersection.net.xml"

ROUTE_FILES = {
    "low": SCENARIO_DIR / "scenario_low.rou.xml",
    "medium": SCENARIO_DIR / "scenario_medium.rou.xml",
    "high": SCENARIO_DIR / "scenario_high.rou.xml",
}


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on the SUMO intersection."
    )

    parser.add_argument(
        "--scenario",
        choices=["low", "medium", "high"],
        default="medium",
        help="Traffic-demand scenario used during evaluation.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the trained DQN model.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--seconds",
        type=int,
        default=600,
        help="Number of simulated seconds per episode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Initial random seed used during evaluation.",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SUMO with the graphical interface.",
    )

    return parser.parse_args()


def main():
    """Evaluate a trained DQN policy without exploration."""

    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    if args.seconds <= 0:
        raise ValueError("--seconds must be greater than zero.")

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "project"
        / "dqn"
        / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model is None:
        model_path = (
            PROJECT_ROOT
            / "models"
            / "project"
            / "dqn"
            / f"dqn_{args.scenario}_seed42.pt"
        )
    else:
        model_path = Path(args.model).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    metrics_file = (
        output_dir
        / f"evaluation_{args.scenario}.csv"
    )

    sumo_output_prefix = (
        output_dir
        / f"sumo_eval_{args.scenario}"
    )

    env = gym.make(
        "sumo-rl-v0",
        net_file=str(NET_FILE),
        route_file=str(ROUTE_FILES[args.scenario]),
        use_gui=args.gui,
        num_seconds=args.seconds,
        fixed_ts=False,
        single_agent=True,
        out_csv_name=None,
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device("cpu")

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        seed=42,
    )

    agent.load(str(model_path))
    agent.policy_network.eval()

    print(f"Model: {model_path}")
    print(f"Scenario: {args.scenario}")
    print(f"Evaluation episodes: {args.episodes}")
    print("Exploration epsilon: 0.0")

    try:
        with metrics_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "episode",
                    "seed",
                    "total_reward",
                    "steps",
                    "action_0_count",
                    "action_1_count",
                    "action_2_count",
                    "action_3_count",
                ],
            )

            writer.writeheader()

            for episode in range(args.episodes):
                evaluation_seed = args.seed + episode

                state, info = env.reset(seed=evaluation_seed)

                terminated = False
                truncated = False

                total_reward = 0.0
                steps = 0

                # Count how many times each action is selected.
                action_counts = np.zeros(
                    action_size,
                    dtype=np.int64,
                )

                while not terminated and not truncated:
                    action = agent.select_action(
                        state=state,
                        epsilon=0.0,
                    )

                    action_counts[action] += 1

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    state = next_state
                    total_reward += float(reward)
                    steps += 1

                writer.writerow(
                    {
                        "episode": episode + 1,
                        "seed": evaluation_seed,
                        "total_reward": total_reward,
                        "steps": steps,
                        "action_0_count": int(action_counts[0]),
                        "action_1_count": int(action_counts[1]),
                        "action_2_count": int(action_counts[2]),
                        "action_3_count": int(action_counts[3]),
                    }
                )

                csv_file.flush()

                # Save SUMO traffic metrics for the current episode.
                env.unwrapped.save_csv(
                    str(sumo_output_prefix),
                    episode=episode + 1,
                )

                print(
                    f"Evaluation episode "
                    f"{episode + 1}/{args.episodes} | "
                    f"seed={evaluation_seed} | "
                    f"reward={total_reward:.3f} | "
                    f"steps={steps} | "
                    f"actions={action_counts.tolist()}"
                )

        print("\nEvaluation completed.")
        print(f"Evaluation metrics: {metrics_file}")

    finally:
        env.close()


if __name__ == "__main__":
    main()