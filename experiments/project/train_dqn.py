from pathlib import Path
import argparse
import csv
import random

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
        description="Train a custom DQN agent on the SUMO intersection."
    )

    parser.add_argument(
        "--scenario",
        choices=["low", "medium", "high"],
        default="medium",
        help="Traffic-demand scenario used during training.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of training episodes.",
    )

    parser.add_argument(
        "--seconds",
        type=int,
        default=3600,
        help="Number of simulated seconds per episode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SUMO with the graphical interface.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save a model checkpoint every N episodes.",
    )

    return parser.parse_args()


def linear_epsilon(
    episode: int,
    total_episodes: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
) -> float:
    """Linearly decrease epsilon during training."""

    if total_episodes <= 1:
        return epsilon_end

    progress = episode / (total_episodes - 1)

    return epsilon_start + progress * (
        epsilon_end - epsilon_start
    )


def main():
    """Run the DQN training procedure."""

    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    if args.seconds <= 0:
        raise ValueError("--seconds must be greater than zero.")

    if args.checkpoint_every <= 0:
        raise ValueError(
            "--checkpoint-every must be greater than zero."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    output_dir = PROJECT_ROOT / "outputs" / "project" / "dqn"
    model_dir = PROJECT_ROOT / "models" / "project" / "dqn"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = (
        f"{args.scenario}_"
        f"ep{args.episodes}_"
        f"sec{args.seconds}_"
        f"seed{args.seed}"
    )

    metrics_file = (
        output_dir
        / f"training_{experiment_name}.csv"
    )

    sumo_output_prefix = (
        output_dir
        / f"sumo_{experiment_name}"
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

    print(f"Device: {device}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.episodes}")
    print(f"Seconds per episode: {args.seconds}")
    print(f"Seed: {args.seed}")

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=50_000,
        batch_size=64,
        target_update_frequency=500,
        seed=args.seed,
    )

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
                    "epsilon",
                    "total_reward",
                    "mean_loss",
                    "steps",
                ],
            )

            writer.writeheader()

            for episode in range(args.episodes):
                state, info = env.reset(
                    seed=args.seed + episode
                )

                terminated = False
                truncated = False

                total_reward = 0.0
                losses = []
                steps = 0

                epsilon = linear_epsilon(
                    episode=episode,
                    total_episodes=args.episodes,
                )

                while not terminated and not truncated:
                    action = agent.select_action(
                        state=state,
                        epsilon=epsilon,
                    )

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    done = terminated or truncated

                    agent.store_transition(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                    )

                    loss = agent.train_step()

                    if loss is not None:
                        losses.append(loss)

                    state = next_state
                    total_reward += float(reward)
                    steps += 1

                mean_loss = (
                    float(np.mean(losses))
                    if losses
                    else 0.0
                )

                writer.writerow(
                    {
                        "episode": episode + 1,
                        "epsilon": epsilon,
                        "total_reward": total_reward,
                        "mean_loss": mean_loss,
                        "steps": steps,
                    }
                )

                csv_file.flush()

                # Save SUMO metrics for the current episode.
                env.unwrapped.save_csv(
                    str(sumo_output_prefix),
                    episode=episode + 1,
                )

                print(
                    f"Episode {episode + 1}/{args.episodes} | "
                    f"epsilon={epsilon:.3f} | "
                    f"reward={total_reward:.3f} | "
                    f"loss={mean_loss:.6f} | "
                    f"steps={steps}"
                )

                # Save the model periodically during training.
                if (
                    episode + 1
                ) % args.checkpoint_every == 0:
                    checkpoint_path = (
                        model_dir
                        / (
                            f"dqn_{experiment_name}_"
                            f"checkpoint_ep{episode + 1}.pt"
                        )
                    )

                    agent.save(str(checkpoint_path))

                    print(
                        "Checkpoint saved: "
                        f"{checkpoint_path.name}"
                    )

        model_path = (
            model_dir
            / f"dqn_{experiment_name}.pt"
        )

        agent.save(str(model_path))

        print("\nTraining completed.")
        print(f"Training metrics: {metrics_file}")
        print(f"Final model: {model_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()