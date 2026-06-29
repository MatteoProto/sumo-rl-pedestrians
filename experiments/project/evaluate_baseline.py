from pathlib import Path
import argparse
import csv

import gymnasium as gym
import sumo_rl


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
        description="Evaluate the fixed-time traffic-light baseline."
    )

    parser.add_argument(
        "--scenario",
        choices=["low", "medium", "high"],
        default="medium",
        help="Traffic-demand scenario used during evaluation.",
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
    """Evaluate the fixed-time traffic-light controller."""

    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    if args.seconds <= 0:
        raise ValueError("--seconds must be greater than zero.")

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "project"
        / "baseline"
        / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = (
        output_dir
        / f"evaluation_{args.scenario}.csv"
    )

    sumo_output_prefix = (
        output_dir
        / f"sumo_fixed_eval_{args.scenario}"
    )

    env = gym.make(
        "sumo-rl-v0",
        net_file=str(NET_FILE),
        route_file=str(ROUTE_FILES[args.scenario]),
        use_gui=args.gui,
        num_seconds=args.seconds,
        fixed_ts=True,
        single_agent=True,
        out_csv_name=None,
    )

    print(f"Scenario: {args.scenario}")
    print(f"Evaluation episodes: {args.episodes}")
    print("Controller: fixed-time traffic light")

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
                ],
            )

            writer.writeheader()

            for episode in range(args.episodes):
                evaluation_seed = args.seed + episode

                observation, info = env.reset(
                    seed=evaluation_seed
                )

                terminated = False
                truncated = False

                total_reward = 0.0
                steps = 0

                while not terminated and not truncated:
                    # The action is ignored because fixed_ts=True.
                    action = 0

                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    total_reward += float(reward)
                    steps += 1

                writer.writerow(
                    {
                        "episode": episode + 1,
                        "seed": evaluation_seed,
                        "total_reward": total_reward,
                        "steps": steps,
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
                    f"steps={steps}"
                )

        print("\nBaseline evaluation completed.")
        print(f"Evaluation metrics: {metrics_file}")

    finally:
        env.close()


if __name__ == "__main__":
    main()