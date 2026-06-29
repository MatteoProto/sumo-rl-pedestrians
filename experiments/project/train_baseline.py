from pathlib import Path
import argparse

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
    parser = argparse.ArgumentParser(
        description="Run fixed-time baseline on the custom SUMO scenario."
    )

    parser.add_argument(
        "--scenario",
        choices=["low", "medium", "high"],
        default="medium",
    )

    parser.add_argument(
        "--seconds",
        type=int,
        default=3600,
    )

    parser.add_argument(
        "--gui",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = PROJECT_ROOT / "outputs" / "project" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = output_dir / f"fixed_{args.scenario}"

    env = gym.make(
        "sumo-rl-v0",
        net_file=str(NET_FILE),
        route_file=str(ROUTE_FILES[args.scenario]),
        out_csv_name=str(output_prefix),
        use_gui=args.gui,
        num_seconds=args.seconds,
        fixed_ts=True,
        single_agent=True,
    )

    observation, info = env.reset()

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = 0

        observation, reward, terminated, truncated, info = env.step(action)

    env.unwrapped.save_csv(
        str(output_prefix),
        episode=1,
    )

    env.close()

    print(
        f"Baseline completato: scenario={args.scenario}, "
        f"durata={args.seconds}s"
    )


if __name__ == "__main__":
    main()