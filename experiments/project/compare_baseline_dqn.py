from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "project"
    / "baseline"
    / "evaluation"
)

DQN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "project"
    / "dqn"
    / "evaluation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "project"
    / "comparisons"
)


def load_episode_files(
    directory: Path,
    pattern: str,
) -> list[pd.DataFrame]:
    """Load all SUMO episode CSV files matching the provided pattern."""

    files = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files found in {directory} with pattern {pattern}"
        )

    return [pd.read_csv(file) for file in files]


def summarize_controller(
    controller_name: str,
    episode_dataframes: list[pd.DataFrame],
) -> dict:
    """Compute average traffic metrics across evaluation episodes."""

    episode_metrics = []

    for dataframe in episode_dataframes:
        required_columns = {
            "system_mean_waiting_time",
            "system_total_stopped",
            "system_mean_speed",
            "system_total_running",
            "system_total_arrived",
        }

        missing_columns = required_columns.difference(dataframe.columns)

        if missing_columns:
            raise KeyError(
                f"Missing columns for {controller_name}: "
                f"{sorted(missing_columns)}"
            )

        final_row = dataframe.iloc[-1]

        episode_metrics.append(
            {
                "controller": controller_name,
                "mean_waiting_time": dataframe[
                    "system_mean_waiting_time"
                ].mean(),
                "mean_stopped_vehicles": dataframe[
                    "system_total_stopped"
                ].mean(),
                "mean_speed": dataframe[
                    "system_mean_speed"
                ].mean(),
                "final_active_vehicles": final_row[
                    "system_total_running"
                ],
                "final_finished_vehicles": final_row[
                    "system_total_arrived"
                ],
            }
        )

    episode_df = pd.DataFrame(episode_metrics)

    return {
        "controller": controller_name,
        "mean_waiting_time": episode_df[
            "mean_waiting_time"
        ].mean(),
        "mean_stopped_vehicles": episode_df[
            "mean_stopped_vehicles"
        ].mean(),
        "mean_speed": episode_df[
            "mean_speed"
        ].mean(),
        "final_active_vehicles": episode_df[
            "final_active_vehicles"
        ].mean(),
        "final_finished_vehicles": episode_df[
            "final_finished_vehicles"
        ].mean(),
    }


def main():
    """Compare fixed-time and DQN evaluation results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_episodes = load_episode_files(
        BASELINE_DIR,
        "sumo_fixed_eval_medium_conn0_ep*.csv",
    )

    dqn_episodes = load_episode_files(
        DQN_DIR,
        "sumo_eval_medium_conn0_ep*.csv",
    )

    baseline_summary = summarize_controller(
        "Fixed-time",
        baseline_episodes,
    )

    dqn_summary = summarize_controller(
        "DQN",
        dqn_episodes,
    )

    comparison = pd.DataFrame(
        [
            baseline_summary,
            dqn_summary,
        ]
    )

    numeric_columns = [
        "mean_waiting_time",
        "mean_stopped_vehicles",
        "mean_speed",
        "final_active_vehicles",
        "final_finished_vehicles",
    ]

    comparison[numeric_columns] = comparison[numeric_columns].round(3)

    output_file = (
        OUTPUT_DIR
        / "baseline_vs_dqn_medium.csv"
    )

    comparison.to_csv(
        output_file,
        index=False,
    )

    print("\nBaseline vs DQN comparison\n")
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison: {output_file}")


if __name__ == "__main__":
    main()