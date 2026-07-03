from pathlib import Path
from sumo_rl import SumoEnvironment


ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "sumo_rl" / "nets" / "nostri"

NET_FILE = SCENARIO_DIR / "cross.net.xml"

ROUTE_FILE = SCENARIO_DIR / "cross_flows.rou.xml"


def get_waiting_time_breakdown(traffic_signal):
    """Return vehicle, pedestrian, and total waiting time for the signal lanes."""

    vehicle_waiting_time = 0.0
    pedestrian_waiting_time = 0.0

    for lane in traffic_signal.lanes:
        if traffic_signal.lanes_type[lane] == "pedestrian":
            pedestrian_waiting_time += (
                traffic_signal.get_lane_accumulated_waiting_time_pedestrians(lane)
            )
        else:
            vehicle_waiting_time += (
                traffic_signal.get_lane_accumulated_waiting_time_vehicles(lane)
            )

    total_waiting_time = vehicle_waiting_time + pedestrian_waiting_time

    return vehicle_waiting_time, pedestrian_waiting_time, total_waiting_time


def main():
    """Run a short random-action test on the custom environment."""

    if not NET_FILE.exists():
        raise FileNotFoundError(f"Network file not found: {NET_FILE}")

    if not ROUTE_FILE.exists():
        raise FileNotFoundError(f"Route file not found: {ROUTE_FILE}")

    env = SumoEnvironment(
        net_file=str(NET_FILE),
        route_file=str(ROUTE_FILE),
        use_gui=False,
        num_seconds=300,
        single_agent=True,
    )

    try:
        observation, info = env.reset(seed=42)

        print(f"Observation type: {type(observation)}")
        print(f"Observation shape: {observation.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Initial info: {info}")

        traffic_signal = env.traffic_signals["J3"]

        print("\nTraffic-light phases:")
        for index, phase in enumerate(traffic_signal.all_phases):
            print(
                f"Phase {index}: "
                f"duration={phase.duration}, "
                f"state={phase.state}"
            )

        print("\nWaiting-time breakdown during simulation:")

        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not terminated and not truncated:
            action = env.action_space.sample()

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += float(reward)
            steps += 1

            if steps % 10 == 0 or terminated or truncated:
                (
                    vehicle_waiting_time,
                    pedestrian_waiting_time,
                    total_waiting_time,
                ) = get_waiting_time_breakdown(traffic_signal)

                print(
                    f"Step {steps:>3} | "
                    f"simulation_time={info['step']:.0f}s | "
                    f"vehicle_wait={vehicle_waiting_time:.2f} | "
                    f"pedestrian_wait={pedestrian_waiting_time:.2f} | "
                    f"total_wait={total_waiting_time:.2f}"
                )

        print("\nEnvironment test completed.")
        print(f"Steps: {steps}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Final observation shape: {observation.shape}")
        print(f"Final info: {info}")

    finally:
        env.close()


if __name__ == "__main__":
    main()