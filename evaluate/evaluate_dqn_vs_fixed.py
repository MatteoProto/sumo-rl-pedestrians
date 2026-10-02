from pathlib import Path
import numpy as np

from stable_baselines3 import DQN
from sumo_rl import SumoEnvironment


# ============================================================
# Percorsi
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = PROJECT_DIR / "sumo_rl" / "nets" / "nostri"

NET_FILE = SCENARIO_DIR / "cross.net.xml"
ROUTE_FILE = SCENARIO_DIR / "cross_flows.rou.xml"

MODEL_FILE = (
    PROJECT_DIR
    / "outputs"
    / "nostri_7phases"
    / "dqn_200000_model.zip"
)

NUM_SECONDS = 300
DELTA_TIME = 5
NUM_EPISODES = 30
FIXED_HOLD_STEPS = 4


# ============================================================
# Creazione ambiente
# ============================================================

def create_environment() -> SumoEnvironment:
    return SumoEnvironment(
        net_file=str(NET_FILE),
        route_file=str(ROUTE_FILE),
        use_gui=False,
        num_seconds=NUM_SECONDS,
        single_agent=True,
        delta_time=DELTA_TIME,
        yellow_time=2,
        min_green=5,
    )


# ============================================================
# Esecuzione episodio
# ============================================================

def run_episode(
    env: SumoEnvironment,
    model: DQN | None,
    policy_type: str,
) -> dict:
    observation, info = env.reset()

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0
    actions = []

    while not (terminated or truncated):
        if policy_type == "dqn":
            action, _ = model.predict(
                observation,
                deterministic=True,
            )
            action = int(np.asarray(action).item())

        elif policy_type == "fixed":
            # Semaforo tradizionale ciclico:
            # 0 -> 1 -> 2 -> ... -> ultima fase -> 0
            action = (steps // FIXED_HOLD_STEPS) % env.action_space.n

        else:
            raise ValueError(
                f"Policy non riconosciuta: {policy_type}"
            )

        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

    action_counts = {
        action: actions.count(action)
        for action in range(env.action_space.n)
    }

    return {
        "reward": total_reward,
        "steps": steps,
        "mean_waiting_time": float(
            info.get("system_mean_waiting_time", 0.0)
        ),
        "total_waiting_time": float(
            info.get("system_total_waiting_time", 0.0)
        ),
        "pedestrian_mean_waiting_time": float(
            info.get("system_mean_waiting_time_pedestrians", 0.0)
        ),
        "pedestrian_total_waiting_time": float(
            info.get("system_total_waiting_time_pedestrians", 0.0)
        ),
        "arrived_vehicles": int(
            info.get("system_total_arrived", 0)
        ),
        "departed_vehicles": int(
            info.get("system_total_departed", 0)
        ),
        "arrived_pedestrians": int(
            info.get("system_total_arrived_pedestrians", 0)
        ),
        "departed_pedestrians": int(
            info.get("system_total_departed_pedestrians", 0)
        ),
        "teleported": int(
            info.get("system_total_teleported", 0)
        ),
        "action_counts": action_counts,
    }


# ============================================================
# Valutazione policy
# ============================================================

def evaluate_policy(
    model: DQN | None,
    policy_name: str,
    policy_type: str,
) -> list[dict]:
    results = []

    print(f"\n{'=' * 60}")
    print(f"Valutazione: {policy_name}")
    print(f"{'=' * 60}")

    for episode in range(1, NUM_EPISODES + 1):
        env = create_environment()

        try:
            result = run_episode(
                env=env,
                model=model,
                policy_type=policy_type,
            )
            results.append(result)

            print(
                f"Episodio {episode:2d} | "
                f"reward={result['reward']:.3f} | "
                f"attesa veicoli={result['mean_waiting_time']:.3f} | "
                f"attesa pedoni={result['pedestrian_mean_waiting_time']:.3f} | "
                f"veicoli arrivati={result['arrived_vehicles']} | "
                f"pedoni arrivati={result['arrived_pedestrians']} | "
                f"teleport={result['teleported']}"
            )

        finally:
            env.close()

    return results


# ============================================================
# Riepilogo risultati
# ============================================================

def print_summary(
    policy_name: str,
    results: list[dict],
) -> None:
    print(f"\n--- Risultati medi: {policy_name} ---")

    metrics = {
        "Reward totale": "reward",
        "Attesa media veicoli": "mean_waiting_time",
        "Attesa totale veicoli": "total_waiting_time",
        "Attesa media pedoni": "pedestrian_mean_waiting_time",
        "Attesa totale pedoni": "pedestrian_total_waiting_time",
        "Veicoli partiti": "departed_vehicles",
        "Veicoli arrivati": "arrived_vehicles",
        "Pedoni partiti": "departed_pedestrians",
        "Pedoni arrivati": "arrived_pedestrians",
        "Veicoli teletrasportati": "teleported",
    }

    for label, key in metrics.items():
        values = [result[key] for result in results]

        print(
            f"{label}: "
            f"{np.mean(values):.3f} "
            f"± {np.std(values):.3f}"
        )


# ============================================================
# Distribuzione azioni
# ============================================================

def print_action_distribution(
    policy_name: str,
    results: list[dict],
) -> None:
    print(f"\n--- Azioni medie scelte: {policy_name} ---")

    all_actions = sorted(
        {
            action
            for result in results
            for action in result["action_counts"]
        }
    )

    for action in all_actions:
        values = [
            result["action_counts"].get(action, 0)
            for result in results
        ]

        print(
            f"Azione {action}: "
            f"{np.mean(values):.2f} "
            f"± {np.std(values):.2f}"
        )


# ============================================================
# Confronto finale
# ============================================================

def compare_results(
    dqn_results: list[dict],
    fixed_results: list[dict],
) -> None:
    print("\n" + "=" * 60)
    print("CONFRONTO DQN VS FIXED TRAFFIC LIGHT")
    print("=" * 60)

    comparisons = {
        "Reward": "reward",
        "Attesa media veicoli": "mean_waiting_time",
        "Attesa totale veicoli": "total_waiting_time",
        "Attesa media pedoni": "pedestrian_mean_waiting_time",
        "Attesa totale pedoni": "pedestrian_total_waiting_time",
        "Veicoli arrivati": "arrived_vehicles",
        "Pedoni arrivati": "arrived_pedestrians",
        "Teletrasporti": "teleported",
    }

    for label, key in comparisons.items():
        dqn_mean = np.mean(
            [result[key] for result in dqn_results]
        )
        fixed_mean = np.mean(
            [result[key] for result in fixed_results]
        )

        print(
            f"{label:25s} | "
            f"DQN: {dqn_mean:10.3f} | "
            f"Fixed: {fixed_mean:10.3f}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not NET_FILE.exists():
        raise FileNotFoundError(
            f"File della rete non trovato: {NET_FILE}"
        )

    if not ROUTE_FILE.exists():
        raise FileNotFoundError(
            f"File delle route non trovato: {ROUTE_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Modello DQN non trovato: {MODEL_FILE}"
        )

    print(f"Rete: {NET_FILE}")
    print(f"Route: {ROUTE_FILE}")
    print(f"Modello DQN: {MODEL_FILE}")
    print(f"Episodi: {NUM_EPISODES}")
    print(f"Durata simulazione: {NUM_SECONDS} secondi")
    print(f"Delta time: {DELTA_TIME} secondi")

    model = DQN.load(
        str(MODEL_FILE),
        device="cpu",
    )

    dqn_results = evaluate_policy(
        model=model,
        policy_name="DQN",
        policy_type="dqn",
    )

    fixed_results = evaluate_policy(
        model=None,
        policy_name="Fixed traffic light",
        policy_type="fixed",
    )

    print_summary("DQN", dqn_results)
    print_summary("Fixed traffic light", fixed_results)

    print_action_distribution("DQN", dqn_results)
    print_action_distribution("Fixed traffic light", fixed_results)

    compare_results(
        dqn_results=dqn_results,
        fixed_results=fixed_results,
    )


if __name__ == "__main__":
    main()