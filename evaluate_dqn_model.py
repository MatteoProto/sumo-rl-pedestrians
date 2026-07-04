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

# Modifica questo percorso se il modello si trova in un'altra cartella.
MODEL_FILE = PROJECT_DIR / "outputs" / "2way-single-intersection" / "dqn_baseline_model.zip"

NUM_SECONDS = 300
DELTA_TIME = 5
NUM_EPISODES = 30


# ============================================================
# Creazione dell'ambiente
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
# Esecuzione di un episodio
# ============================================================

def run_episode(
    env: SumoEnvironment,
    model: DQN | None,
    random_policy: bool = False,
) -> dict:
    observation, info = env.reset()

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0
    actions = []

    while not (terminated or truncated):
        if random_policy:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

        # Converte eventuali valori NumPy in un intero normale.
        action = int(np.asarray(action).item())
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

    action_counts = {
        action: actions.count(action)
        for action in sorted(set(actions))
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
            info.get(
                "system_mean_waiting_time_pedestrians",
                0.0,
            )
        ),
        "pedestrian_total_waiting_time": float(
            info.get(
                "system_total_waiting_time_pedestrians",
                0.0,
            )
        ),
        "arrived_vehicles": int(
            info.get("system_total_arrived", 0)
        ),
        "arrived_pedestrians": int(
            info.get(
                "system_total_arrived_pedestrians",
                0,
            )
        ),
        "teleported": int(
            info.get("system_total_teleported", 0)
        ),
        "action_counts": action_counts,
    }


# ============================================================
# Valutazione su più episodi
# ============================================================

def evaluate_policy(
    model: DQN | None,
    policy_name: str,
    random_policy: bool = False,
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
                random_policy=random_policy,
            )
            results.append(result)

            print(
                f"Episodio {episode:2d} | "
                f"reward={result['reward']:.3f} | "
                f"attesa media={result['mean_waiting_time']:.3f} | "
                f"attesa pedoni={result['pedestrian_mean_waiting_time']:.3f} | "
                f"arrivi={result['arrived_vehicles']} | "
                f"teleport={result['teleported']}"
            )

        finally:
            env.close()

    return results


# ============================================================
# Riepilogo
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
        "Veicoli arrivati": "arrived_vehicles",
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
# Confronto finale
# ============================================================

def compare_results(
    dqn_results: list[dict],
    random_results: list[dict],
) -> None:
    print("\n" + "=" * 60)
    print("CONFRONTO DQN VS RANDOM")
    print("=" * 60)

    comparisons = {
        "Reward": "reward",
        "Attesa media veicoli": "mean_waiting_time",
        "Attesa media pedoni": "pedestrian_mean_waiting_time",
        "Veicoli arrivati": "arrived_vehicles",
        "Pedoni arrivati": "arrived_pedestrians",
        "Teletrasporti": "teleported",
    }

    for label, key in comparisons.items():
        dqn_mean = np.mean(
            [result[key] for result in dqn_results]
        )
        random_mean = np.mean(
            [result[key] for result in random_results]
        )

        print(
            f"{label:25s} | "
            f"DQN: {dqn_mean:10.3f} | "
            f"Random: {random_mean:10.3f}"
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

    print(f"Caricamento modello: {MODEL_FILE}")

    model = DQN.load(
        str(MODEL_FILE),
        device="cpu",
    )

    dqn_results = evaluate_policy(
        model=model,
        policy_name="DQN",
        random_policy=False,
    )

    random_results = evaluate_policy(
        model=None,
        policy_name="Random",
        random_policy=True,
    )

    print_summary("DQN", dqn_results)
    print_summary("Random", random_results)

    compare_results(
        dqn_results=dqn_results,
        random_results=random_results,
    )


if __name__ == "__main__":
    main()