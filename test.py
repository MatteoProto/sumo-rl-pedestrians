from pathlib import Path
from sumo_rl import SumoEnvironment

ROOT = Path(__file__).parent

net = ROOT / "sumo_rl" / "nets" / "single-intersection" / "single-intersection.net.xml"

route = ROOT / "sumo_rl" / "nets" / "single-intersection" / "single-intersection.rou.xml"

env = SumoEnvironment(
    net_file=str(net),
    route_file=str(route),
    use_gui=True,
    num_seconds=1000,
)

obs, info = env.reset()

done = False

while not done:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

env.close()