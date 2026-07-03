from pathlib import Path
import xml.etree.ElementTree as ET

SCENARIO_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCENARIO_DIR / "cross.net.xml"
OUTPUT_FILE = SCENARIO_DIR / "cross_with_yellow.net.xml"

YELLOW_DURATION = 3.0  # Duration of the yellow phase in seconds

def build_yellow_state(current_state: str, next_state: str) -> str:

    """Build a safe transition state between two traffic-light phases."""

    if len(current_state) != len(next_state):

        raise ValueError(

            "Traffic-light states have different lengths: "

            f"{len(current_state)} and {len(next_state)}."

        )

    yellow_state = []

    for current_signal, next_signal in zip(

        current_state,

        next_state,

    ):

        # A green signal that is being closed becomes yellow.

        if (

            current_signal in {"G", "g"}

            and next_signal not in {"G", "g"}

        ):

            yellow_state.append("y")

        # A green signal that remains active stays green.

        elif (

            current_signal in {"G", "g"}

            and next_signal in {"G", "g"}

        ):

            yellow_state.append(current_signal)

        # A new green signal must remain red during the transition.

        elif (

            current_signal not in {"G", "g"}

            and next_signal in {"G", "g"}

        ):

            yellow_state.append("r")

        # All other signals remain red.

        else:

            yellow_state.append("r")

    return "".join(yellow_state)

def main():

    """Insert yellow transitions while preserving the traffic-light controller type."""

    if not INPUT_FILE.exists():

        raise FileNotFoundError(

            f"Network file not found: {INPUT_FILE}"

        )

    tree = ET.parse(INPUT_FILE)

    root = tree.getroot()

    traffic_light = root.find("./tlLogic[@id='J3']")

    if traffic_light is None:

        raise ValueError("Traffic light J3 was not found.")

    original_phases = list(

        traffic_light.findall("phase")

    )

    if not original_phases:

        raise ValueError("No phases were found for J3.")

    if any(
        "y" in phase.get("state", "")
        for phase in original_phases
    ):
        raise RuntimeError(
            "The network already contains yellow phases. "
            "No changes were applied."
        )

    # Remove the original phases before rebuilding the program.

    for phase in original_phases:

        traffic_light.remove(phase)

    yellow_phases_added = 0

    for index, current_phase in enumerate(original_phases):

        next_phase = original_phases[

            (index + 1) % len(original_phases)

        ]

        traffic_light.append(current_phase)

        current_state = current_phase.get("state")

        next_state = next_phase.get("state")

        if current_state is None or next_state is None:

            raise ValueError(

                "A traffic-light phase has no state."

            )

        yellow_state = build_yellow_state(

            current_state,

            next_state,

        )

        if "y" in yellow_state:

            yellow_phase = ET.Element(

                "phase",

                {

                    "duration": str(YELLOW_DURATION),

                    "state": yellow_state,

                },

            )

            traffic_light.append(yellow_phase)

            yellow_phases_added += 1

    ET.indent(tree, space="    ")

    tree.write(

        OUTPUT_FILE,

        encoding="UTF-8",

        xml_declaration=True,

    )

    print(f"Original phases: {len(original_phases)}")

    print(f"Yellow phases added: {yellow_phases_added}")

    print(f"Controller type: {traffic_light.get('type')}")

    print(f"Output network: {OUTPUT_FILE}")

if __name__ == "__main__":

    main()