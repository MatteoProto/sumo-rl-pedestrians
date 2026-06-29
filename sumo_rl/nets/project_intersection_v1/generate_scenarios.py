from pathlib import Path
import copy
import xml.etree.ElementTree as ET


BASE_DIR = Path(__file__).resolve().parent

BASE_ROUTE_FILE = BASE_DIR / "scenario_medium.rou.xml"
BASE_CONFIG_FILE = BASE_DIR / "project_intersection.sumocfg"

SCENARIOS = {
    "low": 0.50,
    "medium": 1.00,
    "high": 1.60,
}


def create_route_scenario(name: str, multiplier: float) -> None:
    """Create a route file by scaling every vehsPerHour value."""

    tree = ET.parse(BASE_ROUTE_FILE)
    root = tree.getroot()

    for flow in root.findall("flow"):
        value = flow.get("vehsPerHour")

        if value is not None:
            original_value = float(value)
            new_value = max(1, round(original_value * multiplier))
            flow.set("vehsPerHour", str(new_value))

    output_file = BASE_DIR / f"scenario_{name}.rou.xml"

    tree.write(
        output_file,
        encoding="UTF-8",
        xml_declaration=True,
    )

    print(f"Creato: {output_file.name}")


def create_config_scenario(name: str) -> None:
    """Create the SUMO configuration for the selected traffic scenario."""

    tree = ET.parse(BASE_CONFIG_FILE)
    root = tree.getroot()

    route_element = root.find("./input/route-files")

    if route_element is None:
        raise RuntimeError(
            "Elemento <route-files> non trovato in project_intersection.sumocfg"
        )

    route_element.set("value", f"scenario_{name}.rou.xml")

    output_element = root.find("./output")

    if output_element is not None:
        for child in output_element:
            current_value = child.get("value")

            if current_value:
                filename = Path(current_value).name
                child.set("value", f"outputs/{name}_{filename}")

    output_file = BASE_DIR / f"scenario_{name}.sumocfg"

    tree.write(
        output_file,
        encoding="UTF-8",
        xml_declaration=True,
    )

    print(f"Creato: {output_file.name}")


def main() -> None:
    if not BASE_ROUTE_FILE.exists():
        raise FileNotFoundError(
            f"File non trovato: {BASE_ROUTE_FILE}"
        )

    if not BASE_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"File non trovato: {BASE_CONFIG_FILE}"
        )

    for name, multiplier in SCENARIOS.items():
        create_route_scenario(name, multiplier)
        create_config_scenario(name)


if __name__ == "__main__":
    main()