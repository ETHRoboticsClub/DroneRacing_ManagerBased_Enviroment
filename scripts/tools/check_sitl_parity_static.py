"""NON-WORKING (as of 2026-09): static SITL-parity checks written for a monorepo layout.

* REPO_ROOT / MANAGER_ROOT assume this repo lives under <root>/ext/, and the Gazebo SDF and
  Betaflight config paths point outside this repo -> FileNotFoundError.
* The hovering assertions (default drone anemos_sitl, MAXIMUM_THRUST, BetaflightSitlSurrogateActionCfg,
  DR flags False) describe a hovering config that no longer exists.
Kept for reference until the SITL tooling is revived.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
MANAGER_ROOT = REPO_ROOT / "ext" / "DroneRacing_ManagerBased_Enviroment"
SDF_PATH = REPO_ROOT / "sim" / "gazebo" / "models" / "drone_sitl" / "model.sdf"
CALIBRATION_PATH = (
    MANAGER_ROOT
    / "source"
    / "Racing_Quadcopter"
    / "Racing_Quadcopter"
    / "tasks"
    / "manager_based"
    / "hovering_quadcopter"
    / "config"
    / "sitl_actuator_calibration.json"
)
SITL_URDF_PATH = (
    MANAGER_ROOT
    / "source"
    / "Racing_Quadcopter"
    / "Racing_Quadcopter"
    / "assets"
    / "robots"
    / "anemos"
    / "anemos_sitl.urdf"
)
HOVERING_CFG_PATH = (
    MANAGER_ROOT
    / "source"
    / "Racing_Quadcopter"
    / "Racing_Quadcopter"
    / "tasks"
    / "manager_based"
    / "hovering_quadcopter"
    / "hovering_quadcopter_env_cfg.py"
)
SITL_CONFIG_PATH = REPO_ROOT / "docs" / "docs" / "files" / "betaflight_conf" / "sitl_config.txt"


def _sdf_plugin(root: ET.Element) -> ET.Element:
    plugin = root.find(".//plugin[@name='gz::sim::systems::EthrcBetaflightSITLPlugin']")
    assert plugin is not None
    return plugin


def test_calibration_matches_gazebo_sitl_sdf() -> None:
    sdf_root = ET.parse(SDF_PATH).getroot()
    plugin = _sdf_plugin(sdf_root)
    calibration = json.loads(CALIBRATION_PATH.read_text())

    assert calibration["vehicle"]["mass_kg"] == float(sdf_root.findtext(".//link[@name='base_link']/inertial/mass"))
    assert calibration["actuator"]["max_total_thrust_n"] == float(plugin.findtext("maxTotalThrust"))
    assert calibration["actuator"]["yaw_torque_coefficient"] == float(plugin.findtext("yawTorqueCoefficient"))
    assert calibration["actuator"]["yaw_damping_coefficient"] == float(plugin.findtext("yawDampingCoefficient"))

    sdf_rotors = []
    sdf_directions = []
    for rotor in sorted(plugin.findall("rotor"), key=lambda node: int(node.attrib["id"])):
        sdf_rotors.append([float(v) for v in rotor.findtext("position").split()])
        sdf_directions.append(-1.0 if rotor.findtext("turningDirection") == "cw" else 1.0)

    assert calibration["actuator"]["rotor_positions_body_m"] == sdf_rotors
    assert calibration["actuator"]["rotor_directions"] == sdf_directions


def test_sitl_urdf_keeps_only_base_link_mass_physical() -> None:
    root = ET.parse(SITL_URDF_PATH).getroot()
    masses = {
        link.attrib["name"]: float(link.find("inertial/mass").attrib["value"])
        for link in root.findall("link")
        if link.find("inertial/mass") is not None
    }

    assert masses["base_link"] == 0.8351086449771665
    prop_masses = [mass for name, mass in masses.items() if "prop" in name]
    assert len(prop_masses) == 4
    assert all(mass <= 1e-9 for mass in prop_masses)


def test_hovering_defaults_to_sitl_parity_profile() -> None:
    cfg = HOVERING_CFG_PATH.read_text()

    assert 'DRONE_MODEL = _os.environ.get(DRONE_MODEL_ENV_VAR, "anemos_sitl")' in cfg
    assert "MAXIMUM_THRUST = 0.60" in cfg
    assert "enable_corruption: bool = False" in cfg
    assert re.search(r"randomize_mass:\s*bool\s*=\s*False", cfg)
    assert re.search(r"apply_external_disturbances:\s*bool\s*=\s*False", cfg)
    assert "BetaflightSitlSurrogateActionCfg" in cfg


def test_betaflight_sitl_config_uses_linear_actual_rates() -> None:
    cli = SITL_CONFIG_PATH.read_text()

    for expected in (
        "set f_roll = 0",
        "set f_pitch = 0",
        "set f_yaw = 0",
        "set rates_type = ACTUAL",
        "set roll_rc_rate = 50",
        "set roll_srate = 50",
        "set roll_expo = 0",
        "set pitch_rc_rate = 50",
        "set pitch_srate = 50",
        "set pitch_expo = 0",
        "set yaw_rc_rate = 50",
        "set yaw_srate = 50",
        "set yaw_expo = 0",
    ):
        assert expected in cli


def main() -> None:
    test_calibration_matches_gazebo_sitl_sdf()
    test_sitl_urdf_keeps_only_base_link_mass_physical()
    test_hovering_defaults_to_sitl_parity_profile()
    test_betaflight_sitl_config_uses_linear_actual_rates()
    print("SITL parity static checks passed.")


if __name__ == "__main__":
    main()
