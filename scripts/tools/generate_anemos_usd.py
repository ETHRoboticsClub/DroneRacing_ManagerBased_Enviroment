# generate_drone_usd.py
"""Generate a correct floating-base USD from the URDF using proven settings."""
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters.urdf_converter import UrdfConverter
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANEMOS_DIR = os.path.join(_THIS_DIR, "..", "..", "source", "Racing_Quadcopter", "Racing_Quadcopter", "assets", "robots", "anemos")
_URDF_PATH = os.path.join(_ANEMOS_DIR, "anemos.urdf")
_USD_NAME = "anemos.usd"

cfg = UrdfConverterCfg(
    asset_path=_URDF_PATH,
    usd_dir=_ANEMOS_DIR,
    usd_file_name=_USD_NAME,
    force_usd_conversion=True,
    fix_base=False,
    merge_fixed_joints=False,
    replace_cylinders_with_capsules=False,
    make_instanceable=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        target_type="none",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.0,
        ),
    ),
)

converter = UrdfConverter(cfg)
print(f"USD generated at: {converter.usd_path}")

simulation_app.close()