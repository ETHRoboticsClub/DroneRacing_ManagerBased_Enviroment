# NOTE: env.fpv_camera_width/height below have NO effect: the FPV camera is built in the cfg's
# __post_init__ (before Hydra overrides), so it stays 1920x1080. Change the defaults in
# RacingQuadcopterEnvCfg_FIXED_PLAY_FPV instead.
DRONE_MODEL=anemos_urdf TRACK_YAML=configs/tracks/wide.yaml python scripts/skrl/play.py --task Isaac-Racing-Quadcopter-Manager-Fixed-Play-FPV-v0 --enable_cameras env.video_length_s=60.0 env.video_filename=race_demo env.fpv_camera_width=2560 env.fpv_camera_height=1440
