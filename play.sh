
TASK=Isaac-Racing-Quadcopter-Competition-RNN-Play-v0
DRONE_MODEL=anemos_urdf

RL_LIBRARY=rsl_rl

HEADLESS=0

NUM_ENVS=32

VIDEO=0
VIDEO_LENGTH=1000
VIDEO_INTERVAL=4000


DRONE_MODEL=$DRONE_MODEL python scripts/$RL_LIBRARY/play.py \
    --task $TASK \
    --num_envs $NUM_ENVS \
    $([ $HEADLESS -eq 1 ] && echo "--headless") \
    $([ $VIDEO -eq 1 ] && echo "--video --video_length ${VIDEO_LENGTH} --video_interval ${VIDEO_INTERVAL}") \
    $@