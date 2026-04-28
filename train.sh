
TASK=Isaac-Racing-Quadcopter-Manager-v0

NUM_ENVS=4096

VIDEO_LENGTH=400
VIDEO_INTERVAL=3000


# check if --headless is in the arguments
if [[ "$@" == *"--headless"* ]]; then
    HEADLESS="--headless"
else
    HEADLESS=""
fi

# check if --video is in the arguments
if [[ "$@" == *"--video"* ]]; then
    VIDEO="--video"
else
    VIDEO=""
fi

python scripts/skrl/train.py \
    --task $TASK \
    --num_envs $NUM_ENVS \
    $HEADLESS \
    $VIDEO \
    --video_length $VIDEO_LENGTH \
    --video_interval $VIDEO_INTERVAL
