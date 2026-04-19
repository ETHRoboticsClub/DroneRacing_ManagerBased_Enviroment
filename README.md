# Autonomous Drone Racing — IsaacLab RL

![visualization](docs/videos/readme_vis.gif)

Reinforcement learning environment for autonomous drone racing built on [IsaacLab](https://isaac-sim.github.io/IsaacLab/). A quadcopter learns to navigate gate circuits using a manager-based RL pipeline.

---

## Setup

**conda**
```bash
python -m pip install -e source/Racing_Quadcopter
```

**venv**
```bash
pip install -e source/Racing_Quadcopter
```

---

## Training

```bash
# SKRL (recommended)
python scripts/skrl/train.py --task Isaac-Racing-Quadcopter-Manager-v0

# RSL-RL
python scripts/rsl_rl/train.py --task Isaac-Racing-Quadcopter-Manager-v0
```

---

## Evaluation

```bash
# SKRL (recommended)
python scripts/skrl/play.py --task Isaac-Racing-Quadcopter-Manager-v0

# RSL-RL
python scripts/rsl_rl/play.py --task Isaac-Racing-Quadcopter-Manager-v0
```

---

## Other Scripts

```bash
python scripts/random_agent.py   # random action baseline
python scripts/zero_agent.py     # zero action baseline
python scripts/list_envs.py      # list registered environments
```
