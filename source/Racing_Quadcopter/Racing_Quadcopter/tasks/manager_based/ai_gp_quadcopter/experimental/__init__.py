# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AI-GP task, FROZEN post-June experiments (snapshot of 2026-09-26).

Second-order rate action, route-progress curricula and heavy domain randomization, developed
after the competition (Jun 29 – Aug 24). Self-contained: the env runtime, shared MDP terms,
terrains, robot asset, track and agent configs are copies and import nothing from the rest of
``Racing_Quadcopter``. Registered by the parent package with ``AI_GP_VARIANT=experimental``.
Do not modify unless a change to this snapshot is explicitly intended. See ``../README.md``.
"""
