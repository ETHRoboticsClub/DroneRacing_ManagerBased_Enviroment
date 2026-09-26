# Observation Space — MonoRace G&CNet (24D), with Swift reference

This README documents the observation vector of the **MonoRace** policy
(Bahnam, Ferede et al., arXiv:2601.15222) and contrasts it with **Swift**
(Kaufmann et al., *Nature* 2023). The MonoRace spec below is the version
that has been **validated against the public source code** of MonoRace's
reference [24] (Ferede et al., *One Net to Rule Them All*, ICRA 2025), at
[`tudelft/optimal_quad_control_RL @ icra2025`](https://github.com/tudelft/optimal_quad_control_RL/tree/icra2025),
file `quad_race_env.py` (function `Quadcopter3DGates.update_states_gate`).
MonoRace explicitly states it uses the same observation formulation as that
prior work.

---

## 1. Frames

Three frames, as used by the **MonoRace / ICRA-2025 reference code** (NED-style, z points
**down**):

- **World frame `W`** — inertial. NED-down confirmed by the reference env code:
  `pos[2] > 0` ⇒ ground collision, `pos[2] < −7` ⇒ ceiling.
- **Body frame `B`** — standard FRD aircraft body axes.
  Body angular rates `(p, q, r)` and propeller speeds live here.
- **Gate frame `G_i`** — origin at the geometric center of gate `i`,
  obtained from `W` by a **pure yaw rotation about the world z-axis** by
  `yaw_{g_i}`. The gate's x-axis is its outward normal (passing direction).

> **NOTE — this repo's frames differ from the reference paper's.** `Racing_Quadcopter` is an
> IsaacLab environment: the simulator's world frame is **Z-up** (not NED-down) and its body frame
> is **FLU** (forward-left-up), not FRD. `mdp/observations.py`'s `mono_race_*` functions operate
> directly in the sim's native Z-up/FLU frames — there is no NED/FRD conversion anywhere in this
> codebase. Read the rest of this section as describing the *reference* MonoRace/ICRA code's own
> convention, not this repo's; the "gate frame" construction (yaw-only rotation, or full SO(3) for
> our tilted gates per the note below) is the part that carries over.

Rotation conventions (from `quad_race_env.py`):
- Euler angles `λ = (φ, θ, ψ)` are **ZYX intrinsic**,
  body→world: `R = R_z(ψ) · R_y(θ) · R_x(φ)`.
- The internal simulator uses quaternions; the policy is fed Euler angles.

> **NOTE — gates with roll/pitch in our environment.**
> MonoRace assumes gates are upright (roll = pitch = 0), so `G_i` is
> yaw-only and roll/pitch are frame-invariant between `W` and `G_i`.
> **Our environment has gates with non-zero roll and pitch**, so the
> "gate frame" must be a full SO(3) rotation `R_{W→G_i}`, not a 2D yaw
> rotation. Concretely:
> - Build `R_{W→G_i}` from the gate's full orientation (e.g. quaternion
>   or `R_z(ψ_g)·R_y(θ_g)·R_x(φ_g)` transposed).
> - Rotate **both** `pos − pos_{g_i}` and `vel` by `R_{W→G_i}`.
> - The drone attitude in `G_i` is `R_{B→G_i} = R_{W→G_i} · R_{B→W}`;
>   convert that back to ZYX Euler to fill `(φ_{g_i}, θ_{g_i}, ψ_{g_i})`
>   — none of these three components stays unchanged from `W` anymore.
> - Likewise, the relative-pose terms for future gates must use the full
>   3D rotation, not just yaw subtraction.

---

## 2. Layout (24 components)

Index map. Total = `16 + 4 · gates_ahead`. MonoRace ⇒ `gates_ahead = 2` ⇒ 24.

| idx     | symbol                       | dim | frame   | meaning                                                                                  |
|---------|------------------------------|-----|---------|------------------------------------------------------------------------------------------|
| 0..2    | `p^{g_i}`                    | 3   | `G_i`   | drone position relative to gate `i` center, expressed in `G_i`                           |
| 3..5    | `v^{g_i}`                    | 3   | `G_i`   | drone velocity expressed in `G_i` (rotate only, do **not** translate)                    |
| 6       | `φ`                          | 1   | B→W (= B→G_i in MonoRace) | roll (ZYX intrinsic)                                                |
| 7       | `θ`                          | 1   | B→W (= B→G_i in MonoRace) | pitch                                                              |
| 8       | `ψ_{g_i}`                    | 1   | B→G_i   | yaw of body relative to `G_i`, wrapped to `(−π, π]`                                       |
| 9..11   | `Ω = (p, q, r)`              | 3   | **B**   | body angular rates expressed in **body frame** (see Inconsistency #1)                     |
| 12..15  | `ω = (ω_1, ω_2, ω_3, ω_4)`   | 4   | —       | **propeller speeds** in rad/s (see Inconsistency #2)                                      |
| 16..18  | `p^{g_i}_{g_{i+1}}`          | 3   | `G_i`   | center of gate `i+1` relative to gate `i`, expressed in `G_i`                            |
| 19      | `ψ^{g_i}_{g_{i+1}}`          | 1   | G_i→G_{i+1} | yaw of gate `i+1` relative to `G_i`, wrapped to `(−π, π]`                              |
| 20..22  | `p^{g_i}_{g_{i+2}}`          | 3   | `G_i`   | center of gate `i+2` relative to gate `i`, expressed in `G_i`                            |
| 23      | `ψ^{g_i}_{g_{i+2}}`          | 1   | G_i→G_{i+2} | yaw of gate `i+2` relative to `G_i`, wrapped                                            |

**Env implementation:** `mono_race_future_gates_pose_in_current_gate_frame` in `mdp/observations.py` generalises this block to `k` future gates (`k = 2` in the table above, from `cfg.observation.mono_race_future_gates_pose_ahead`): for each `m ∈ {1,…,k}` it outputs **6 values** — `p^{g_i}_{g_{i+m}}` (3, position in `G_i`) followed by the **full relative attitude** `(φ, θ, ψ)` (3, ZYX Euler of `q_{G_i→W}^{-1} ⊗ q_{G_{i+m}→W}`) — not just a single yaw scalar, since gates in this environment can have non-zero roll/pitch (see the note above). There is no `params`/`num_future_gates` override on the `ObsTerm`; `k` is read only from the cfg field.

Future-gate indices **wrap modulo the env's gate count** (`% env._gate_counts`) rather than being
zero-padded past the last gate — every track behaves as a closed loop for this observation.

**This repo's `mono_race` observation vector is not 24D.** `OBSERVATION_SPACE=mono_race`
(`MonoRaceObservationsCfg` in `racing_quadcopter_env_cfg.py`) concatenates:
`mono_race_gate_position_velocity_gate_frame` (6, `k=1` from `mono_race_num_next_gates`) +
`mono_race_gate_orientations` (3) + `mono_race_future_gates_pose_in_current_gate_frame` (12,
`k=2`) + `base_ang_vel` (3, scaled) + `motor_speeds_obs` (4) = **28**, plus 2 more
(`speed_limits`) when `speed_limits.enabled=True` (RACING_REWARD_PRESET=`current`) → **30**.

---

## 3. Inconsistencies in the MonoRace paper (vs. its own code)

1. **`Ω` and `ω` labels are swapped.** The paper's text reads
   "`Ω` the angular velocity in the world frame, `ω` the angular velocity
   in the body frame." In the code, indices 9..11 are body rates `(p, q, r)`
   in the **body frame**, and indices 12..15 are the four motor speeds.
   The ICRA-2025 paper writes this correctly: "`Ω` are the body rates and
   `ω` are the propeller speed in rad/s".
2. **`ω` is not an angular-velocity vector.** It is a 4-vector of
   per-rotor speeds (rad/s), giving 4 scalars, not a 3D body angular
   velocity.
3. **`Φ_{g_i} = (φ, θ, ψ)` is misleading.** A yaw-only frame change
   cannot affect roll or pitch, so in MonoRace's setup `φ` and `θ` are
   identical in `W` and `G_i`. Only `ψ` is rotated. The subscript `g_i`
   is therefore meaningful only on `ψ`.
   *(In our environment, where gates have full 3D orientation, this
   inaccuracy disappears: all three Euler components legitimately change
   when transformed into `G_i`.)*
4. **24 vs. 19 mismatch in the equation.** The paper writes "24
   observations" but the displayed equation enumerates only 19 scalars
   (one future-gate block). The only self-consistent reading is
   `gates_ahead = 2`, i.e. the next *and* the next-next gate are
   included, giving `16 + 2·4 = 24`. The prose ("current and future
   gates", plural) supports this. The ICRA prior work uses
   `gates_ahead = 1` ⇒ 20D.
5. **The gate frame is never explicitly defined.** From the code it is
   yaw-only about world z, origin at gate center, x-axis along the
   gate's outward normal (the gate-passage test is
   `cos(yaw)·Δx + sin(yaw)·Δy > 0`).

---

## 4. Reference: Swift (UZH, *Nature* 2023) observation space

For comparison, **31 values** total (normalized):

| block          | size | content                                                      |
|----------------|------|--------------------------------------------------------------|
| Robot state    | 3    | global position                                              |
|                | 3    | global velocity                                              |
|                | 9    | global rotation matrix (flattened `R_{B→W}`)                 |
| Next gate      | 12   | relative xyz of the 4 inner gate corners                     |
| Previous action| 4    | last action sent to the actuators                            |

> Critic privileged info: at training time the value/critic network can
> see ground-truth quantities that the policy cannot. The policy itself
> sees only the 31 values above (with realistic noise).

### High-level differences vs. MonoRace

| aspect            | Swift                                          | MonoRace                                              |
|-------------------|------------------------------------------------|-------------------------------------------------------|
| Frame for state   | World frame                                    | Current-gate frame `G_i`                              |
| Attitude encoding | Full 9-element rotation matrix                 | 3 ZYX Euler angles                                    |
| Gate encoding     | 12 = 4 corners × xyz of next gate              | `(p, ψ)` of next 1–2 gates (4 or 8 scalars)           |
| Body rates        | not in observation                             | included (3)                                          |
| Motor / action    | previous action (4)                            | propeller speeds in rad/s (4) — current state, not a1 |
| Total dim         | 31                                             | 24                                                    |

---

**Primary sources used to build this spec**
- MonoRace, arXiv:2601.15222 — text and equation (3) of "Policy and Training".
- *One Net to Rule Them All*, Ferede et al., arXiv:2504.21586 — eq. (3) with
  `gates_ahead = 1` (20D).
- Code: [`tudelft/optimal_quad_control_RL @ icra2025`](https://github.com/tudelft/optimal_quad_control_RL/tree/icra2025),
  `quad_race_env.py` (`update_states_gate`) and `Generate_C_code.ipynb`
  (matching C inference indexing and rotation formulas).
