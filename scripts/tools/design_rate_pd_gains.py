#!/usr/bin/env python3
"""Design per-axis PD gains for body-rate control (Mx, My, Mz).

Edit the USER CONFIG block below, then run:
    python scripts/tools/design_rate_pd_gains.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# =============================================================================
# USER CONFIG — edit these values
# =============================================================================

# Anemos base_link (URDF)
MASS_KG = 0.8351086449771665
I_ROLL = 0.002158   # Ixx [kg·m²]
I_PITCH = 0.002919  # Iyy
I_YAW = 0.002743    # Izz

# Per-axis design targets — tuples are (roll, pitch, yaw)
OVERSHOOT_PCT = (5.0, 5.0, 8.0)    # peak overshoot [%]  →  ζ_i     (roll, pitch, yaw)
TS_SETTLING_S = (0.4, 0.5, 0.6)    # 2% settling time [s] →  ωn_i = 4 / (ζ_i * ts_i)

# =============================================================================


@dataclass
class AxisDesign:
    name: str
    inertia_kg_m2: float
    zeta: float
    wn_rad_s: float
    kp_nm_per_rad_s: float
    kd_nm_s_per_rad: float
    overshoot_pct: float
    ts_s: float


def overshoot_from_zeta(zeta: float) -> float:
    if zeta >= 1.0:
        return 0.0
    return 100.0 * math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))


def zeta_from_overshoot_pct(overshoot_pct: float) -> float:
    if overshoot_pct <= 0.0:
        return 1.0
    if overshoot_pct >= 100.0:
        raise ValueError("TARGET_OVERSHOOT_PCT must be < 100")
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if overshoot_from_zeta(mid) > overshoot_pct:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wn_from_ts_zeta(ts_s: float, zeta: float) -> float:
    return 4.0 / (zeta * ts_s)


def design_axis(name: str, inertia: float, zeta: float, wn: float) -> AxisDesign:
    kp = inertia * wn * wn
    kd = 2.0 * zeta * wn * inertia
    return AxisDesign(
        name=name,
        inertia_kg_m2=inertia,
        zeta=zeta,
        wn_rad_s=wn,
        kp_nm_per_rad_s=kp,
        kd_nm_s_per_rad=kd,
        overshoot_pct=overshoot_from_zeta(zeta),
        ts_s=4.0 / (zeta * wn),
    )


def run() -> None:
    if len(OVERSHOOT_PCT) != 3 or len(TS_SETTLING_S) != 3:
        raise ValueError("OVERSHOOT_PCT and TS_SETTLING_S must each have 3 values: (roll, pitch, yaw)")

    axes = (
        ("roll", I_ROLL, OVERSHOOT_PCT[0], TS_SETTLING_S[0]),
        ("pitch", I_PITCH, OVERSHOOT_PCT[1], TS_SETTLING_S[1]),
        ("yaw", I_YAW, OVERSHOOT_PCT[2], TS_SETTLING_S[2]),
    )

    designs = []
    for name, inertia, os_pct, ts in axes:
        zeta = zeta_from_overshoot_pct(os_pct)
        wn = wn_from_ts_zeta(ts, zeta)
        designs.append(design_axis(name, inertia, zeta, wn))

    print("=" * 72)
    print("RATE PD GAIN DESIGN (principal-axis, decoupled)")
    print("=" * 72)
    print(f"Mass (reference):  m = {MASS_KG:.6f} kg")
    print(f"Inertia:  I_roll={I_ROLL:.6f},  I_pitch={I_PITCH:.6f},  I_yaw={I_YAW:.6f} kg·m²")
    print()

    header = (
        f"{'Axis':<6} {'I':>10} {'ζ':>8} {'ωn':>10} "
        f"{'Kp':>14} {'Kd':>12} {'OS%':>8} {'ts':>8}"
    )
    print(header)
    print("-" * len(header))
    for d in designs:
        print(
            f"{d.name:<6} {d.inertia_kg_m2:10.6f} {d.zeta:8.4f} {d.wn_rad_s:10.4f} "
            f"{d.kp_nm_per_rad_s:14.6f} {d.kd_nm_s_per_rad:12.6f} "
            f"{d.overshoot_pct:8.3f} {d.ts_s:8.4f}"
        )

    print()
    print("Relations:")
    print("  Kp = I * ωn²")
    print("  Kd = 2 * ζ * ωn * I")
    print("  OS% = 100 * exp(-πζ / sqrt(1-ζ²))")
    print("  ts  = 4 / (ζ * ωn)")
    print()
    print("=" * 72)
    print("CONTROLLER")
    print("=" * 72)
    print("""
  Plant:     I_i · ω̇_i = M_i
  Error:     e_i = ω_des,i − ω_i
  PD:        M_i = Kp_i · e_i − Kd_i · ω̇_i

  Closed loop (ω_des → ω):
      ω_i/ω_des,i = ωn² / (s² + 2ζωn s + ωn²)

  Discrete (sim):
      M_k = Kp · (ω_des,k − ω_k) − Kd · (ω_k − ω_{k−1}) / Δt
""")
    for d in designs:
        print(f"  {d.name}:  Kp = {d.kp_nm_per_rad_s:.6f} N·m/(rad/s),  Kd = {d.kd_nm_s_per_rad:.6f} N·m·s/rad")


if __name__ == "__main__":
    run()
