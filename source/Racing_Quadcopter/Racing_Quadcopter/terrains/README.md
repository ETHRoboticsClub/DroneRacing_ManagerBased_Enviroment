# `terrains/` — Racing Gate Terrain System

Vendored extension of IsaacLab's terrain system (`__init__.py` notes upstream has no
`isaaclab.terrains.config.racing`). This package builds the gate-based racing terrains
consumed by `RacingQuadcopterEnvCfg` and `CompetitionEnvCfg`: two independent gate-layout
generators — procedural (random) and fixed (deterministic YAML) — sharing one data model
and one module-level registry.

## File map

| File | What it is | Read this when... |
|---|---|---|
| `gate_base.py` | Shared data model: `GateData`, the module-level `gate_registry` dict, `create_gate_frame()` (builds the 4-bar trimesh gate mesh), `get_gate_registry()` / `clear_gate_registry()`. | You need to understand how gate data crosses from terrain generation to the env, or you're touching gate mesh construction. |
| `racing_gates_cfg.py` | Config dataclasses: `GateSpec` (one gate: position + RPY), `MeshRacingGatesTerrainCfg` (procedural params), `MeshFixedGatesTerrainCfg` (fixed params). | You need to tune generation parameters (gate size range, spacing, height) or construct a `GateSpec` by hand. |
| `racing_gates.py` | Procedural generator: `racing_gates_terrain()`, trajectory generators (`_generate_figure8`, `_generate_oval`, `_generate_random_track`, `_generate_chicane_track`), gate placement (`_place_gates_along_trajectory`). | You're changing how random tracks are generated (trajectory shape, gate spacing/placement logic). |
| `fixed_gates.py` | Deterministic generator: `fixed_gates_terrain()` — takes a list of `GateSpec` and places gates at exactly those positions, no randomization. | You're changing how a *fixed* circuit becomes a terrain mesh, or debugging the yaw sign convention. |
| `track_io.py` | YAML ↔ `GateSpec` conversion: `load_gates_from_yaml()`. | You're loading/authoring a fixed track from `configs/tracks/*.yaml`. |
| `racing_terrain.py` | Public factories: `create_racing_gates_terrain(seed)`, `create_fixed_track_terrain(gates, ...)`. Build the full `TerrainGeneratorCfg` (grid size, rows/cols, sub-terrain assignment) handed to IsaacLab's `TerrainImporterCfg`. | **Start here if you're new to this package** — this is the entry point called from `racing_quadcopter_env_cfg.py`. |
| `__init__.py` | Public API surface — re-exports the names other packages are meant to import (`GateData`, `GateSpec`, `clear_gate_registry`, `get_gate_registry`, `create_fixed_track_terrain`, `create_racing_gates_terrain`, `load_gates_from_yaml`, ...). | You're importing from `Racing_Quadcopter.terrains` elsewhere and want the intended surface, not internals. |

## Two pipelines, one shared core

```
                          gate_base.py
                    GateData / gate_registry {}
                    create_gate_frame()
                          ▲          ▲
                  writes  │          │  writes
        ┌──────────────────┘          └──────────────────┐
        │                                                 │
  racing_gates.py                                  fixed_gates.py
  racing_gates_terrain()                            fixed_gates_terrain()
  (procedural, random)                              (deterministic)
        ▲                                                 ▲
  MeshRacingGatesTerrainCfg                       MeshFixedGatesTerrainCfg
        ▲                                                 ▲
  create_racing_gates_terrain(seed)             create_fixed_track_terrain(gates)
        └───────────────── racing_terrain.py ─────────────┘
                              │
              called from racing_quadcopter_env_cfg.py
        (RacingQuadcopterEnvCfg.__post_init__ / _FIXED variant)
                              │
              RacingQuadcopterRLEnv._setup_gate_tracking()
              reads gate_registry after terrain generation,
              converts local → world frame
```

Both generators are called once per terrain cell by IsaacLab's `TerrainGenerator`, and both
write into the same module-level `gate_registry` (keyed by `(round(difficulty, 6), seed)`).
This indirection exists because `TerrainGenerator` copies the cfg dataclass before invoking
the terrain function — any state written to `cfg` itself is silently discarded, so the
registry is the only channel back to the env.

- **Procedural** (`racing_gates.py`) — used by `-Manager-v0` / `-Manager-Play-v0`. Generates a
  random closed-loop trajectory (figure-8, oval, or a `"random"` pick between cubic-spline
  and chicane), then places gates at arc-length intervals along it. Nothing touches disk.
- **Fixed** (`fixed_gates.py`) — used by `-Manager-Fixed-v0` and `CompetitionEnvCfg`. Gate
  positions come verbatim from a `GateSpec` list, loaded from `configs/tracks/*.yaml` via
  `track_io.py`.

**Registry ↔ cell mapping is positional, not keyed.** `RacingQuadcopterRLEnv._setup_gate_tracking()`
does `gate_datas = list(registry.values())` and then indexes into that list with
`cell_idx = (row * num_cols + col) % len(gate_datas)` — it relies on dict insertion order
matching `TerrainGenerator`'s row-major cell generation order, not on the registry's own
`(difficulty, seed)` keys. This is harmless for **fixed** tracks (every cell holds an
identical layout, so any ordering gives the same result) but is a real, unverified assumption
for **procedural** tracks, where a mismatch would silently pair a cell's rendered gate mesh
with a *different* cell's logical gate positions. It appears to hold in practice with the
current IsaacLab `TerrainGenerator`, but nothing in this codebase asserts it.

## Coordinate conventions — read this before touching gate positions

- **Frame**: sub-terrain **corner-origin local frame**. `(0, 0, 0)` is the bottom-left corner
  of the 15×15 m tile; a gate at `(7.5, 7.5, z)` sits at the tile centre. The env adds the
  terrain's centre offset (`-size/2`) and then `terrain_origins[row, col]` to get world frame
  — see `RacingQuadcopterRLEnv._setup_gate_tracking()` (outside this package).
- **Gate-local axes**: `+X` = right (bar width), `+Y` = forward = passage direction (drone
  flies through in `+Y`), `+Z` = up (always world up — gates never tilt in the procedural
  generator; `fixed_gates.py` allows roll/pitch via `GateSpec` for the fixed pipeline).
- **`GateSpec.yaw_deg`**: **clockwise from +Y**, viewed from above. `0°` → drone passes in
  world `+Y`. `90°` → passes in world `+X`. `fixed_gates.py` negates the yaw before handing
  it to `scipy.Rotation.from_euler("xyz", ...)`, because scipy's convention is CCW.
- **Quaternions**: always stored `(w, x, y, z)` — IsaacLab convention. scipy uses
  `(x, y, z, w)`; every conversion site does the swap explicitly, never assume it's implicit.

**This is a different convention from the visual track builder** (`scripts/custom_track/`):
that tool stores gates in **world frame**, angles in **radians**, and yaw is **not negated**
(CCW from +Y). The same numeric yaw value means opposite physical directions between the two
formats, and there is currently no converter between them — don't hand-port coordinates
between a `scripts/custom_track/tracks/*.json` file and a `configs/tracks/*.yaml` file without
re-deriving the angle.

## Common tasks

| I want to... | Look at |
|---|---|
| Add/edit a fixed circuit | `configs/tracks/*.yaml`, loaded via `track_io.load_gates_from_yaml()` |
| Change random-track difficulty scaling, gate spacing/size | `MeshRacingGatesTerrainCfg` fields in `racing_gates_cfg.py`, consumed by `racing_gates.py` |
| Add a new procedural trajectory shape | `racing_gates.py` — add a `_generate_*` function and wire it into `_generate_racing_trajectory()`'s dispatch |
| Change the terrain grid size (rows/cols/cell size) | `racing_terrain.py` — `create_racing_gates_terrain()` / `create_fixed_track_terrain()` |
| Change gate mesh appearance (bar thickness, shape) | `gate_base.py::create_gate_frame()` is the default frame builder. For different geometry entirely, `MeshFixedGatesTerrainCfg.gate_frame_fn` (any callable matching `create_gate_frame`'s signature) swaps it out per-terrain — see `competition_env_cfg.py::create_competition_gate_frame()` for a real, live example (competition gate: frame members sit fully *outside* the aperture instead of straddling it). Note: `gate_color` is set on the trimesh *after* `apply_transform()` (deliberately, to avoid a known trimesh `TextureVisuals` ordering issue) but still doesn't reliably show up once `TerrainGenerator` bakes the mesh to USD — see the `# ... (DOESN'T WORK)` caveat in `racing_terrain.py`. Unresolved; needs a running Isaac Sim session to debug, not fixable by reading source alone. |
| Build a fixed track with a non-standard tile size | Don't assume every fixed track sits in the default 15×15 m tile from `create_fixed_track_terrain()`. `CompetitionEnvCfg` builds its `TerrainGeneratorCfg` **by hand** (`competition_env_cfg.py`, not via `racing_terrain.py`) with a custom `COMPETITION_TILE_SIZE = (175.0, 28.0)` because its course is ~136 m long. If a `configs/tracks/*.yaml` has coordinates outside `[0, 15]`, check the consuming env config for a custom tile size before assuming it's a bug. |
| Understand how the env reads gate data after terrain gen | `RacingQuadcopterRLEnv._setup_gate_tracking()` in `../racing_quadcopter_env.py` (outside this package) |
