# Custom Track — Design, Save, and Test Racing Tracks

This directory lets you design drone-racing gate layouts interactively, save them as YAML files, and test a trained policy on any saved track — all without booting Isaac Sim until the final play step.

```
scripts/custom_track/
├── track_io.py            # shared I/O + math library (no Isaac, no GUI deps)
├── track_designer.py      # PySide6 visual editor
├── play_custom_track.py   # load a saved track + run a trained policy
├── config.yaml            # default values for gate_size, bar_thickness, loop_closed
└── tracks/                # where saved tracks live (YAML files)
    ├── track_01_slalom_straights.yaml
    └── ...
```

There is currently no `tests/` directory or offline smoke-test script in this folder.

---

## Files at a Glance

### `track_io.py`
Pure Python (numpy + scipy + PyYAML). No Isaac Sim or torch imports — safe to use in the designer or anywhere you want to read/write a track programmatically.

Key functions:

| Function | What it does |
|---|---|
| `default_track()` | Empty track dict with defaults from `config.yaml` |
| `save_track(path, track)` | Write track to disk as YAML |
| `load_track(path)` | Read a track YAML (schema `1`); raises on an unsupported `schema_version` |
| `track_to_arrays(track)` | Convert to `(positions[N,3], quats_wxyz[N,4], meta)` — resolves auto-yaw |
| `interpolate_loop(centres, n, closed)` | Sample a chord-length cubic spline through the gates |
| `gate_tangents_from_spline(centres, closed)` | Unit tangent at each gate — used for auto-yaw |
| `euler_to_quat_wxyz(roll, pitch, yaw)` | Extrinsic XYZ (ROS RPY) Euler → Isaac Lab `(w,x,y,z)` quaternion |
| `quat_axes_wxyz(quats)` | Rotate canonical basis by each quaternion → `(right, forward, up)` |

**Track YAML format (schema v1):**
```yaml
schema_version: 1
track_name: ""
description: ""
elements_tested: []
gate_size:     0.7
bar_thickness: 0.05
loop_closed:   true
gate_track:
  - position: [0.0, 0.0, 1.5]
    rotation: [0.0, 0.0, 0.0]   # [roll, pitch, yaw], degrees
    auto_yaw: true
  - ...
```

Gate angles use **degrees on disk** (same convention the designer shows).

**Angle conventions** (extrinsic XYZ / ROS RPY order; see `track_io.py::euler_to_quat_wxyz`):
- `rotation[0]` (**roll**) — rotation about world/gate-local `+X` (the bar's right axis).
- `rotation[1]` (**pitch**) — rotation about `+Y` (the passage direction), applied after roll.
- `rotation[2]` (**yaw**) — rotation about world `+Z`, **clockwise from +Y** viewed from above: `yaw=0` → passage direction points along world `+Y`, `yaw=90` → `+X`.
- `auto_yaw=true` — the stored yaw is ignored; the yaw is computed from the closed-loop spline tangent at that gate when `track_to_arrays` is called. Pitch and roll are unaffected by `auto_yaw` and always come from the stored value.

---

### `track_designer.py`
Interactive PySide6 application. Two panels:

- **Left — Top-down view (XY):** place and drag gates in the horizontal plane. The spline loop is drawn in real time. Each gate shows its bar orientation as a white line and passage direction as a green arrow.
- **Right — 3D preview:** full perspective view of the gate layout; click and drag to rotate the camera.

The property panel on the right lets you set the position, angles, and flags for the selected gate, reorder gates, and change global track parameters.

---

### `play_custom_track.py`
Fork of `scripts/skrl/play.py`. Loads a saved track, builds a one-cell Isaac terrain with exactly those gates, then runs a trained checkpoint through the standard skrl eval loop.

Usage:
```bash
python scripts/custom_track/play_custom_track.py \
    --track my_track \                          # bare name → tracks/my_track.yaml
    --task Isaac-Racing-Quadcopter-Manager-v0 \
    --checkpoint logs/skrl/my_run/checkpoints/agent_200000.pt
```

Pass a full path instead of a bare name if the track lives elsewhere:
```bash
--track /absolute/path/to/track.yaml
```

---

## Track Designer — First-Time User Guide

### Prerequisites

You need PySide6 and scipy installed in your Python environment:

```bash
pip install PySide6 scipy
```

Then launch the designer from the repo root:

```bash
python scripts/custom_track/track_designer.py
```

To open an existing track directly:

```bash
python scripts/custom_track/track_designer.py scripts/custom_track/tracks/track_01_slalom_straights.yaml
```

---

### The Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  File   Edit   [Add] [Select] [Delete]   Snap: 0.5   Extent: 20.0          │
├─────────────────────────────────┬───────────────────────────────────────────┤
│                                 │  Gate Properties                          │
│   Top-down (XY) view            │  ─────────────────                        │
│                                 │  X  Y  Z   (metres)                       │
│   · blue dots = gates           │  Yaw  Pitch  Roll  (degrees)              │
│   · blue line = spline loop     │  ☑ Auto-yaw                               │
│   · white line = gate bar       │                                           │
│   · green arrow = passage dir   │  Gate list (↑↓ reorder, click to select) │
│                                 │                                           │
│                                 │  ─────────────────                        │
│                                 │  Track: gate_size  bar_thickness          │
│                                 │  ☑ loop_closed    Default Z               │
├─────────────────────────────────┴───────────────────────────────────────────┤
│   3D preview (drag to rotate)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step-by-Step: Building Your First Track

#### 1. Choose "Add" mode
Click the **Add** button in the toolbar (or it is the default when you open a new file). The cursor becomes a crosshair over the top-down canvas.

#### 2. Place gates
**Left-click** anywhere on the top-down canvas to place a gate. Each click adds a new gate at the clicked XY position and at the "Default Z" height shown in the track panel (default 1.5 m). The spline loop updates instantly.

Place at least 3 gates for a meaningful loop.

#### 3. Select and adjust a gate
Click the **Select** button in the toolbar. Then **left-click** a gate (blue dot) to select it — it turns red. Its properties appear in the **Gate Properties** panel on the right.

- **X / Y / Z** — position in metres. Z is the height above the ground.
- **Yaw** — horizontal rotation in degrees. Leave **Auto-yaw** checked and this field is ignored; the gate will automatically face the next gate along the spline.
- **Roll** — rotates about the gate's right axis (`+X`), tipping the top forward/backward (0 = upright).
- **Pitch** — rotates about the gate's passage axis (`+Y`, applied after roll), banking the gate left/right (0 = upright).
- **Auto-yaw** — when checked, the yaw is derived from the spline tangent at that gate. Uncheck if you want a specific yaw angle.

You can also **drag** a selected gate to a new XY position directly on the canvas.

#### 4. Adjust height (Z)
The top-down canvas is a bird's-eye view, so Z is edited in the properties panel. Set the **Z** field to whatever altitude you want the gate centre at. The 3D preview reflects the change immediately.

To change the default altitude for future gates, edit **Default Z** in the Track section of the properties panel.

#### 5. Reorder gates
The gate list in the properties panel shows gates in the order the drone visits them. Use the **↑** and **↓** buttons to move the selected gate earlier or later in the sequence. The spline redraws to reflect the new visit order.

#### 6. Delete a gate
Either:
- Switch to **Select** mode, click a gate, then press the **Delete** key, or
- Switch to **Delete** mode and left-click any gate to remove it immediately.

#### 7. Set global track parameters
In the **Track** section at the bottom of the properties panel:

| Field | Default | Meaning |
|---|---|---|
| `gate_size` | 0.7 m | Width (and height) of the square gate aperture |
| `bar_thickness` | 0.05 m | Diameter of the tubular bars making up the frame |
| `loop_closed` | ✓ | Whether to connect the last gate back to the first |
| `Default Z` | 1.5 m | Height applied to newly placed gates |

#### 8. Save the track
**File → Save** (or `Ctrl+S`) saves to the current file. **File → Save As** lets you choose a name. Tracks are always saved to `scripts/custom_track/tracks/` by default. Give your file a descriptive name, e.g. `my_oval.yaml`.

#### 9. Open an existing track
**File → Open** opens a file dialog rooted at `scripts/custom_track/tracks/`. Select any `.yaml` file.

---

### Tips

- **Snap** (toolbar spinbox) rounds gate placements to a grid. Set to `0.0` to disable snapping.
- **Extent** (toolbar spinbox) controls the visible range of the top-down canvas in metres.
- **Auto-yaw** (default on) means you rarely need to touch the Yaw field. Uncheck it only if a specific gate needs a non-tangent orientation.
- Pitch and roll default to `0` (upright gate). Non-zero values are useful for banked turns or gates angled into a dive.
- The 3D preview is read-only — all editing happens in the top-down view or the property panel.
- The window title shows an asterisk (`*`) when there are unsaved changes.

---

## Testing a Track with a Trained Policy

Once you have saved a track, run:

```bash
python scripts/custom_track/play_custom_track.py \
    --track <track_name> \
    --task Isaac-Racing-Quadcopter-Manager-v0 \
    --checkpoint <path/to/checkpoint.pt>
```

Replace `<track_name>` with the filename (without `.yaml`) of any track in `scripts/custom_track/tracks/`, and `<path/to/checkpoint.pt>` with the path to a trained skrl checkpoint.

The script:
1. Loads the track YAML.
2. Computes a square terrain cell just large enough to contain all gates (plus a 2 m margin).
3. Spawns a single Isaac Sim environment with one drone.
4. Runs the policy in eval mode — no training, no gradient updates.

The drone spawns at the first gate facing the direction of the spline, then tries to fly the loop.
