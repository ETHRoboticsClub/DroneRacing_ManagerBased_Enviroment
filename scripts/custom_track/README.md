# Custom Track — Design, Save, and Test Racing Tracks

This directory lets you design drone-racing gate layouts interactively, save them as JSON files, and test a trained policy on any saved track — all without booting Isaac Sim until the final play step.

```
scripts/custom_track/
├── track_io.py            # shared I/O + math library (no Isaac, no GUI deps)
├── track_designer.py      # PySide6 visual editor
├── play_custom_track.py   # load a saved track + run a trained policy
├── config.json            # default values for gate_size, bar_thickness, etc.
└── tracks/                # where saved tracks live (JSON files)
    ├── track_01_slalom_straights.json
    └── ...
```

---

## Files at a Glance

### `track_io.py`
Pure Python (numpy + scipy + stdlib). No Isaac Sim or torch imports — safe to use in the designer, in tests, or anywhere you want to read/write a track programmatically.

Key functions:

| Function | What it does |
|---|---|
| `default_track()` | Empty track dict with defaults from `config.json` |
| `save_track(path, track)` | Write track to disk as pretty-printed JSON |
| `load_track(path)` | Read JSON; back-fills `pitch`/`roll`/`auto_yaw` for old tracks |
| `track_to_arrays(track)` | Convert to `(positions[N,3], quats_wxyz[N,4], meta)` — resolves auto-yaw |
| `interpolate_loop(centres, n, closed)` | Sample a chord-length cubic spline through the gates |
| `gate_tangents_from_spline(centres, closed)` | Unit tangent at each gate — used for auto-yaw |
| `euler_to_quat_wxyz(yaw, pitch, roll)` | Intrinsic ZXY Euler → Isaac Lab `(w,x,y,z)` quaternion |
| `quat_axes_wxyz(quats)` | Rotate canonical basis by each quaternion → `(right, forward, up)` |

> `save_track()` only persists the schema fields listed below — it
> silently drops any extra dict keys, e.g. `track_name`, `description`, `elements_tested`, or a
> per-gate `comment`. Several of the shipped files under `tracks/*.json` carry that metadata;
> opening one in the designer and saving over it discards it with no warning.

**Track JSON format (schema v1):**
```json
{
  "schema_version": 1,
  "gate_size":     0.7,
  "bar_thickness": 0.05,
  "loop_closed":   true,
  "gates": [
    {"x": 0.0, "y": 0.0, "z": 1.5,
     "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "auto_yaw": true},
    ...
  ]
}
```

Gate angles use **radians on disk**; the designer shows them in degrees.

**Angle conventions:**
- `yaw` — rotates the gate about the world vertical (`+Z`). `yaw=0` means the passage direction points along world `+Y`.
- `pitch` — tips the gate forward/back about its local right axis (`+X`, after yaw). Positive pitch tilts the top toward the incoming drone.
- `roll` — banks the gate left/right about its local passage axis (`+Y`, after yaw + pitch).
- `auto_yaw=true` — the stored yaw is ignored; the yaw is computed from the closed-loop spline tangent at that gate when `track_to_arrays` is called.

> **Relationship to the environment's fixed-track YAML:** `configs/tracks/*.yaml`,
> loaded by `Racing_Quadcopter.terrains.track_io.load_gates_from_yaml` for the actual gym tasks, is a
> **different schema**: corner-origin local frame (not world frame), degrees (not radians), and yaw
> measured **clockwise from +Y** (this JSON format's `yaw` is CCW from +Y, unnegated) — the same
> numeric yaw value points in opposite physical directions between the two. There is currently no
> converter between this JSON schema and that YAML schema; don't hand-port gate coordinates from one
> to the other without re-deriving the angle. Full convention reference on the YAML side:
> `source/Racing_Quadcopter/Racing_Quadcopter/terrains/README.md`.

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
    --track my_track \                          # bare name → tracks/my_track.json
    --task Isaac-Racing-Quadcopter-Manager-v0 \
    --checkpoint logs/skrl/my_run/checkpoints/agent_200000.pt
```

Pass a full path instead of a bare name if the track lives elsewhere:
```bash
--track /absolute/path/to/track.json
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
python scripts/custom_track/track_designer.py scripts/custom_track/tracks/track_01_slalom_straights.json
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
- **Pitch** — tips the gate forward or backward (0 = upright). Positive values tilt the top toward the incoming drone.
- **Roll** — banks the gate left or right around the passage axis (0 = upright).
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
**File → Save** (or `Ctrl+S`) saves to the current file. **File → Save As** lets you choose a name. Tracks are always saved to `scripts/custom_track/tracks/` by default. Give your file a descriptive name, e.g. `my_oval.json`.

#### 9. Open an existing track
**File → Open** opens a file dialog rooted at `scripts/custom_track/tracks/`. Select any `.json` file.

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

Replace `<track_name>` with the filename (without `.json`) of any track in `scripts/custom_track/tracks/`, and `<path/to/checkpoint.pt>` with the path to a trained skrl checkpoint.

The script:
1. Loads the track JSON.
2. Computes a square terrain cell just large enough to contain all gates (plus a 2 m margin).
3. Spawns a single Isaac Sim environment with one drone.
4. Runs the policy in eval mode — no training, no gradient updates.

The drone spawns at the first gate facing the direction of the spline, then tries to fly the loop.
