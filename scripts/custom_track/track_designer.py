"""Interactive PySide6 track designer for drone-racing tracks.

Standalone GUI (no Isaac Lab / torch dependency). Click on a top-down 2D map
to place gates, drag to move, edit z/yaw in the side panel, preview in 3D,
save to JSON.  The saved track is loaded by the play-side script that runs
the trained policy against the custom course.

Run::

    python scripts/custom_track/track_designer.py             # blank track
    python scripts/custom_track/track_designer.py track.json  # open existing

Conventions are documented in ``track_io`` — yaw is in **radians on disk**
and **degrees in the UI**.  Gates are upright (rotation about world +Z only).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Tell matplotlib (and its qtagg backend) to bind against PySide6 specifically,
# rather than auto-detecting and possibly picking PyQt5 if it ever becomes
# importable in this venv.
os.environ.setdefault("QT_API", "pyside6")

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6 import QtCore, QtGui, QtWidgets

# Allow `python scripts/custom_track/track_designer.py` from any cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import track_io  # noqa: E402


# ---------------------------------------------------------------------------
# 2D top-down editor
# ---------------------------------------------------------------------------

class TopDownCanvas(FigureCanvas):
    """Top-down (XY) editor.

    Modes:
      - ``add``    : click empty area → append new gate; click existing → select
      - ``select`` : click existing → select; drag selected gate to move
      - ``delete`` : click existing → remove

    Right-click always deletes the nearest gate within the hit radius
    regardless of current mode.
    """

    gates_changed = QtCore.Signal()
    selection_changed = QtCore.Signal(object)  # int or None
    mouse_moved = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 6))
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)
        fig.tight_layout()

        self._track: dict = track_io.default_track()
        self._mode: str = "add"
        self._selected: int | None = None
        self._dragging: bool = False
        self._snap: float = 0.0
        self._default_z: float = 1.5
        self._bounds = (-15.0, 15.0, -15.0, 15.0)

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

        self.redraw()

    # ----- public API ----------------------------------------------------
    @property
    def track(self) -> dict:
        return self._track

    def set_track(self, track: dict) -> None:
        self._track = track
        self._selected = None
        self.redraw()
        self.selection_changed.emit(None)
        self.gates_changed.emit()

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_snap(self, snap: float) -> None:
        self._snap = max(0.0, float(snap))

    def set_default_z(self, z: float) -> None:
        self._default_z = float(z)

    def set_bounds(self, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        self._bounds = (xmin, xmax, ymin, ymax)
        self.redraw()

    def selected_index(self) -> int | None:
        return self._selected

    def select(self, idx: int | None) -> None:
        if idx is not None and (idx < 0 or idx >= len(self._track["gates"])):
            idx = None
        self._selected = idx
        self.redraw()
        self.selection_changed.emit(idx)

    def update_gate(self, idx: int, **kwargs) -> None:
        if 0 <= idx < len(self._track["gates"]):
            self._track["gates"][idx].update(kwargs)
            self.redraw()
            self.gates_changed.emit()

    def update_track_meta(self, **kwargs) -> None:
        self._track.update(kwargs)
        self.redraw()
        self.gates_changed.emit()

    def auto_yaw_all(self) -> None:
        for g in self._track["gates"]:
            g["auto_yaw"] = True
        self.redraw()
        self.gates_changed.emit()

    def clear(self) -> None:
        self._track["gates"].clear()
        self._selected = None
        self.redraw()
        self.selection_changed.emit(None)
        self.gates_changed.emit()

    def delete_gate(self, idx: int) -> None:
        self._delete(idx)

    def reorder(self, idx: int, direction: str) -> None:
        gates = self._track["gates"]
        if direction == "up" and idx > 0:
            gates[idx - 1], gates[idx] = gates[idx], gates[idx - 1]
            self.select(idx - 1)
        elif direction == "down" and idx < len(gates) - 1:
            gates[idx + 1], gates[idx] = gates[idx], gates[idx + 1]
            self.select(idx + 1)
        else:
            return
        self.gates_changed.emit()

    # ----- mouse events --------------------------------------------------
    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        # Right-click: always delete nearest gate within radius.
        if event.button == 3:
            idx = self._hit_test(x, y)
            if idx is not None:
                self._delete(idx)
            return

        if event.button != 1:
            return

        idx = self._hit_test(x, y)
        if self._mode == "delete":
            if idx is not None:
                self._delete(idx)
            return

        if self._mode == "add":
            if idx is not None:
                self.select(idx)
            else:
                self._add(x, y)
            return

        # Select mode.
        if idx is not None:
            self.select(idx)
            self._dragging = True
        else:
            self.select(None)

    def _on_motion(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        self.mouse_moved.emit(x, y)
        if self._dragging and self._selected is not None:
            x, y = self._snap_to_grid(x, y)
            g = self._track["gates"][self._selected]
            g["x"] = x
            g["y"] = y
            self.redraw()
            self.gates_changed.emit()

    def _on_release(self, _event):
        self._dragging = False

    # ----- helpers -------------------------------------------------------
    def _snap_to_grid(self, x: float, y: float) -> tuple[float, float]:
        if self._snap > 0.0:
            x = round(x / self._snap) * self._snap
            y = round(y / self._snap) * self._snap
        return float(x), float(y)

    def _hit_test(self, x: float, y: float) -> int | None:
        gates = self._track["gates"]
        if not gates:
            return None
        cx = np.array([g["x"] for g in gates])
        cy = np.array([g["y"] for g in gates])
        d = np.hypot(cx - x, cy - y)
        # Hit radius: half a gate plus pad. In data units, so it's slightly
        # zoom-dependent in feel, which is acceptable.
        r = max(0.5, float(self._track.get("gate_size", 0.7)) * 0.6)
        i = int(np.argmin(d))
        return i if d[i] <= r else None

    def _add(self, x: float, y: float) -> None:
        x, y = self._snap_to_grid(x, y)
        # New gates inherit z from last selection's z if there is one — keeps
        # successive clicks at the same height by default.
        z = self._default_z
        gates = self._track["gates"]
        if self._selected is not None and 0 <= self._selected < len(gates):
            z = float(gates[self._selected]["z"])
        gate = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw": 0.0,
            "auto_yaw": True,
        }
        gates.append(gate)
        self.select(len(gates) - 1)
        self.gates_changed.emit()

    def _delete(self, idx: int) -> None:
        gates = self._track["gates"]
        if not (0 <= idx < len(gates)):
            return
        del gates[idx]
        if self._selected is not None:
            if self._selected == idx:
                self._selected = None
            elif self._selected > idx:
                self._selected -= 1
        self.redraw()
        self.selection_changed.emit(self._selected)
        self.gates_changed.emit()

    # ----- rendering -----------------------------------------------------
    def redraw(self):
        ax = self.ax
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        xmin, xmax, ymin, ymax = self._bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.axvline(0.0, color="0.7", lw=0.5)

        gates = self._track["gates"]
        if not gates:
            self.draw_idle()
            return

        positions = np.array([[g["x"], g["y"], g["z"]] for g in gates])
        closed = bool(self._track.get("loop_closed", True))
        gate_size = float(self._track.get("gate_size", 0.7))

        # Spline curve through gate centres (visual aid; same construction the
        # simulator uses for the random track type).
        if len(gates) >= 2:
            try:
                pts, _ = track_io.interpolate_loop(positions, num_samples=300, closed=closed)
                ax.plot(pts[:, 0], pts[:, 1], color="#4a90e2", lw=1.2, alpha=0.85)
            except Exception:
                ext = np.vstack([positions, positions[:1]]) if closed else positions
                ax.plot(ext[:, 0], ext[:, 1], color="#4a90e2", lw=1.0, alpha=0.5, ls="--")

        # Resolved yaws: re-derive from quaternions so auto-yaw gates show their
        # spline-tangent orientation rather than the stale stored yaw.
        try:
            _, quats, _ = track_io.track_to_arrays(self._track)
            yaws = 2.0 * np.arctan2(quats[:, 3], quats[:, 0])
        except Exception:
            yaws = np.array([float(g.get("yaw", 0.0)) for g in gates])

        for i, g in enumerate(gates):
            yaw = float(yaws[i])
            cosy, siny = np.cos(yaw), np.sin(yaw)
            half = gate_size / 2.0
            # Bar from above is a line of length ``gate_size`` in gate +X (right).
            rx, ry = cosy * half, siny * half
            x0, y0 = g["x"] - rx, g["y"] - ry
            x1, y1 = g["x"] + rx, g["y"] + ry
            color = "#ff6a00" if i == self._selected else "#222222"
            lw = 3.0 if i == self._selected else 1.8
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="round")

            # Forward arrow along gate +Y (passage direction).
            fx, fy = -siny, cosy
            arrow_len = max(0.4, gate_size * 0.6)
            ax.annotate(
                "",
                xy=(g["x"] + fx * arrow_len, g["y"] + fy * arrow_len),
                xytext=(g["x"], g["y"]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )

            ax.text(
                g["x"] + 0.15, g["y"] + 0.15, str(i),
                color=color, fontsize=9, weight="bold",
            )

        self.draw_idle()


# ---------------------------------------------------------------------------
# 3D preview
# ---------------------------------------------------------------------------

class PreviewCanvas3D(FigureCanvas):
    """Read-only 3D preview of the current track."""

    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4))
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111, projection="3d")
        fig.tight_layout()
        self._track: dict = track_io.default_track()
        self.redraw()

    def set_track(self, track: dict) -> None:
        self._track = track
        self.redraw()

    def redraw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        gates = self._track["gates"]
        if not gates:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(0, 3)
            self.draw_idle()
            return

        positions = np.array([[g["x"], g["y"], g["z"]] for g in gates])
        closed = bool(self._track.get("loop_closed", True))
        gate_size = float(self._track.get("gate_size", 0.7))

        if len(gates) >= 2:
            try:
                pts, _ = track_io.interpolate_loop(positions, num_samples=300, closed=closed)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#4a90e2", lw=1.0, alpha=0.7)
            except Exception:
                pass

        try:
            _, quats, _ = track_io.track_to_arrays(self._track)
            yaws = 2.0 * np.arctan2(quats[:, 3], quats[:, 0])
        except Exception:
            yaws = np.array([float(g.get("yaw", 0.0)) for g in gates])

        half = gate_size / 2.0
        for i, g in enumerate(gates):
            yaw = float(yaws[i])
            c, s = np.cos(yaw), np.sin(yaw)
            cx, cy, cz = g["x"], g["y"], g["z"]
            # Aperture rectangle: 4 corners closed back to first.
            corners = np.array([
                [cx - half * c, cy - half * s, cz - half],
                [cx + half * c, cy + half * s, cz - half],
                [cx + half * c, cy + half * s, cz + half],
                [cx - half * c, cy - half * s, cz + half],
                [cx - half * c, cy - half * s, cz - half],
            ])
            ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="#ff6a00", lw=1.5)

        # Equal-aspect bbox around the gates (with padding).
        xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
        pad = 1.5
        cx = (xs.min() + xs.max()) / 2.0
        cy = (ys.min() + ys.max()) / 2.0
        cz = (zs.min() + zs.max()) / 2.0
        r = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()) / 2.0 + pad
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(max(0.0, cz - r), cz + r)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

        self.draw_idle()


# ---------------------------------------------------------------------------
# Side panel (track meta + selected gate + ordered gate list)
# ---------------------------------------------------------------------------

class PropertyPanel(QtWidgets.QWidget):
    track_meta_changed = QtCore.Signal(dict)
    gate_changed = QtCore.Signal(int, dict)
    select_requested = QtCore.Signal(object)   # int or None
    delete_requested = QtCore.Signal(int)
    reorder_requested = QtCore.Signal(int, str)  # idx, "up"|"down"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._block: bool = False
        self._track: dict = track_io.default_track()
        self._selected: int | None = None

        layout = QtWidgets.QVBoxLayout(self)

        # ---- Track meta ----
        track_box = QtWidgets.QGroupBox("Track")
        f = QtWidgets.QFormLayout(track_box)
        self._gate_size = self._spin(0.1, 5.0, 0.05, 3)
        self._bar_thickness = self._spin(0.005, 0.5, 0.005, 3)
        self._loop_closed = QtWidgets.QCheckBox("Closed loop")
        self._default_z = self._spin(0.1, 20.0, 0.1, 2)
        f.addRow("Gate size (m)", self._gate_size)
        f.addRow("Bar thickness (m)", self._bar_thickness)
        f.addRow("", self._loop_closed)
        f.addRow("Default new-gate Z (m)", self._default_z)
        layout.addWidget(track_box)
        for w in (self._gate_size, self._bar_thickness, self._default_z):
            w.valueChanged.connect(self._emit_meta)
        self._loop_closed.toggled.connect(self._emit_meta)

        # ---- Selected gate ----
        sel_box = QtWidgets.QGroupBox("Selected gate")
        f2 = QtWidgets.QFormLayout(sel_box)
        self._gx = self._spin(-1000, 1000, 0.1, 3)
        self._gy = self._spin(-1000, 1000, 0.1, 3)
        self._gz = self._spin(0.0, 50.0, 0.1, 3)
        self._gyaw = self._spin(-360.0, 360.0, 1.0, 1)
        self._gauto = QtWidgets.QCheckBox("Auto-yaw (use spline tangent)")
        f2.addRow("X (m)", self._gx)
        f2.addRow("Y (m)", self._gy)
        f2.addRow("Z (m)", self._gz)
        f2.addRow("Yaw (deg)", self._gyaw)
        f2.addRow("", self._gauto)
        layout.addWidget(sel_box)
        for w in (self._gx, self._gy, self._gz, self._gyaw):
            w.valueChanged.connect(self._emit_gate)
        self._gauto.toggled.connect(self._emit_gate)

        # ---- Ordered gate list ----
        list_box = QtWidgets.QGroupBox("Gates (visit order)")
        v = QtWidgets.QVBoxLayout(list_box)
        self._list = QtWidgets.QListWidget()
        self._list.itemSelectionChanged.connect(self._on_list_selection)
        v.addWidget(self._list)
        btns = QtWidgets.QHBoxLayout()
        self._up_btn = QtWidgets.QPushButton("↑")
        self._up_btn.setToolTip("Move selected gate earlier in visit order")
        self._up_btn.clicked.connect(lambda: self._on_reorder("up"))
        self._down_btn = QtWidgets.QPushButton("↓")
        self._down_btn.setToolTip("Move selected gate later in visit order")
        self._down_btn.clicked.connect(lambda: self._on_reorder("down"))
        self._del_btn = QtWidgets.QPushButton("Delete")
        self._del_btn.clicked.connect(self._on_delete_clicked)
        btns.addWidget(self._up_btn)
        btns.addWidget(self._down_btn)
        btns.addStretch(1)
        btns.addWidget(self._del_btn)
        v.addLayout(btns)
        layout.addWidget(list_box, stretch=1)

        layout.addStretch(0)
        self._refresh_meta_widgets()
        self._refresh_selected_widgets()
        self._refresh_list()

    @staticmethod
    def _spin(lo: float, hi: float, step: float, dec: int) -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(dec)
        return s

    # ----- bind -----
    def set_track(self, track: dict) -> None:
        self._track = track
        self._refresh_meta_widgets()
        self._refresh_list()

    def set_selected(self, idx: int | None) -> None:
        self._selected = idx
        self._refresh_selected_widgets()
        self._sync_list_selection()

    def refresh(self) -> None:
        self._refresh_meta_widgets()
        self._refresh_selected_widgets()
        self._refresh_list()

    def default_z(self) -> float:
        return float(self._default_z.value())

    # ----- internals -----
    def _refresh_meta_widgets(self):
        self._block = True
        self._gate_size.setValue(float(self._track.get("gate_size", 0.7)))
        self._bar_thickness.setValue(float(self._track.get("bar_thickness", 0.05)))
        self._loop_closed.setChecked(bool(self._track.get("loop_closed", True)))
        # default_z is a UI-only field carried on the track dict in memory.
        self._default_z.setValue(float(self._track.get("_default_z", 1.5)))
        self._block = False

    def _refresh_selected_widgets(self):
        gates = self._track.get("gates", [])
        idx = self._selected
        enabled = idx is not None and 0 <= idx < len(gates)
        for w in (self._gx, self._gy, self._gz, self._gyaw, self._gauto):
            w.setEnabled(enabled)
        if not enabled:
            return
        g = gates[idx]
        self._block = True
        self._gx.setValue(float(g["x"]))
        self._gy.setValue(float(g["y"]))
        self._gz.setValue(float(g["z"]))
        self._gyaw.setValue(float(np.degrees(g.get("yaw", 0.0))))
        self._gauto.setChecked(bool(g.get("auto_yaw", True)))
        # Yaw spinbox is informational while auto-yaw is on.
        self._gyaw.setEnabled(not self._gauto.isChecked())
        self._block = False

    def _refresh_list(self):
        self._block = True
        self._list.clear()
        for i, g in enumerate(self._track.get("gates", [])):
            self._list.addItem(
                f"{i:>3}  ({g['x']:+.2f}, {g['y']:+.2f}, {g['z']:+.2f})"
            )
        self._block = False
        self._sync_list_selection()

    def _sync_list_selection(self):
        self._block = True
        self._list.clearSelection()
        if self._selected is not None and 0 <= self._selected < self._list.count():
            self._list.setCurrentRow(self._selected)
        self._block = False

    # ----- emitters / slot adapters -----
    def _emit_meta(self):
        if self._block:
            return
        self.track_meta_changed.emit({
            "gate_size": self._gate_size.value(),
            "bar_thickness": self._bar_thickness.value(),
            "loop_closed": self._loop_closed.isChecked(),
            "_default_z": self._default_z.value(),
        })

    def _emit_gate(self):
        if self._block or self._selected is None:
            return
        self._gyaw.setEnabled(not self._gauto.isChecked())
        self.gate_changed.emit(self._selected, {
            "x": self._gx.value(),
            "y": self._gy.value(),
            "z": self._gz.value(),
            "yaw": float(np.radians(self._gyaw.value())),
            "auto_yaw": self._gauto.isChecked(),
        })

    def _on_list_selection(self):
        if self._block:
            return
        rows = [it.row() for it in self._list.selectedIndexes()]
        self.select_requested.emit(rows[0] if rows else None)

    def _on_delete_clicked(self):
        if self._selected is not None:
            self.delete_requested.emit(self._selected)

    def _on_reorder(self, direction: str):
        if self._selected is not None:
            self.reorder_requested.emit(self._selected, direction)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Drone Racing — Custom Track Designer")
        self.resize(1400, 900)

        self._path: Path | None = None
        self._dirty: bool = False

        # Widgets.
        self.top = TopDownCanvas()
        self.preview = PreviewCanvas3D()
        self.panel = PropertyPanel()

        plots = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        plots.addWidget(self.top)
        plots.addWidget(self.preview)
        plots.setStretchFactor(0, 3)
        plots.setStretchFactor(1, 2)

        central = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        central.addWidget(plots)
        central.addWidget(self.panel)
        central.setStretchFactor(0, 4)
        central.setStretchFactor(1, 1)
        self.setCentralWidget(central)

        # Signals.
        self.top.gates_changed.connect(self._on_gates_changed)
        self.top.selection_changed.connect(self._on_top_selection)
        self.top.mouse_moved.connect(self._on_mouse)
        self.panel.track_meta_changed.connect(self._on_meta_changed)
        self.panel.gate_changed.connect(self._on_gate_edit)
        self.panel.select_requested.connect(self._on_panel_select)
        self.panel.delete_requested.connect(self._on_panel_delete)
        self.panel.reorder_requested.connect(self._on_panel_reorder)

        self._build_menu()
        self._build_toolbar()
        self.statusBar().showMessage("Ready")

        # Delete-key shortcut to remove selected gate.
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self,
                        activated=self._delete_selected)

        # Wire initial state into preview/panel.
        self.top.track["_default_z"] = 1.5
        self._sync_panels()

        if initial_path:
            self.open_path(Path(initial_path))

    # ----- menus / toolbar -----
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        m.addAction(self._act("&New", self.new_track, "Ctrl+N"))
        m.addAction(self._act("&Open…", self.open_dialog, "Ctrl+O"))
        m.addAction(self._act("&Save", self.save, "Ctrl+S"))
        m.addAction(self._act("Save &As…", self.save_as, "Ctrl+Shift+S"))
        m.addSeparator()
        m.addAction(self._act("&Quit", self.close, "Ctrl+Q"))

        e = self.menuBar().addMenu("&Edit")
        e.addAction(self._act("Auto-yaw all gates", self._auto_yaw_all))
        e.addAction(self._act("Clear all gates", self._clear_all))

    def _build_toolbar(self):
        tb = self.addToolBar("Mode")
        tb.setObjectName("ModeToolbar")
        tb.setMovable(False)

        self._mode_group = QtGui.QActionGroup(self)
        self._mode_group.setExclusive(True)
        self._add_act = self._mode_act("Add", "add", checked=True)
        self._sel_act = self._mode_act("Select / Move", "select")
        self._del_act = self._mode_act("Delete", "delete")
        for a in (self._add_act, self._sel_act, self._del_act):
            tb.addAction(a)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" Snap (m): "))
        self._snap = QtWidgets.QDoubleSpinBox()
        self._snap.setRange(0.0, 5.0)
        self._snap.setSingleStep(0.1)
        self._snap.setDecimals(2)
        self._snap.valueChanged.connect(self.top.set_snap)
        tb.addWidget(self._snap)

        tb.addWidget(QtWidgets.QLabel(" View half-extent (m): "))
        self._extent = QtWidgets.QDoubleSpinBox()
        self._extent.setRange(2.0, 200.0)
        self._extent.setSingleStep(1.0)
        self._extent.setDecimals(1)
        self._extent.setValue(15.0)
        self._extent.valueChanged.connect(self._on_extent_changed)
        tb.addWidget(self._extent)

    def _mode_act(self, label: str, mode_id: str, checked: bool = False) -> QtGui.QAction:
        a = QtGui.QAction(label, self)
        a.setCheckable(True)
        a.setChecked(checked)
        a.triggered.connect(lambda _checked, m=mode_id: self.top.set_mode(m))
        self._mode_group.addAction(a)
        return a

    def _act(self, label: str, slot, shortcut: str | None = None) -> QtGui.QAction:
        a = QtGui.QAction(label, self)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
        a.triggered.connect(slot)
        return a

    def _on_extent_changed(self, v: float):
        v = float(v)
        self.top.set_bounds(-v, v, -v, v)

    # ----- file operations -----
    def new_track(self):
        if not self._confirm_discard():
            return
        track = track_io.default_track()
        track["_default_z"] = self.panel.default_z()
        self._path = None
        self.top.set_track(track)
        self._sync_panels()
        self._dirty = False
        self._update_title()

    def open_dialog(self):
        if not self._confirm_discard():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open track", str(self._suggest_dir()),
            "Track JSON (*.json);;All files (*)",
        )
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path):
        try:
            track = track_io.load_track(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(e))
            return
        track["_default_z"] = self.panel.default_z()
        self._path = path
        self.top.set_track(track)
        self._sync_panels()
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Loaded {path}", 4000)

    def save(self) -> bool:
        if self._path is None:
            return self.save_as()
        return self._save_to(self._path)

    def save_as(self) -> bool:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save track", str(self._suggest_dir() / "track.json"),
            "Track JSON (*.json)",
        )
        if not path:
            return False
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".json")
        return self._save_to(p)

    def _save_to(self, path: Path) -> bool:
        try:
            track_io.save_track(path, self.top.track)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
            return False
        self._path = path
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {path}", 4000)
        return True

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        SB = QtWidgets.QMessageBox.StandardButton
        r = QtWidgets.QMessageBox.question(
            self, "Unsaved changes", "Discard unsaved changes?",
            SB.Save | SB.Discard | SB.Cancel,
            SB.Cancel,
        )
        if r == SB.Cancel:
            return False
        if r == SB.Save:
            return self.save()
        return True

    # Tracks are always saved to (and opened from) scripts/custom_track/tracks
    # by default. The user can still navigate elsewhere from the file dialog
    # but this is the canonical location.
    TRACKS_DIR = Path(__file__).resolve().parent / "tracks"

    def _suggest_dir(self) -> Path:
        self.TRACKS_DIR.mkdir(parents=True, exist_ok=True)
        return self.TRACKS_DIR

    def closeEvent(self, ev: QtGui.QCloseEvent):
        if self._confirm_discard():
            ev.accept()
        else:
            ev.ignore()

    # ----- edit menu -----
    def _auto_yaw_all(self):
        self.top.auto_yaw_all()

    def _clear_all(self):
        SB = QtWidgets.QMessageBox.StandardButton
        if QtWidgets.QMessageBox.question(
            self, "Clear all", "Remove every gate?",
            SB.Yes | SB.No, SB.No,
        ) == SB.Yes:
            self.top.clear()

    def _delete_selected(self):
        idx = self.top.selected_index()
        if idx is not None:
            self.top.delete_gate(idx)

    # ----- internal sync -----
    def _sync_panels(self):
        track = self.top.track
        self.preview.set_track(track)
        self.panel.set_track(track)

    def _on_gates_changed(self):
        self._dirty = True
        self.preview.set_track(self.top.track)
        self.panel.refresh()
        self._update_title()

    def _on_top_selection(self, idx):
        self.panel.set_selected(idx)

    def _on_meta_changed(self, meta: dict):
        # _default_z is UI-only; carry it on the in-memory track dict but it
        # is stripped on save by track_io.save_track.
        if "_default_z" in meta:
            z = float(meta.pop("_default_z"))
            self.top.set_default_z(z)
            self.top.track["_default_z"] = z
        if meta:
            self.top.update_track_meta(**meta)

    def _on_gate_edit(self, idx: int, fields: dict):
        self.top.update_gate(idx, **fields)

    def _on_panel_select(self, idx):
        self.top.select(idx)

    def _on_panel_delete(self, idx: int):
        self.top.delete_gate(idx)

    def _on_panel_reorder(self, idx: int, direction: str):
        self.top.reorder(idx, direction)

    def _on_mouse(self, x: float, y: float):
        n = len(self.top.track["gates"])
        self.statusBar().showMessage(f"x={x:+.2f}  y={y:+.2f}   gates={n}")

    def _update_title(self):
        name = self._path.name if self._path else "untitled"
        star = " *" if self._dirty else ""
        self.setWindowTitle(f"Drone Racing — Custom Track Designer — {name}{star}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Custom drone-racing track designer.")
    p.add_argument("track", nargs="?", default=None,
                   help="Optional track JSON to open at startup.")
    args = p.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = MainWindow(initial_path=args.track)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
