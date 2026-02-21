import sys
import os
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QDoubleSpinBox, QCheckBox, QGroupBox, QSplitter
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib.colors import LogNorm


class QwrapperNP:
    def __init__(self, theta_range=(-170, 170), L_range=(0.1, 10), fixed_centers=None):
        self.theta_range = theta_range
        self.L_range = L_range

        if fixed_centers is None:
            raise ValueError("Нужно передать фиксированные центры каналов d.")

        self.centers = np.asarray(fixed_centers, dtype=np.float32)

        edges = np.zeros(self.centers.size + 1, dtype=np.float32)
        edges[1:-1] = (self.centers[:-1] + self.centers[1:]) * 0.5
        edges[0] = self.centers[0] - (self.centers[1] - self.centers[0]) * 0.5
        edges[-1] = self.centers[-1] + (self.centers[-1] - self.centers[-2]) * 0.5
        self.edges = edges

        self._cache_shape = None
        self._cache_d_grid = None
        self._cache_mask = None

    def _ensure_grids(self, H, W):
        key = (H, W, self.theta_range, self.L_range)
        if self._cache_shape == key:
            return

        theta_deg = np.linspace(self.theta_range[0], self.theta_range[1], W, dtype=np.float32)
        L_vals = np.linspace(self.L_range[0], self.L_range[1], H, dtype=np.float32)

        theta_rad = np.deg2rad(theta_deg).astype(np.float32)
        L_grid, theta_grid = np.meshgrid(L_vals, theta_rad, indexing="ij")

        denom = 2.0 * np.sin(np.abs(theta_grid) * 0.5, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_grid = L_grid / denom

        mask = np.isfinite(d_grid) & (d_grid <= 7.5)

        self._cache_shape = key
        self._cache_d_grid = d_grid.astype(np.float32, copy=False)
        self._cache_mask = mask

    def tensor_to_d(self, batch_tensor):
        arr = batch_tensor
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        if arr.ndim != 4:
            raise ValueError("Ожидается массив размера [B, 1, H, W]")

        B, C, H, W = arr.shape
        if C != 1:
            raise ValueError("Ожидается канал C=1")

        self._ensure_grids(H, W)

        d_grid = self._cache_d_grid
        mask = self._cache_mask

        d_vals = d_grid[mask].astype(np.float32, copy=False)
        idx = np.searchsorted(self.edges, d_vals, side="right") - 1
        idx = np.clip(idx, 0, self.centers.size - 1).astype(np.int64, copy=False)

        results = []
        for b in range(B):
            I_mat = arr[b, 0].astype(np.float32, copy=False)
            I_vals = I_mat[mask].astype(np.float32, copy=False)

            I_summed = np.zeros(self.centers.size, dtype=np.float32)
            np.add.at(I_summed, idx, I_vals)

            results.append({"d": self.centers.copy(), "I": I_summed})

        return results


def is_mask_array(arr: np.ndarray) -> bool:
    if not isinstance(arr, np.ndarray) or arr.ndim < 2:
        return False
    if arr.dtype == np.bool_:
        return True
    if np.issubdtype(arr.dtype, np.integer):
        u = np.unique(arr.reshape(-1)[: min(arr.size, 200000)])
        return u.size <= 3 and np.all(np.isin(u, [0, 1]))
    return False


def normalize_to_stack(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported array shape {arr.shape}, expected 2D or 3D.")


def compute_d_map(theta_range, lambda_range, H, W):
    theta_min, theta_max = theta_range
    lam_min, lam_max = lambda_range

    theta_vals = np.linspace(theta_min, theta_max, W, dtype=np.float32)
    lam_vals = np.linspace(lam_min, lam_max, H, dtype=np.float32)

    theta_rad = np.deg2rad(np.abs(theta_vals)).astype(np.float32)
    Lam, Theta = np.meshgrid(lam_vals, theta_rad, indexing="ij")

    with np.errstate(divide="ignore", invalid="ignore"):
        d = Lam / (2.0 * np.sin(Theta / 2.0))

    d[~np.isfinite(d)] = np.nan
    return d


def make_fixed_centers(d_min, d_max, n):
    return np.linspace(d_min, d_max, int(n)).astype(np.float32)


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10, 5), dpi=120, constrained_layout=True)
        super().__init__(self.fig)
        self.ax_img = self.fig.add_subplot(1, 2, 1)
        self.ax_prof = self.fig.add_subplot(1, 2, 2)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setWindowTitle("Diffraction labeling tool | NumPy Qwrapper")

        self.diffractions = None
        self.base_masks = None
        self.intervals = []
        self.idx = 0
        self.mode_view = False

        self.fixed_centers = np.linspace(0.05318052, 7.49710258, 1241).astype(np.float32)
        self.d_centers_min = float(self.fixed_centers[0])
        self.d_centers_max = float(self.fixed_centers[-1])
        self.d_centers_n = int(self.fixed_centers.size)

        self.qw = QwrapperNP(theta_range=(-170, 170), L_range=(0.1, 10), fixed_centers=self.fixed_centers)

        self._d_map = None
        self._d_x = None
        self._d_y = None

        self._cbar = None
        self._im2d = None

        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(360)
        left_layout = QVBoxLayout(left)

        btn_row = QGridLayout()
        self.btn_open = QPushButton("Open .npy")
        self.btn_open.clicked.connect(self.open_npy)
        self.btn_save = QPushButton("Save masks .npy")
        self.btn_save.clicked.connect(self.save_masks)
        self.btn_prev = QPushButton("Prev")
        self.btn_prev.clicked.connect(self.prev_item)
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.next_item)
        self.btn_toggle_mode = QPushButton("View masks")
        self.btn_toggle_mode.clicked.connect(self.toggle_mode)
        self.btn_clear_intervals = QPushButton("Clear intervals (current)")
        self.btn_clear_intervals.clicked.connect(self.clear_intervals_current)

        btn_row.addWidget(self.btn_open, 0, 0, 1, 2)
        btn_row.addWidget(self.btn_save, 1, 0, 1, 2)
        btn_row.addWidget(self.btn_prev, 2, 0)
        btn_row.addWidget(self.btn_next, 2, 1)
        btn_row.addWidget(self.btn_toggle_mode, 3, 0, 1, 2)
        btn_row.addWidget(self.btn_clear_intervals, 4, 0, 1, 2)
        left_layout.addLayout(btn_row)

        dgrid_box = QGroupBox("d-grid (fixed_centers)")
        dg = QGridLayout(dgrid_box)
        self.spin_dgrid_n = QDoubleSpinBox()
        self.spin_dgrid_n.setDecimals(0)
        self.spin_dgrid_n.setRange(16, 200000)
        self.spin_dgrid_n.setSingleStep(1)
        self.spin_dgrid_n.setValue(self.d_centers_n)
        self.btn_apply_dgrid = QPushButton("Apply d-grid")
        self.btn_apply_dgrid.clicked.connect(self.apply_d_grid)
        dg.addWidget(QLabel("N (discretization)"), 0, 0)
        dg.addWidget(self.spin_dgrid_n, 0, 1)
        dg.addWidget(self.btn_apply_dgrid, 1, 0, 1, 2)
        left_layout.addWidget(dgrid_box)

        left_layout.addWidget(self._build_ranges_group())
        left_layout.addWidget(self._build_profile_group())
        left_layout.addStretch(1)

        self.lbl_status = QLabel("No data loaded")
        self.lbl_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_status)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.canvas = MplCanvas()
        right_layout.addWidget(self.canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._span = None
        self._connect_span_selector()
        self._render_empty()

    def apply_d_grid(self):
        n = int(self.spin_dgrid_n.value())
        self.d_centers_n = n
        self.fixed_centers = make_fixed_centers(self.d_centers_min, self.d_centers_max, self.d_centers_n)
        self.qw = QwrapperNP(theta_range=(-170, 170), L_range=(0.1, 10), fixed_centers=self.fixed_centers)
        self.spin_d_min.setValue(self.d_centers_min)
        self.spin_d_max.setValue(self.d_centers_max)
        self.refresh_current()

    def _spin(self, val, lo, hi, step, decimals=4):
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.valueChanged.connect(self.refresh_current)
        return s

    def _build_ranges_group(self):
        g = QGroupBox("2D ranges (theta/lambda)")
        grid = QGridLayout(g)

        self.spin_theta_min = self._spin(-170.0, -360.0, 360.0, 1.0, 3)
        self.spin_theta_max = self._spin(170.0, -360.0, 360.0, 1.0, 3)
        self.spin_lam_min = self._spin(0.1, 1e-6, 1e6, 0.1, 4)
        self.spin_lam_max = self._spin(10.0, 1e-6, 1e6, 0.1, 4)

        grid.addWidget(QLabel("theta_min"), 0, 0); grid.addWidget(self.spin_theta_min, 0, 1)
        grid.addWidget(QLabel("theta_max"), 0, 2); grid.addWidget(self.spin_theta_max, 0, 3)
        grid.addWidget(QLabel("lambda_min"), 1, 0); grid.addWidget(self.spin_lam_min, 1, 1)
        grid.addWidget(QLabel("lambda_max"), 1, 2); grid.addWidget(self.spin_lam_max, 1, 3)
        return g

    def _build_profile_group(self):
        g = QGroupBox("1D profile axes")
        grid = QGridLayout(g)

        self.spin_d_min = self._spin(self.d_centers_min, -1e6, 1e6, 0.1, 6)
        self.spin_d_max = self._spin(self.d_centers_max, -1e6, 1e6, 0.1, 6)
        self.spin_I_min = self._spin(0.1, 1e-12, 1e30, 0.1, 6)
        self.spin_I_max = self._spin(1e5, 1e-12, 1e30, 100.0, 6)

        self.chk_logy = QCheckBox("log Y")
        self.chk_logy.setChecked(True)
        self.chk_logy.stateChanged.connect(self.refresh_current)

        grid.addWidget(QLabel("d_min"), 0, 0); grid.addWidget(self.spin_d_min, 0, 1)
        grid.addWidget(QLabel("d_max"), 0, 2); grid.addWidget(self.spin_d_max, 0, 3)
        grid.addWidget(QLabel("I_min"), 1, 0); grid.addWidget(self.spin_I_min, 1, 1)
        grid.addWidget(QLabel("I_max"), 1, 2); grid.addWidget(self.spin_I_max, 1, 3)
        grid.addWidget(self.chk_logy, 2, 0, 1, 2)
        return g

    def _connect_span_selector(self):
        if self._span is not None:
            try:
                self._span.disconnect_events()
            except Exception:
                pass

        def onselect(xmin, xmax):
            if self.mode_view:
                return
            if self.diffractions is None or self.idx >= self.diffractions.shape[0]:
                return
            a, b = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
            self.intervals[self.idx].append((float(a), float(b)))
            self.refresh_current()

        self._span = SpanSelector(
            self.canvas.ax_prof,
            onselect,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="red"),
        )

        def on_click(event):
            if self.mode_view:
                return
            if self.diffractions is None or self.idx >= self.diffractions.shape[0]:
                return
            if event.inaxes != self.canvas.ax_prof:
                return
            if event.button == 3 and self.intervals[self.idx]:
                self.intervals[self.idx].pop()
                self.refresh_current()

        self.canvas.mpl_connect("button_press_event", on_click)

    def _ensure_sizes(self, N, H, W):
        if self.base_masks is None or self.base_masks.shape != (N, H, W):
            self.base_masks = np.zeros((N, H, W), dtype=np.uint8)
        if not self.intervals or len(self.intervals) != N:
            self.intervals = [[] for _ in range(N)]

    def load_npy_path(self, path: str):
        try:
            arr = np.load(path, allow_pickle=False)
            stack = normalize_to_stack(arr)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load npy:\n{path}\n\n{e}")
            return

        if is_mask_array(stack):
            masks = stack.astype(np.uint8)
            if self.diffractions is not None:
                N, H, W = self.diffractions.shape
                self._ensure_sizes(N, H, W)
                nmin = min(N, masks.shape[0])
                hmin = min(H, masks.shape[1])
                wmin = min(W, masks.shape[2])
                self.base_masks[:nmin, :hmin, :wmin] = masks[:nmin, :hmin, :wmin]
            else:
                self.base_masks = masks
                if not self.intervals or len(self.intervals) != masks.shape[0]:
                    self.intervals = [[] for _ in range(masks.shape[0])]
            self.idx = 0
        else:
            self.diffractions = stack.astype(np.float32, copy=False)
            N, H, W = self.diffractions.shape
            self._ensure_sizes(N, H, W)
            self.idx = 0

        self.refresh_current()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for u in event.mimeData().urls():
                if u.isLocalFile() and u.toLocalFile().lower().endswith(".npy"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        paths = []
        for u in event.mimeData().urls():
            if u.isLocalFile():
                p = u.toLocalFile()
                if p.lower().endswith(".npy"):
                    paths.append(p)
        if not paths:
            return
        for p in paths:
            self.load_npy_path(p)
        event.acceptProposedAction()

    def open_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open .npy file", "", "NumPy files (*.npy)")
        if not path:
            return
        self.load_npy_path(path)

    def get_current_masks_stack(self):
        if self.base_masks is None:
            return None
        if self.diffractions is None:
            return self.base_masks.copy()

        N, H, W = self.diffractions.shape
        out = self.base_masks.copy()

        tmin = float(self.spin_theta_min.value())
        tmax = float(self.spin_theta_max.value())
        lmin = float(self.spin_lam_min.value())
        lmax = float(self.spin_lam_max.value())

        d_map = compute_d_map((tmin, tmax), (lmin, lmax), H, W)

        for i in range(min(N, len(self.intervals))):
            interval_mask = np.zeros((H, W), dtype=bool)
            for (a, b) in self.intervals[i]:
                interval_mask |= ((d_map >= a) & (d_map <= b))
            out[i] = (out[i].astype(bool) | interval_mask).astype(np.uint8)

        return out

    def save_masks(self):
        masks = self.get_current_masks_stack()
        if masks is None:
            QMessageBox.information(self, "Nothing to save", "No masks in memory.")
            return

        non_empty_idx = np.where(masks.reshape(masks.shape[0], -1).any(axis=1))[0]
        if non_empty_idx.size == 0:
            QMessageBox.information(self, "Nothing to save", "All masks are empty.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save labeled masks .npy", "masks_labeled.npy", "NumPy files (*.npy)")
        if not path:
            return

        np.save(path, masks[non_empty_idx].astype(np.uint8))
        np.save(path.replace(".npy", "_indices.npy"), non_empty_idx.astype(np.int32))

    def prev_item(self):
        if self.idx > 0:
            self.idx -= 1
            self.refresh_current()

    def next_item(self):
        N = None
        if self.mode_view:
            if self.base_masks is not None:
                N = self.base_masks.shape[0]
        else:
            if self.diffractions is not None:
                N = self.diffractions.shape[0]

        if N is None:
            return

        if self.idx + 1 < N:
            self.idx += 1
            self.refresh_current()
        else:
            self.idx = N
            self._render_empty()
            self._update_status_text()

    def toggle_mode(self):
        self.mode_view = not self.mode_view
        self.btn_toggle_mode.setText("Back to labeling" if self.mode_view else "View masks")
        if self.mode_view and self.base_masks is None and self.diffractions is None:
            self.mode_view = False
            self.btn_toggle_mode.setText("View masks")
        self.refresh_current()

    def clear_intervals_current(self):
        if self.intervals and 0 <= self.idx < len(self.intervals):
            self.intervals[self.idx] = []
            self.refresh_current()

    def refresh_current(self):
        if self.mode_view:
            if self.idx < 0:
                self.idx = 0
            self._render_masks_view_only()
            self._update_status_text()
            return

        if self.diffractions is None or self.idx >= self.diffractions.shape[0]:
            self._render_empty()
            self._update_status_text()
            return

        self._recompute_maps_and_profile()
        self._render_label_mode()
        self._update_status_text()

    def _recompute_maps_and_profile(self):
        diff = self.diffractions[self.idx]
        H, W = diff.shape

        tmin = float(self.spin_theta_min.value())
        tmax = float(self.spin_theta_max.value())
        lmin = float(self.spin_lam_min.value())
        lmax = float(self.spin_lam_max.value())

        self._d_map = compute_d_map((tmin, tmax), (lmin, lmax), H, W)

        tens = diff.astype(np.float32, copy=False)[None, None, :, :]
        out = self.qw.tensor_to_d(tens)[0]
        self._d_x = out["d"]
        self._d_y = out["I"]

    def _current_final_mask(self):
        if self.diffractions is not None:
            H, W = self.diffractions[self.idx].shape
        elif self.base_masks is not None and 0 <= self.idx < self.base_masks.shape[0]:
            H, W = self.base_masks[self.idx].shape
        else:
            return None

        if self.base_masks is not None and 0 <= self.idx < self.base_masks.shape[0]:
            base = self.base_masks[self.idx].astype(bool)
        else:
            base = np.zeros((H, W), dtype=bool)

        if self._d_map is None and self.diffractions is not None:
            tmin = float(self.spin_theta_min.value())
            tmax = float(self.spin_theta_max.value())
            lmin = float(self.spin_lam_min.value())
            lmax = float(self.spin_lam_max.value())
            self._d_map = compute_d_map((tmin, tmax), (lmin, lmax), H, W)

        if self.intervals and 0 <= self.idx < len(self.intervals) and self._d_map is not None:
            interval_mask = np.zeros((H, W), dtype=bool)
            for (a, b) in self.intervals[self.idx]:
                interval_mask |= ((self._d_map >= a) & (self._d_map <= b))
            return (base | interval_mask).astype(np.uint8)

        return base.astype(np.uint8)

    def _render_empty(self):
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self._im2d = None

        self.canvas.ax_img.clear()
        self.canvas.ax_prof.clear()
        self.canvas.ax_img.set_title("No data")
        self.canvas.ax_prof.set_title("No data")
        self.canvas.draw_idle()

    def _render_masks_view_only(self):
        self.canvas.ax_img.clear()
        self.canvas.ax_prof.clear()

        mask = self._current_final_mask()
        if mask is None:
            self._render_empty()
            return

        tmin = float(self.spin_theta_min.value())
        tmax = float(self.spin_theta_max.value())
        lmin = float(self.spin_lam_min.value())
        lmax = float(self.spin_lam_max.value())

        self.canvas.ax_img.imshow(
            mask.astype(np.uint8),
            cmap="gray",
            vmin=0,
            vmax=1,
            aspect="auto",
            extent=(tmin, tmax, lmax, lmin),
        )

        self.canvas.ax_img.set_title("MASK VIEW")
        self.canvas.ax_img.set_xlabel("theta")
        self.canvas.ax_img.set_ylabel("Lambda")
        self.canvas.ax_img.set_box_aspect(1)

        self.canvas.draw_idle()

    def _render_label_mode(self):
        diff = self.diffractions[self.idx]
        H, W = diff.shape

        tmin = float(self.spin_theta_min.value())
        tmax = float(self.spin_theta_max.value())
        lmin = float(self.spin_lam_min.value())
        lmax = float(self.spin_lam_max.value())

        final_mask = self._current_final_mask()

        self.canvas.ax_img.clear()

        data = diff.astype(np.float32, copy=False)
        pos = data[data > 0]
        vmin = float(np.percentile(pos, 1)) if pos.size else 1e-6
        vmax = float(np.percentile(pos, 99)) if pos.size else max(vmin * 10.0, 1.0)

        self._im2d = self.canvas.ax_img.imshow(
            data,
            cmap="viridis",
            norm=LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * 1.01)),
            aspect="auto",
            extent=(tmin, tmax, lmax, lmin),
        )

        if final_mask is not None and final_mask.shape == (H, W):
            self.canvas.ax_img.imshow(
                final_mask,
                cmap="Reds",
                vmin=0, vmax=1,
                alpha=0.55,
                aspect="auto",
                extent=(tmin, tmax, lmax, lmin),
            )

        self.canvas.ax_img.set_title("2D diffraction + overlay")
        self.canvas.ax_img.set_xlabel("theta")
        self.canvas.ax_img.set_ylabel("Lambda")
        self.canvas.ax_img.set_box_aspect(1)

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self._cbar = self.canvas.fig.colorbar(
            self._im2d,
            ax=self.canvas.ax_img,
            fraction=0.03,
            pad=0.02,
            shrink=0.701,
            aspect=40,
            anchor=(0.0, 0.5),
            panchor=(1.0, 0.5),
        )

        self.canvas.ax_prof.clear()
        if self._d_x is not None and self._d_y is not None:
            self.canvas.ax_prof.plot(self._d_x, self._d_y)

        dmin = float(self.spin_d_min.value())
        dmax = float(self.spin_d_max.value())
        Imin = float(self.spin_I_min.value())
        Imax = float(self.spin_I_max.value())

        self.canvas.ax_prof.set_xlim(dmin, dmax)

        has_pos = bool(self._d_y is not None and np.any(self._d_y > 0))
        want_log = self.chk_logy.isChecked() and has_pos and (Imin > 0) and (Imax > 0)

        if want_log:
            self.canvas.ax_prof.set_yscale("log")
            self.canvas.ax_prof.set_ylim(max(Imin, 1e-12), max(Imax, max(Imin, 1e-12) * 10))
        else:
            self.canvas.ax_prof.set_yscale("linear")
            if Imax <= Imin:
                Imax = Imin + 1.0
            self.canvas.ax_prof.set_ylim(Imin, Imax)

        self.canvas.ax_prof.set_title("Select intervals with LMB, remove last with RMB")
        self.canvas.ax_prof.set_xlabel("d, Å")
        self.canvas.ax_prof.set_ylabel("Intensity")
        self.canvas.ax_prof.set_box_aspect(1)

        self.canvas.draw_idle()

    def _update_status_text(self):
        if self.mode_view:
            if self.base_masks is None:
                self.lbl_status.setText("MASK VIEW | computed mask (unsaved) or none loaded")
                return
            N = self.base_masks.shape[0]
            if 0 <= self.idx < N:
                self.lbl_status.setText(f"MASK VIEW | {self.idx+1}/{N}")
            else:
                self.lbl_status.setText("MASK VIEW | out of range")
            return

        if self.diffractions is None:
            self.lbl_status.setText("LABEL | no diffractions loaded")
            return

        N = self.diffractions.shape[0]
        if 0 <= self.idx < N:
            k = len(self.intervals[self.idx]) if self.intervals else 0
            self.lbl_status.setText(f"LABEL | {self.idx+1}/{N} | intervals={k}")
        else:
            self.lbl_status.setText("LABEL | out of range")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1500, 850)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
