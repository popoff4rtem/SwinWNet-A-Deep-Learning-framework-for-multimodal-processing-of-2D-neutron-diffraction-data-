import sys
import os
import csv
import numpy as np

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon, QPainter, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QDoubleSpinBox, QSpinBox, QGroupBox, QSplitter, QScrollArea,
    QFrame, QToolButton, QCheckBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
from matplotlib.colors import LogNorm

import torch
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")


APP_DIR = os.path.dirname(os.path.abspath(__file__))

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    PROJECT_ROOT = sys._MEIPASS
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------
# utils: scaling + colormap
# ----------------------------
def _norm01(a2d: np.ndarray, vmin=None, vmax=None, do_log=False) -> np.ndarray:
    x = np.asarray(a2d, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)

    if do_log:
        x = np.log1p(x)

    if vmin is None:
        vmin = np.nanpercentile(x, 1)
    if vmax is None:
        vmax = np.nanpercentile(x, 99)

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    y = (x - vmin) / (vmax - vmin)
    return np.clip(y, 0.0, 1.0)


def _rgba_qimage_from_norm01(norm01: np.ndarray, cmap_name="viridis") -> QImage:
    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(norm01) * 255.0).astype(np.uint8)  # H,W,4
    h, w, _ = rgba.shape
    qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return qimg.copy()


def _gray_qimage_from_mask(mask2d: np.ndarray) -> QImage:
    m = np.asarray(mask2d, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    m = (m > 0.5).astype(np.uint8) * 255
    h, w = m.shape
    qimg = QImage(m.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()


def _is_seg_mask_stage(stage: str) -> bool:
    s = stage.lower()
    return s in ("seg_map_lr", "seg_map_hr") or s.startswith("seg_map")


def make_stage_icon(sample_chw: np.ndarray, size: int, is_mask_stage: bool, log2d: bool) -> QIcon:
    arr = np.asarray(sample_chw)
    if arr.ndim == 2:
        panels = [arr]
    else:
        panels = [arr[0], arr[1]] if arr.shape[0] >= 2 else [arr[0]]

    pix = []
    for p in panels:
        if is_mask_stage:
            qi = _gray_qimage_from_mask(p)
        else:
            n = _norm01(p, do_log=log2d)
            qi = _rgba_qimage_from_norm01(n, "viridis")

        pm = QPixmap.fromImage(qi).scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pix.append(pm)

    total_w = sum(pm.width() for pm in pix)
    total_h = max(pm.height() for pm in pix)
    out = QPixmap(total_w, total_h)
    out.fill(Qt.transparent)

    qp = QPainter(out)
    x0 = 0
    for pm in pix:
        qp.drawPixmap(x0, 0, pm)
        x0 += pm.width()
    qp.end()

    return QIcon(out)


def _as_4d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x[None, None, ...]
    if x.ndim == 3:
        return x[:, None, ...]
    if x.ndim == 4:
        return x
    raise ValueError(f"Unsupported array shape: {x.shape}")


# ----------------------------
# weights loading
# ----------------------------
def load_state_dict_any(pth_path: str, map_location="cpu") -> dict:
    ckpt = torch.load(pth_path, map_location=map_location)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def infer_error_matrix_flag_from_sd(sd: dict) -> bool:
    k = "patch_embed.proj.weight"
    if k in sd and hasattr(sd[k], "shape"):
        in_ch = int(sd[k].shape[1])
        return in_ch >= 2
    return False


# ----------------------------
# Qwrapper (numpy) -> 1D I(d)
# ----------------------------
class QwrapperNP:
    def __init__(self, theta_range=(-170, 170), L_range=(0.1, 10), fixed_centers=None):
        self.theta_range = theta_range
        self.L_range = L_range
        if fixed_centers is None:
            raise ValueError("fixed_centers must be provided")

        self.centers = np.asarray(fixed_centers, dtype=np.float32)

        edges = np.zeros(self.centers.size + 1, dtype=np.float32)
        edges[1:-1] = (self.centers[:-1] + self.centers[1:]) * 0.5
        edges[0] = self.centers[0] - (self.centers[1] - self.centers[0]) * 0.5
        edges[-1] = self.centers[-1] + (self.centers[-1] - self.centers[-2]) * 0.5
        self.edges = edges

        self._cache_key = None
        self._cache_d_grid = None
        self._cache_mask = None

    def _ensure_grids(self, H, W):
        key = (H, W, self.theta_range, self.L_range)
        if self._cache_key == key:
            return

        theta_deg = np.linspace(self.theta_range[0], self.theta_range[1], W, dtype=np.float32)
        L_vals = np.linspace(self.L_range[0], self.L_range[1], H, dtype=np.float32)

        theta_rad = np.deg2rad(theta_deg).astype(np.float32)
        L_grid, theta_grid = np.meshgrid(L_vals, theta_rad, indexing="ij")

        denom = 2.0 * np.sin(np.abs(theta_grid) * 0.5, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_grid = L_grid / denom

        mask = np.isfinite(d_grid) & (d_grid <= 7.5)

        self._cache_key = key
        self._cache_d_grid = d_grid.astype(np.float32, copy=False)
        self._cache_mask = mask

    def tensor_to_d(self, batch_tensor):
        arr = np.asarray(batch_tensor)
        if arr.ndim != 4:
            raise ValueError("Expected array [B,1,H,W] or [B,?,H,W]")

        B, C, H, W = arr.shape
        if C < 1:
            raise ValueError("No channels")

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


def make_fixed_centers(d_min, d_max, n):
    return np.linspace(float(d_min), float(d_max), int(n)).astype(np.float32)


# ----------------------------
# UI widgets
# ----------------------------
class Mpl1DCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 3.2), dpi=120, constrained_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)


class StageIconButton(QToolButton):
    def __init__(self, stage_name: str):
        super().__init__()
        self.stage_name = stage_name
        self.setCheckable(True)
        self.setToolTip(stage_name)
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.setIconSize(QSize(64, 64))
        self.setFixedSize(QSize(74, 74))


class TileWidget(QFrame):
    def __init__(
        self,
        title: str,
        left_img: np.ndarray,
        right_img: np.ndarray | None,
        theta_range=(-170, 170),
        lam_range=(0.1, 10),
        left_title="Diffraction",
        right_title="Error",
        log2d=False,
        cmap_name="viridis",
    ):
        super().__init__()
        self.theta_range = theta_range
        self.lam_range = lam_range
        self.left_title = left_title
        self.right_title = right_title
        self.log2d = log2d
        self.cmap_name = cmap_name

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        lbl = QLabel(title)
        lbl.setWordWrap(True)
        outer.addWidget(lbl)

        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setStyleSheet("QFrame { border: 1px solid #333; border-radius: 6px; }")

        self.fig = Figure(figsize=(6.2, 2.9), dpi=120, constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        outer.addWidget(self.canvas)

        self.set_images(left_img, right_img)

    def _extent(self):
        th0, th1 = self.theta_range
        l0, l1 = self.lam_range
        return (th0, th1, l0, l1)

    def _imshow_2d(self, ax, x2d: np.ndarray, title: str, kind: str):
        ax.clear()
        ext = self._extent()

        if kind == "mask":
            m = np.asarray(x2d, dtype=np.float32)
            m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
            im = ax.imshow(m, cmap="gray", aspect="auto", extent=ext, vmin=0.0, vmax=1.0)
        else:
            x = np.asarray(x2d, dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.maximum(x, 0.0)

            vmin = np.nanpercentile(x, 1)
            vmax = np.nanpercentile(x, 99)
            if not np.isfinite(vmin) or vmin <= 0:
                vmin = 1e-6
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin * 10.0

            if self.log2d:
                im = ax.imshow(x, cmap=self.cmap_name, aspect="auto", extent=ext,
                               norm=LogNorm(vmin=vmin, vmax=vmax))
            else:
                im = ax.imshow(x, cmap=self.cmap_name, aspect="auto", extent=ext,
                               vmin=vmin, vmax=vmax)

        ax.set_xlabel("theta, deg")
        ax.set_ylabel("Lambda, Å")
        ax.set_title(title, fontsize=9)
        return im

    def set_images(self, left_img: np.ndarray, right_img: np.ndarray | None):
        self.fig.clear()

        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)

        if right_img is None:
            kind = "mask" if self.left_title.lower() == "mask" else "image"
            im1 = self._imshow_2d(ax1, left_img, self.left_title, kind=kind)
            ax2.axis("off")
            self.fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        else:
            im1 = self._imshow_2d(ax1, left_img, self.left_title, kind="image")
            im2 = self._imshow_2d(ax2, right_img, self.right_title, kind="image")
            self.fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            self.fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        self.canvas.draw()

# ----------------------------
# Main window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setWindowTitle("Inference GUI")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.infer = None
        self.current_images = None  # torch.Tensor [B,1/2,H,W]

        self.data = {}
        self.stage_order = []
        self.selected_stages = []

        self.fixed_centers = np.linspace(0.05318052, 7.49710258, 1241).astype(np.float32)
        self.qw = QwrapperNP(theta_range=(-170, 170), L_range=(0.1, 10), fixed_centers=self.fixed_centers)

        self.lines = {}
        self.bands = {}
        self._legend = None
        self._legend_map = {}  # Text -> (kind, stage)

        self.stage_visible = {}  # stage -> bool
        self.band_visible = {}   # stage -> bool (DEFAULT OFF)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(380)
        left_layout = QVBoxLayout(left)

        file_box = QGroupBox("I/O")
        fg = QGridLayout(file_box)
        self.btn_open = QPushButton("Open .npy (dict or array)")
        self.btn_open.clicked.connect(self.open_file)
        self.btn_export_csv = QPushButton("Export visible curves to CSV")
        self.btn_export_csv.clicked.connect(self.export_csv)
        fg.addWidget(self.btn_open, 0, 0, 1, 2)
        fg.addWidget(self.btn_export_csv, 1, 0, 1, 2)
        left_layout.addWidget(file_box)

        w_box = QGroupBox("Model")
        wg = QGridLayout(w_box)
        self.btn_load_weights = QPushButton("Load weights (.pth/.pt)")
        self.btn_load_weights.clicked.connect(self.action_load_weights)
        self.btn_run_infer = QPushButton("Run inference on loaded images")
        self.btn_run_infer.clicked.connect(self.run_inference_and_refresh)
        wg.addWidget(self.btn_load_weights, 0, 0, 1, 2)
        wg.addWidget(self.btn_run_infer, 1, 0, 1, 2)
        left_layout.addWidget(w_box)

        view_box = QGroupBox("View")
        vg = QGridLayout(view_box)

        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(0, 0)
        self.spin_sample.valueChanged.connect(self._on_sample_changed)

        self.spin_tiles_per_stage = QSpinBox()
        self.spin_tiles_per_stage.setRange(1, 64)
        self.spin_tiles_per_stage.setValue(1)
        self.spin_tiles_per_stage.valueChanged.connect(self.refresh_all)

        self.chk_norm_1d = QCheckBox("Normalize 1D (each curve / max)")
        self.chk_norm_1d.setChecked(False)
        self.chk_norm_1d.stateChanged.connect(self.refresh_all)

        self.chk_log_2d = QCheckBox("Log scale for 2D")
        self.chk_log_2d.setChecked(False)
        self.chk_log_2d.stateChanged.connect(self._on_log2d_changed)

        self.chk_auto_ylim = QCheckBox("Auto Y")
        self.chk_auto_ylim.setChecked(True)
        self.chk_auto_ylim.stateChanged.connect(self.refresh_all)

        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-1e12, 1e12)
        self.spin_ymin.setDecimals(6)
        self.spin_ymin.setValue(-1.0)
        self.spin_ymin.valueChanged.connect(self.refresh_all)

        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-1e12, 1e12)
        self.spin_ymax.setDecimals(6)
        self.spin_ymax.setValue(1.0)
        self.spin_ymax.valueChanged.connect(self.refresh_all)

        vg.addWidget(self.chk_auto_ylim, 4, 0, 1, 2)
        vg.addWidget(QLabel("y_min"), 5, 0);
        vg.addWidget(self.spin_ymin, 5, 1)
        vg.addWidget(QLabel("y_max"), 6, 0);
        vg.addWidget(self.spin_ymax, 6, 1)

        vg.addWidget(QLabel("Sample index (B)"), 0, 0)
        vg.addWidget(self.spin_sample, 0, 1)
        vg.addWidget(QLabel("Tiles per stage"), 1, 0)
        vg.addWidget(self.spin_tiles_per_stage, 1, 1)
        vg.addWidget(self.chk_norm_1d, 2, 0, 1, 2)
        vg.addWidget(self.chk_log_2d, 3, 0, 1, 2)
        left_layout.addWidget(view_box)

        dgrid_box = QGroupBox("d-grid")
        dg = QGridLayout(dgrid_box)
        self.spin_d_min = QDoubleSpinBox(); self.spin_d_min.setDecimals(6); self.spin_d_min.setRange(1e-6, 1e6); self.spin_d_min.setValue(float(self.fixed_centers[0]))
        self.spin_d_max = QDoubleSpinBox(); self.spin_d_max.setDecimals(6); self.spin_d_max.setRange(1e-6, 1e6); self.spin_d_max.setValue(float(self.fixed_centers[-1]))
        self.spin_d_n = QSpinBox(); self.spin_d_n.setRange(16, 200000); self.spin_d_n.setValue(int(self.fixed_centers.size))
        self.btn_apply_dgrid = QPushButton("Apply d-grid")
        self.btn_apply_dgrid.clicked.connect(self.apply_d_grid)
        dg.addWidget(QLabel("d_min"), 0, 0); dg.addWidget(self.spin_d_min, 0, 1)
        dg.addWidget(QLabel("d_max"), 1, 0); dg.addWidget(self.spin_d_max, 1, 1)
        dg.addWidget(QLabel("N"), 2, 0); dg.addWidget(self.spin_d_n, 2, 1)
        dg.addWidget(self.btn_apply_dgrid, 3, 0, 1, 2)
        left_layout.addWidget(dgrid_box)

        geo_box = QGroupBox("2D geometry (theta / lambda)")
        gg = QGridLayout(geo_box)
        self.spin_theta_min = QDoubleSpinBox(); self.spin_theta_min.setDecimals(3); self.spin_theta_min.setRange(-360, 360); self.spin_theta_min.setValue(-170.0)
        self.spin_theta_max = QDoubleSpinBox(); self.spin_theta_max.setDecimals(3); self.spin_theta_max.setRange(-360, 360); self.spin_theta_max.setValue(170.0)
        self.spin_lam_min = QDoubleSpinBox(); self.spin_lam_min.setDecimals(6); self.spin_lam_min.setRange(1e-9, 1e6); self.spin_lam_min.setValue(0.1)
        self.spin_lam_max = QDoubleSpinBox(); self.spin_lam_max.setDecimals(6); self.spin_lam_max.setRange(1e-9, 1e6); self.spin_lam_max.setValue(10.0)
        for s in (self.spin_theta_min, self.spin_theta_max, self.spin_lam_min, self.spin_lam_max):
            s.valueChanged.connect(self.refresh_all)
        gg.addWidget(QLabel("theta_min"), 0, 0); gg.addWidget(self.spin_theta_min, 0, 1)
        gg.addWidget(QLabel("theta_max"), 0, 2); gg.addWidget(self.spin_theta_max, 0, 3)
        gg.addWidget(QLabel("lambda_min"), 1, 0); gg.addWidget(self.spin_lam_min, 1, 1)
        gg.addWidget(QLabel("lambda_max"), 1, 2); gg.addWidget(self.spin_lam_max, 1, 3)
        left_layout.addWidget(geo_box)

        left_layout.addStretch(1)
        self.lbl_status = QLabel("Drop a .npy or open file.")
        self.lbl_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_status)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.icons_row = QHBoxLayout()
        self.icons_row.setSpacing(6)
        right_layout.addLayout(self.icons_row)

        self.tiles_scroll = QScrollArea()
        self.tiles_scroll.setWidgetResizable(True)
        tiles_root = QWidget()
        self.tiles_grid = QGridLayout(tiles_root)
        self.tiles_grid.setSpacing(8)
        self.tiles_scroll.setWidget(tiles_root)
        right_layout.addWidget(self.tiles_scroll, 3)

        self.plot = Mpl1DCanvas()
        right_layout.addWidget(self.plot, 2)
        self.plot.ax.set_title("I(d) | click legend items to hide/show")
        self.plot.ax.set_xlabel("d, Å")
        self.plot.ax.set_ylabel("Intensity / Mask sum")

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.plot.mpl_connect("pick_event", self._on_pick_legend)

        act_open = QAction("Open .npy", self)
        act_open.triggered.connect(self.open_file)
        act_w = QAction("Load weights (.pth/.pt)", self)
        act_w.triggered.connect(self.action_load_weights)
        self.menuBar().addAction(act_open)
        self.menuBar().addAction(act_w)

    def _on_sample_changed(self, *_):
        self.refresh_all()
        self._rebuild_icons()

    def _on_log2d_changed(self, *_):
        self.refresh_all()
        self._rebuild_icons()

    # ---------- DnD ----------
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
        if paths:
            self.load_npy(paths[-1])
            event.acceptProposedAction()

    # ---------- IO ----------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open .npy", "", "NumPy files (*.npy)")
        if path:
            self.load_npy(path)

    def _to_torch_images(self, x: np.ndarray) -> torch.Tensor:
        arr = np.asarray(x)
        if arr.ndim == 2:
            arr = arr[None, None, ...]
        elif arr.ndim == 3:
            arr = arr[:, None, ...]
        elif arr.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported images shape: {arr.shape}")
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def _sorted_stage_names(self, names: list[str]) -> list[str]:
        preferred = [
            "images",
            "seg_map_lr",
            "images_masked_lr",
            "norm",
            "upscaled_norm",
            "upscaled_denorm",
            "seg_map_hr",
            "images_masked_hr",
        ]
        out = []
        used = set()
        for p in preferred:
            if p in names and p not in used:
                out.append(p); used.add(p)
        for n in sorted(names):
            if n not in used:
                out.append(n)
        return out

    def _infer_batch_size(self) -> int:
        b = 1
        for n in self.stage_order:
            x = np.asarray(self.data[n])
            if x.ndim == 4:
                b = max(b, x.shape[0])
            elif x.ndim == 3:
                b = max(b, x.shape[0])
        return b

    def load_npy(self, path: str):
        try:
            obj = np.load(path, allow_pickle=True)
            item = obj.item() if getattr(obj, "shape", None) == () else obj
            if isinstance(item, dict):
                data = item
            else:
                data = {"images": item}
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load:\n{path}\n\n{e}")
            return

        self.data = {}
        self.stage_order = []
        self.selected_stages = []
        self.lines = {}
        self.bands = {}
        self._legend = None
        self._legend_map = {}
        self.stage_visible = {}
        self.band_visible = {}
        self.current_images = None

        if isinstance(data, dict):
            if "images" in data:
                try:
                    x = np.asarray(data["images"])
                    if x.ndim in (2, 3, 4):
                        self.current_images = self._to_torch_images(x)
                except Exception:
                    pass

            for k, v in data.items():
                if v is None:
                    continue
                try:
                    vv = np.asarray(v)
                    if vv.ndim in (2, 3, 4):
                        self.data[str(k)] = vv
                except Exception:
                    pass

        if not self.data and self.current_images is None:
            QMessageBox.information(self, "No data", "No 2D/3D/4D arrays found in file.")
            return

        if self.data:
            self.stage_order = self._sorted_stage_names(list(self.data.keys()))
            bmax = self._infer_batch_size()
            self.spin_sample.setRange(0, max(0, bmax - 1))
            self.spin_sample.setValue(0)

            for st in self.stage_order:
                self.stage_visible[st] = True
                self.band_visible[st] = False  # DEFAULT OFF

            self._rebuild_icons()

        base = os.path.basename(path)
        msg = f"Loaded: {base} | device={self.device}"
        if self.current_images is not None:
            msg += f" | images_torch={tuple(self.current_images.shape)}"
        if self.data:
            msg += f" | stages={len(self.stage_order)}"
        self.lbl_status.setText(msg)

        self.refresh_all()

    # ---------- icons ----------
    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _rebuild_icons(self):
        self._clear_layout(self.icons_row)
        if not self.stage_order:
            self.icons_row.addStretch(1)
            return

        sample_idx = int(self.spin_sample.value()) if self.spin_sample.maximum() >= 0 else 0
        log2d = self.chk_log_2d.isChecked()

        for name in self.stage_order:
            btn = StageIconButton(name)
            try:
                x = _as_4d(self.data[name])
                b = max(0, min(sample_idx, x.shape[0] - 1))
                icon = make_stage_icon(x[b], size=64, is_mask_stage=_is_seg_mask_stage(name), log2d=log2d)
                btn.setIcon(icon)
            except Exception:
                pass

            btn.toggled.connect(lambda checked, n=name: self.toggle_stage(n, checked))

            btn.blockSignals(True)
            btn.setChecked(name in self.selected_stages)
            btn.blockSignals(False)

            self.icons_row.addWidget(btn)

        self.icons_row.addStretch(1)

    def toggle_stage(self, name: str, checked: bool):
        if checked:
            if name not in self.selected_stages:
                self.selected_stages.append(name)
        else:
            if name in self.selected_stages:
                self.selected_stages.remove(name)
        self.refresh_all()

    # ---------- weights / inference ----------
    def action_load_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load weights", "", "PyTorch weights (*.pth *.pt)")
        if not path:
            return
        try:
            sd = load_state_dict_any(path, map_location="cpu")
            use_err = infer_error_matrix_flag_from_sd(sd)

            try:
                from SwinWNet import SwinWNet
            except Exception as e:
                raise ImportError(
                    "Cannot import SwinWNet. Run from repo root or fix sys.path.\n"
                    f"__file__={__file__}\nAPP_DIR={APP_DIR}\nPROJECT_ROOT={PROJECT_ROOT}\n"
                    f"sys.path[0:5]={sys.path[:5]}\n\nOriginal error: {e}"
                )

            model = SwinWNet(error_matrix=use_err, in_chans=1)
            missing, unexpected = model.load_state_dict(sd, strict=False)

            try:
                from ST_Inference_Pipline import SwinWNetInference
            except Exception as e:
                raise ImportError(
                    "Cannot import ST_Inference_Pipline. Same reason: module not on sys.path.\n"
                    f"Original error: {e}"
                )

            self.model = model
            self.infer = SwinWNetInference(self.model, self.device)

            QMessageBox.information(
                self,
                "Weights loaded",
                f"{os.path.basename(path)}\nerror_matrix={use_err}\nmissing={len(missing)} unexpected={len(unexpected)}\ndevice={self.device}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load weights:\n{e}")

    def run_inference_and_refresh(self):
        if self.infer is None:
            QMessageBox.information(self, "No model", "Load weights first.")
            return
        if self.current_images is None:
            QMessageBox.information(self, "No images", "Load images (.npy) first (array or dict with key 'images').")
            return
        try:
            images = self.current_images.to(self.device)
            _ = self.infer(images)

            out = {
                "images": self.infer.images,
                "seg_map_lr": self.infer.seg_map_lr,
                "images_masked_lr": self.infer.images_masked_lr,
                "norm": self.infer.norm,
                "upscaled_norm": self.infer.upscaled_norm,
                "upscaled_denorm": self.infer.upscaled_denorm,
                "seg_map_hr": self.infer.seg_map_hr,
                "images_masked_hr": self.infer.images_masked_hr,
            }

            self.data = {}
            for k, v in out.items():
                if v is None:
                    continue
                self.data[k] = v.detach().cpu().numpy()

            self.stage_order = self._sorted_stage_names(list(self.data.keys()))
            self.selected_stages = []

            self.lines = {}
            self.bands = {}
            self._legend = None
            self._legend_map = {}
            self.stage_visible = {}
            self.band_visible = {}

            bmax = self._infer_batch_size()
            self.spin_sample.setRange(0, max(0, bmax - 1))
            self.spin_sample.setValue(0)

            for st in self.stage_order:
                self.stage_visible[st] = True
                self.band_visible[st] = False  # DEFAULT OFF

            self._rebuild_icons()
            self.lbl_status.setText(f"Inference done | stages={len(self.stage_order)} | device={self.device}")
            self.refresh_all()
        except Exception as e:
            QMessageBox.critical(self, "Inference error", f"{e}")

    # ---------- d-grid ----------
    def apply_d_grid(self):
        dmin = float(self.spin_d_min.value())
        dmax = float(self.spin_d_max.value())
        n = int(self.spin_d_n.value())
        self.fixed_centers = make_fixed_centers(dmin, dmax, n)
        self.qw = QwrapperNP(
            theta_range=(float(self.spin_theta_min.value()), float(self.spin_theta_max.value())),
            L_range=(float(self.spin_lam_min.value()), float(self.spin_lam_max.value())),
            fixed_centers=self.fixed_centers
        )
        self.refresh_all()

    def _refresh_qwrapper_geometry(self):
        self.qw.theta_range = (float(self.spin_theta_min.value()), float(self.spin_theta_max.value()))
        self.qw.L_range = (float(self.spin_lam_min.value()), float(self.spin_lam_max.value()))

    # ---------- rendering ----------
    def refresh_all(self):
        self._refresh_qwrapper_geometry()
        self._render_tiles()
        self._render_plot()

    def _render_tiles(self):
        while self.tiles_grid.count():
            item = self.tiles_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        if not self.selected_stages:
            return

        sample_idx = int(self.spin_sample.value())
        per_stage = int(self.spin_tiles_per_stage.value())
        log2d = self.chk_log_2d.isChecked()

        row = 0
        col = 0
        max_cols = 2

        for stage in self.selected_stages:
            x = _as_4d(self.data[stage])
            B, C, _, _ = x.shape

            start = max(0, min(sample_idx, B - 1))
            end = min(B, start + per_stage)

            for b in range(start, end):
                title = f"{stage} | b={b}"

                is_seg = _is_seg_mask_stage(stage)

                if (C >= 2) and (not is_seg):
                    left_img = x[b, 0]
                    right_img = x[b, 1]
                    left_title = "Diffraction"
                    right_title = "Error"
                else:
                    left_img = x[b, 0]
                    right_img = None
                    left_title = "Mask" if is_seg else "Diffraction"
                    right_title = ""

                w = TileWidget(
                    title, left_img, right_img,
                    theta_range=(float(self.spin_theta_min.value()), float(self.spin_theta_max.value())),
                    lam_range=(float(self.spin_lam_min.value()), float(self.spin_lam_max.value())),
                    left_title=left_title,
                    right_title=right_title,
                    log2d=log2d,
                    cmap_name="viridis",
                )
                self.tiles_grid.addWidget(w, row, col)

                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

            if col != 0:
                col = 0
                row += 1

        self.tiles_grid.setRowStretch(row + 1, 1)

    def _render_plot(self):
        import matplotlib.patches as mpatches

        ax = self.plot.ax
        ax.clear()
        ax.set_title("I(d) | click legend items to hide/show")
        ax.set_xlabel("d, Å")
        ax.set_ylabel("Intensity / Mask sum")

        self.lines = {}
        self.bands = {}
        self._legend = None
        self._legend_items = {}

        if not self.selected_stages:
            self.plot.draw()
            return

        sample_idx = int(self.spin_sample.value())
        norm1d = self.chk_norm_1d.isChecked()

        for stage in self.selected_stages:
            x4 = _as_4d(self.data[stage]).astype(np.float32, copy=False)
            B, C, _, _ = x4.shape
            b = max(0, min(sample_idx, B - 1))

            I_pack = self.qw.tensor_to_d(x4[b:b + 1, 0:1])[0]
            d = I_pack["d"]
            I = np.nan_to_num(I_pack["I"])

            err = None
            if C >= 2:
                E_pack = self.qw.tensor_to_d(x4[b:b + 1, 1:2])[0]
                err = np.abs(np.nan_to_num(E_pack["I"]))

            if norm1d:
                m = float(np.max(I)) if I.size else 0.0
                if m > 0:
                    I /= m
                    if err is not None:
                        err /= m

            line, = ax.plot(d, I, label=stage)
            self.lines[stage] = line
            line.set_visible(self.stage_visible.get(stage, True))

            if err is not None:
                low = I - err
                high = I + err

                band = ax.fill_between(
                    d, low, high,
                    color=line.get_color(),
                    alpha=0.2,
                    linewidth=0.0,
                    label=f"{stage} error"
                )
                self.bands[stage] = band

                vis_band = self.band_visible.get(stage, False)
                vis_line = self.stage_visible.get(stage, True)
                band.set_visible(vis_band and vis_line)

        leg = ax.legend(loc="best", frameon=True)
        self._legend = leg

        # ЧЁТКАЯ ПРИВЯЗКА legend item → stage/type
        handles = leg.legend_handles
        texts = leg.get_texts()

        for h, t in zip(handles, texts):
            label = t.get_text()

            t.set_picker(True)
            h.set_picker(True)

            if label.endswith(" error"):
                stage = label[:-6]
                kind = "band"
            else:
                stage = label
                kind = "line"

            h._stage = stage
            h._kind = kind
            t._stage = stage
            t._kind = kind

            self._legend_items[label] = (h, t)

            # корректный alpha
            if kind == "line":
                on = self.stage_visible.get(stage, True)
            else:
                on = self.band_visible.get(stage, False)

            alpha = 1.0 if on else 0.3
            h.set_alpha(alpha)
            t.set_alpha(alpha)

        if norm1d:
            ax.set_ylim(0.0, 1.5)

        if self.chk_auto_ylim.isChecked():
            ax.autoscale(enable=True, axis="y")
        else:
            ymin = float(self.spin_ymin.value())
            ymax = float(self.spin_ymax.value())
            if ymax <= ymin:
                ymax = ymin + 1e-6
            ax.set_ylim(ymin, ymax)
        self.plot.draw()

    def _on_pick_legend(self, event):
        if self._legend is None:
            return

        artist = event.artist

        stage = getattr(artist, "_stage", None)
        kind = getattr(artist, "_kind", None)

        if stage is None or kind is None:
            return

        if kind == "line":
            new_vis = not self.stage_visible.get(stage, True)
            self.stage_visible[stage] = new_vis
            if stage in self.lines:
                self.lines[stage].set_visible(new_vis)

            if stage in self.bands:
                band_should = self.band_visible.get(stage, False) and new_vis
                self.bands[stage].set_visible(band_should)


        else:

            new_vis = not self.band_visible.get(stage, False)

            self.band_visible[stage] = new_vis

            line_vis = self.stage_visible.get(stage, True)

            if stage in self.bands:
                self.bands[stage].set_visible(new_vis and line_vis)

        # обновляем alpha
        for label, (h, t) in self._legend_items.items():
            if label.endswith(" error"):
                st = label[:-6]
                on = self.band_visible.get(st, False)
            else:
                st = label
                on = self.stage_visible.get(st, True)

            alpha = 1.0 if on else 0.3
            h.set_alpha(alpha)
            t.set_alpha(alpha)

        self.plot.draw_idle()

    # ---------- export ----------
    def export_csv(self):
        if not self.lines:
            QMessageBox.information(self, "Nothing to export", "No curves to export.")
            return

        series = []
        for name, line in self.lines.items():
            if not line.get_visible():
                continue
            xdata = np.asarray(line.get_xdata(), dtype=np.float64)
            ydata = np.asarray(line.get_ydata(), dtype=np.float64)
            series.append((name, xdata, ydata))

        if not series:
            QMessageBox.information(self, "Nothing to export", "All curves are hidden.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "diffraction_curves.csv", "CSV files (*.csv)")
        if not path:
            return

        dref = series[0][1]
        n = len(dref)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["d"] + [f"I_{name}" for (name, _, _) in series])
            for i in range(n):
                row = [float(dref[i])]
                for (_, _, yd) in series:
                    row.append(float(yd[i]) if i < len(yd) else "")
                w.writerow(row)

        QMessageBox.information(self, "Saved", f"CSV saved:\n{path}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
