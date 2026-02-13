import torch
import numpy as np
import math
from scipy.signal import find_peaks


# ============================================================
# Q-wrapper (2D → d-space)
# ============================================================

class Qwrapper:

    def __init__(self,
                 theta_range,
                 L_range,
                 fixed_centers,
                 device="cuda"):

        if fixed_centers is None:
            raise ValueError("fixed_centers must be provided")

        self.device = device
        self.theta_range = theta_range
        self.L_range = L_range

        centers = torch.tensor(fixed_centers, dtype=torch.float32)
        self.centers = centers.to(device)

        edges = torch.zeros(len(centers) + 1, dtype=torch.float32)
        edges[1:-1] = (centers[:-1] + centers[1:]) * 0.5
        edges[0] = centers[0] - (centers[1] - centers[0]) * 0.5
        edges[-1] = centers[-1] + (centers[-1] - centers[-2]) * 0.5
        self.edges = edges.to(device)

    def tensor_to_d(self, batch_tensor):

        if batch_tensor.dim() != 4:
            raise ValueError("Expected tensor [B,1,H,W]")

        B, _, H, W = batch_tensor.shape
        batch_tensor = batch_tensor.to(self.device)

        theta_deg = torch.linspace(*self.theta_range, W, device=self.device)
        L_vals = torch.linspace(*self.L_range, H, device=self.device)

        theta_rad = torch.deg2rad(theta_deg)

        L_grid, theta_grid = torch.meshgrid(L_vals, theta_rad, indexing="ij")
        d_grid = L_grid / (2 * torch.sin(torch.abs(theta_grid) * 0.5))

        mask = d_grid <= 7.5

        results = []

        for b in range(B):

            I_mat = batch_tensor[b, 0]
            d_vals = d_grid[mask]
            I_vals = I_mat[mask]

            idx = torch.bucketize(d_vals, self.edges) - 1
            I_summed = torch.zeros(len(self.centers), device=self.device)
            I_summed.scatter_add_(0, idx.clamp(0, len(I_summed) - 1), I_vals)

            results.append({
                "d": self.centers.detach().cpu().numpy(),
                "I": I_summed.detach().cpu().numpy()
            })

        return results

# ============================================================
# Peak Detection
# ============================================================

def extract_peak_region(d, I, peak_idx, peaks, properties,
                        scale_factor=1.5,
                        default_window=15):

    try:
        peak_array_idx = np.where(peaks == peak_idx)[0][0]
    except IndexError:
        return d[peak_idx:peak_idx+1], I[peak_idx:peak_idx+1]

    if "widths" in properties:
        window = int(properties["widths"][peak_array_idx] * scale_factor)
    else:
        window = default_window

    start = max(peak_idx - window, 0)
    end = min(peak_idx + window, len(d))

    return d[start:end], I[start:end]


def find_peaks_for_batch(batch_DI,
                         height=0.05,
                         distance=10,
                         prominence=0.1,
                         width=5,
                         scale_factor=1.5,
                         default_window=15,
                         scale=False):

    batch_results = []

    for sample in batch_DI:

        d = sample["d"]
        I = sample["I"] / 4 if scale else sample["I"]

        peaks, properties = find_peaks(
            I,
            height=height,
            distance=distance,
            prominence=prominence,
            width=width
        )

        sample_peaks = []

        for peak_idx in peaks:

            d_window, I_window = extract_peak_region(
                d, I, peak_idx, peaks, properties,
                scale_factor, default_window
            )

            integral_intensity = float(np.sum(I_window))
            max_intensity = float(I[peak_idx])
            com = np.sum(d_window * I_window) / np.sum(I_window)

            sample_peaks.append({
                "d": float(d[peak_idx]),
                "d_com": float(com),
                "integral_intensity": integral_intensity,
                "max_intensity": max_intensity,
                "profile_d": d_window,
                "profile_I": I_window
            })

        batch_results.append(sample_peaks)

    return batch_results

# ============================================================
# Peak Shape Metrics (EMD)
# ============================================================

def normalize_profile(I):
    s = np.sum(I)
    if s <= 0:
        return None
    return I / s


def resample_profile(d, I, d_center, x_ref):

    x = (d - d_center) / d_center
    I_norm = normalize_profile(I)

    if I_norm is None:
        return None

    return np.interp(x_ref, x, I_norm, left=0.0, right=0.0)


def emd_1d(p, q, dx):

    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)

    return np.sum(np.abs(cdf_p - cdf_q)) * dx


def emd_shape_loss(peak1, peak2, x_ref, eps=1e-12):

    p1 = resample_profile(
        peak1["profile_d"],
        peak1["profile_I"],
        peak1["d"],
        x_ref
    )

    p2 = resample_profile(
        peak2["profile_d"],
        peak2["profile_I"],
        peak2["d"],
        x_ref
    )

    if p1 is None or p2 is None:
        return 0.0

    p1 = np.maximum(p1, 0)
    p2 = np.maximum(p2, 0)

    p1 /= (np.sum(p1) + eps)
    p2 /= (np.sum(p2) + eps)

    dx = x_ref[1] - x_ref[0]

    return emd_1d(p1, p2, dx)

# ============================================================
# Peak Comparison
# ============================================================

def compare_peak_sets(pred_peaks, true_peaks, tol=0.05):

    total_Iint = 0.0
    total_Imax = 0.0
    total_shape = 0.0

    if len(pred_peaks) == 0 or len(true_peaks) == 0:
        return total_Iint, total_Imax, total_shape

    x_ref = np.linspace(-0.03, 0.03, 64)

    for p1 in pred_peaks:

        d1 = p1["d_com"]

        p2 = min(true_peaks, key=lambda p: abs(p["d"] - d1))
        d2 = p2["d_com"]

        if abs(d1 - d2) > tol:
            continue

        # Integral intensity
        Iint1 = max(p1["integral_intensity"], 0)
        Iint2 = max(p2["integral_intensity"], 0)

        total_Iint += (math.log(Iint1 + 1) - math.log(Iint2 + 1)) ** 2

        # Max intensity
        Imax1 = max(p1["max_intensity"], 0)
        Imax2 = max(p2["max_intensity"], 0)

        total_Imax += (math.log(Imax1 + 1) - math.log(Imax2 + 1)) ** 2

        # Shape
        total_shape += emd_shape_loss(p1, p2, x_ref)

    return total_Iint, total_Imax, total_shape

# ============================================================
# Batch Aggregator
# ============================================================

def peak_matching_loss(batch_pred, batch_true, tol=0.05):

    batch_Iint = []
    batch_Imax = []
    batch_shape = []

    for pred_peaks, true_peaks in zip(batch_pred, batch_true):

        Iint, Imax, shape = compare_peak_sets(
            pred_peaks, true_peaks, tol
        )

        batch_Iint.append(Iint)
        batch_Imax.append(Imax)
        batch_shape.append(shape)

    return {
        "Integral Intensity": batch_Iint,
        "Peak Intensity": batch_Imax,
        "Shape": batch_shape
    }

# ============================================================
# High-level Metrics Calculator
# ============================================================

class DiffractionMetricsCalculator:

    def __init__(self,
                 fixed_centers_pred,
                 fixed_centers_true,
                 theta_range=(-170, 170),
                 L_range=(0.1, 10),
                 device="cuda"):

        self.device = device

        self.qw_pred = Qwrapper(theta_range, L_range,
                                fixed_centers_pred, device)

        self.qw_true = Qwrapper(theta_range, L_range,
                                fixed_centers_true, device)

    def __call__(self,
                 batch_pred_2d,
                 batch_true_2d,
                 peak_params_pred={},
                 peak_params_true={},
                 tol=0.05):

        pred_DI = self.qw_pred.tensor_to_d(batch_pred_2d)
        true_DI = self.qw_true.tensor_to_d(batch_true_2d)

        pred_peaks = find_peaks_for_batch(pred_DI, **peak_params_pred)
        true_peaks = find_peaks_for_batch(true_DI, **peak_params_true)

        return peak_matching_loss(pred_peaks, true_peaks, tol)