import torch
import numpy as np
from scipy.signal import find_peaks
import math


# -------------------------------
# 1) Qwrapper
# -------------------------------
class Qwrapper:
    def __init__(self, theta_range, L_range, fixed_centers, device="cuda"):
        self.theta_range = theta_range
        self.L_range = L_range
        self.device = device

        if fixed_centers is None:
            raise ValueError("Нужно передать фиксированные центры каналов d.")

        centers = torch.tensor(fixed_centers, dtype=torch.float32)
        self.centers = centers.to(device)

        # Строим edges
        edges = torch.zeros(len(centers) + 1, dtype=torch.float32)
        edges[1:-1] = (centers[:-1] + centers[1:]) * 0.5
        edges[0] = centers[0] - (centers[1] - centers[0]) * 0.5
        edges[-1] = centers[-1] + (centers[-1] - centers[-2]) * 0.5
        self.edges = edges.to(device)

    def tensor_to_d(self, batch_tensor):
        if batch_tensor.dim() != 4:
            raise ValueError("Ожидается тензор [B,1,H,W]")

        B, _, H, W = batch_tensor.shape
        batch_tensor = batch_tensor.to(self.device)

        theta_deg = torch.linspace(self.theta_range[0], self.theta_range[1], W, device=self.device)
        L_vals = torch.linspace(self.L_range[0], self.L_range[1], H, device=self.device)

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


# -------------------------------
# 2) Поиск и интегрирование пиков
# -------------------------------
def extract_peak_region(d, I, peak_idx, peaks, properties,
                        scale_factor=1.5, default_window=15):

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
            integral_intensity = np.sum(I_window)
            sample_peaks.append({
                "d": d[peak_idx],
                "integral_intensity": float(integral_intensity)
            })

        batch_results.append(sample_peaks)

    return batch_results


# -------------------------------
# 3) Peak matching loss
# -------------------------------
def peak_matching_loss(batch_pred, batch_true, tol=0.05):
    total_loss = 0.0

    B = len(batch_pred)
    for b in range(B):
        pred_peaks = batch_pred[b]
        true_peaks = batch_true[b]

        if len(pred_peaks) == 0 or len(true_peaks) == 0:
            continue

        for peak1 in pred_peaks:
            d1 = peak1['d']
            I1 = peak1['integral_intensity']

            closest_peak = min(true_peaks, key=lambda p: abs(p['d'] - d1))
            d2 = closest_peak['d']
            I2 = closest_peak['integral_intensity']

            if abs(d1 - d2) <= tol:
                # 🔒 Безопасный лог
                I1_safe = max(I1, 0)
                I2_safe = max(I2, 0)
                total_loss += (math.log(I1_safe + 1) - math.log(I2_safe + 1))**2

    return torch.tensor(total_loss, dtype=torch.float32)




# ============================================================
# 4) Единый ООП-класс для всего пайплайна
# ============================================================
class DiffractionPipeline:
    def __init__(self,
                 fixed_centers_pred,
                 fixed_centers_true,
                 theta_range=(-170, 170),
                 L_range=(0.1, 10),
                 device="cuda"):

        self.device = device

        self.qw_pred = Qwrapper(theta_range, L_range, fixed_centers_pred, device)
        self.qw_true = Qwrapper(theta_range, L_range, fixed_centers_true, device)

    def __call__(self,
                 batch_pred_2d,
                 batch_true_2d,
                 peak_params_pred={},
                 peak_params_true={},
                 tol=0.05):

        # 1D проекции
        pred_DI = self.qw_pred.tensor_to_d(batch_pred_2d)
        true_DI = self.qw_true.tensor_to_d(batch_true_2d)

        # Пики
        pred_peaks = find_peaks_for_batch(pred_DI, **peak_params_pred)
        true_peaks = find_peaks_for_batch(true_DI, **peak_params_true)

        # Лосс
        return peak_matching_loss(pred_peaks, true_peaks, tol)
