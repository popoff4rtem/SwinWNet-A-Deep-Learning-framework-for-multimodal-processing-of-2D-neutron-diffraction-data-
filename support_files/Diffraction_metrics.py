import torch
import numpy as np
from scipy.signal import find_peaks
import math
from Peak_loss import Qwrapper, extract_peak_region

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

def normalize_profile(I):
    s = np.sum(I)
    if s <= 0:
        return None
    return I / s

def resample_profile(d, I, d_center, x_ref):
    """
    d        : массив d
    I        : интенсивность
    d_center : центр пика
    x_ref    : общая относительная сетка
    """
    x = (d - d_center) / d_center
    I_norm = normalize_profile(I)
    if I_norm is None:
        return None

    return np.interp(x_ref, x, I_norm, left=0.0, right=0.0)

def emd_1d(p, q, dx):
    """
    p, q : нормированные 1D распределения (sum = 1)
    dx   : шаг сетки
    """
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return np.sum(np.abs(cdf_p - cdf_q)) * dx

def emd_shape_loss(peak1, peak2,
                   x_ref,
                   eps=1e-12):
    """
    peak1, peak2 — словари пиков
    peak["d"], peak["profile_d"], peak["profile_I"]
    """

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

    # защита от численного мусора
    p1 = np.maximum(p1, 0)
    p2 = np.maximum(p2, 0)

    p1 /= (np.sum(p1) + eps)
    p2 /= (np.sum(p2) + eps)

    dx = x_ref[1] - x_ref[0]

    return emd_1d(p1, p2, dx)


# -------------------------------
# 3) Peak matching loss
# -------------------------------
def peak_matching_loss(batch_pred, batch_true, tol=0.05):
    # total_loss_Iint = 0.0
    # total_loss_Imax = 0.0
    # total_loss_shape = 0.0

    total_loss_Iint = []
    total_loss_Imax = []
    total_loss_shape = []

    for pred_peaks, true_peaks in zip(batch_pred, batch_true):
        total_loss_Iint_bach = 0.0
        total_loss_Imax_bach = 0.0
        total_loss_shape_bach = 0.0

        if len(pred_peaks) == 0 or len(true_peaks) == 0:
            continue

        for p1 in pred_peaks:
            d1 = p1["d_com"]
            Iint1 = p1["integral_intensity"]
            Imax1 = p1["max_intensity"]

            p2 = min(true_peaks, key=lambda p: abs(p["d"] - d1))
            d2 = p2["d_com"]

            if abs(d1 - d2) > tol:
                continue

            Iint2 = p2["integral_intensity"]
            Imax2 = p2["max_intensity"]

            # 🔒 Безопасный лог
            Iint1_safe = max(Iint1, 0)
            Iint2_safe = max(Iint2, 0)
            Imax1_safe = max(Imax1, 0)
            Imax2_safe = max(Imax2, 0)

            # 1) Интегральная интенсивность
            loss_Iint = (math.log(Iint1_safe + 1) - math.log(Iint2_safe + 1))**2

            # 2) Максимальная интенсивность
            loss_Imax = (math.log(Imax1_safe + 1) - math.log(Imax2_safe + 1))**2

            # 2) Форма пиков
            x_ref = np.linspace(-0.03, 0.03, 64)
            loss_shape = emd_shape_loss(p1, p2, x_ref)

            # total_loss_Iint += loss_Iint
            # total_loss_Imax += loss_Imax
            # total_loss_shape += loss_shape

    # return {'Integral Intensity': total_loss_Iint/len(batch_pred),
    #         'Peak Intensity': total_loss_Imax/len(batch_pred),
    #         'Shape': total_loss_shape/len(batch_pred),
    #         }


            total_loss_Iint_bach += loss_Iint
            total_loss_Imax_bach += loss_Imax
            total_loss_shape_bach += loss_shape

        total_loss_Iint.append(total_loss_Iint_bach)
        total_loss_Imax.append(total_loss_Imax_bach)
        total_loss_shape.append(total_loss_shape_bach)


    return {'Integral Intensity': total_loss_Iint,
            'Peak Intensity': total_loss_Imax,
            'Shape': total_loss_shape,
            }

class DiffractionMetricsCalculator:
    """
    Класс для вычисления всех метрик дифракции:
    1. Integral Intensity - интегральная интенсивность
    2. Peak Intensity - пиковая интенсивность
    3. Shape - форма пика (ширина, асимметрия)
    """
    
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