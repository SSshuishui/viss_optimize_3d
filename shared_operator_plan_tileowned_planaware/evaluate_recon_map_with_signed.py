#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate and visualize HEALPix full-sky reconstruction results.

Typical use:
python evaluate_recon_map.py \
  --origin sky_sao/10M_n4096_ring_ecliptic/B_10M.bin \
  --recon out10M/C_day1.bin \
  --nside 4096 \
  --order ring \
  --freq-label "10 MHz" \
  --out-dir eval_10M_n4096 \
  --band-deg 30 \
  --calib-region band \
  --plot-nside 1024

Notes:
1. The input origin/recon files must have the same HEALPix ordering.
2. The default evaluation is in log10 domain.
3. The reconstruction is linearly calibrated in log10 domain before reporting
   the calibrated metrics.
4. For large nside=4096 maps, plotting at plot_nside=1024 is usually enough
   for paper-quality visual inspection and faster output generation.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MetricResult:
    region: str
    variant: str
    n: int
    a: float
    b: float
    pearson: float
    nrmse: float
    rmse: float
    mae: float
    cosine: float
    ref_min: float
    ref_max: float


@dataclass
class SignedLinearMetricResult:
    region: str
    variant: str
    n: int
    a: float
    b: float
    pearson: float
    nrmse: float
    rel_l2: float
    rmse: float
    mae: float
    cosine: float
    ref_min: float
    ref_max: float
    recon_min: float
    recon_max: float
    nonpositive_recon: int
    nonpositive_recon_ratio: float


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--origin", required=True, help="Original sky map raw float32 bin")
    parser.add_argument("--recon", required=True, help="Reconstructed sky map raw float32 bin")
    parser.add_argument("--nside", type=int, required=True)
    parser.add_argument("--order", choices=["ring", "nest"], default="ring")
    parser.add_argument("--freq-label", default="10 MHz")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--band-deg", type=float, default=30.0)
    parser.add_argument(
        "--calib-region",
        choices=["full", "band"],
        default="band",
        help="Region used to fit log_origin ~= a * log_recon + b",
    )

    parser.add_argument(
        "--signed-linear-calib-region",
        choices=["full", "band", "none"],
        default="band",
        help=(
            "Region used to fit origin ~= a * recon + b for signed linear-domain metrics. "
            "Use none to report only uncalibrated signed linear-domain metrics."
        ),
    )

    parser.add_argument(
        "--chunk-pix",
        type=int,
        default=5_000_000,
        help="Chunk size for streaming metric computation",
    )

    parser.add_argument(
        "--plot-nside",
        type=int,
        default=1024,
        help="Downsampled nside used only for visualization. Set to 4096 for full-resolution plotting.",
    )

    parser.add_argument(
        "--origin-percentile",
        nargs=2,
        type=float,
        default=[1.0, 99.0],
        help="Percentiles of original log map used as shared visualization range",
    )

    parser.add_argument(
        "--log-eps",
        type=float,
        default=1e-30,
        help="Small positive value used before log10",
    )

    parser.add_argument(
        "--recon-transform",
        choices=["none", "abs"],
        default="none",
        help="Use abs(recon) before log if reconstructed map contains signed values",
    )

    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also save PDF versions of figures",
    )

    return parser.parse_args()


def expected_npix(nside: int) -> int:
    return hp.nside2npix(nside)


def check_file_size(path: str, npix: int):
    expected_bytes = npix * 4
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"File size mismatch for {path}: "
            f"expected {expected_bytes} bytes, got {actual_bytes} bytes"
        )


def open_raw_map(path: str, nside: int) -> np.memmap:
    npix = expected_npix(nside)
    check_file_size(path, npix)
    return np.memmap(path, dtype="<f4", mode="r", shape=(npix,))


def transform_recon(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "abs":
        return np.abs(x)
    return x


def safe_log10(x: np.ndarray, eps: float) -> np.ndarray:
    return np.log10(np.maximum(x, eps))


def region_mask_for_chunk(
    nside: int,
    order: str,
    start: int,
    end: int,
    region: str,
    band_deg: float,
) -> np.ndarray:
    n = end - start

    if region == "full":
        return np.ones(n, dtype=bool)

    nest = order == "nest"
    pix = np.arange(start, end, dtype=np.int64)
    theta, _ = hp.pix2ang(nside, pix, nest=nest)

    # HEALPix theta is colatitude. latitude = pi/2 - theta.
    lat = np.pi / 2.0 - theta
    return np.abs(lat) <= np.deg2rad(band_deg)


def fit_linear_log_stream(
    origin: np.memmap,
    recon: np.memmap,
    nside: int,
    order: str,
    region: str,
    band_deg: float,
    chunk_pix: int,
    eps: float,
    recon_mode: str,
):
    npix = len(origin)

    n_total = 0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0

    nonpositive_origin = 0
    nonpositive_recon = 0

    for start in range(0, npix, chunk_pix):
        end = min(start + chunk_pix, npix)

        y0 = np.asarray(origin[start:end], dtype=np.float64)
        x0 = np.asarray(recon[start:end], dtype=np.float64)
        x0 = transform_recon(x0, recon_mode)

        nonpositive_origin += int(np.sum(y0 <= 0))
        nonpositive_recon += int(np.sum(x0 <= 0))

        valid = np.isfinite(y0) & np.isfinite(x0) & (y0 > 0) & (x0 > 0)
        reg = region_mask_for_chunk(nside, order, start, end, region, band_deg)
        mask = valid & reg

        if not np.any(mask):
            continue

        y = safe_log10(y0[mask], eps)
        x = safe_log10(x0[mask], eps)

        n = x.size
        n_total += n
        sx += float(np.sum(x))
        sy += float(np.sum(y))
        sxx += float(np.sum(x * x))
        sxy += float(np.sum(x * y))

        print(f"[fit] processed {end}/{npix}", flush=True)

    denom = sxx - sx * sx / max(n_total, 1)
    if abs(denom) < 1e-30:
        raise RuntimeError("Linear calibration failed: denominator is too small.")

    a = (sxy - sx * sy / n_total) / denom
    b = sy / n_total - a * sx / n_total

    info = {
        "calib_region": region,
        "n_calib": n_total,
        "a": a,
        "b": b,
        "nonpositive_origin": nonpositive_origin,
        "nonpositive_recon": nonpositive_recon,
    }

    return a, b, info


def compute_metrics_log_stream(
    origin: np.memmap,
    recon: np.memmap,
    nside: int,
    order: str,
    region: str,
    band_deg: float,
    chunk_pix: int,
    eps: float,
    recon_mode: str,
    a: float,
    b: float,
    variant: str,
) -> MetricResult:
    npix = len(origin)

    n_total = 0

    sx = 0.0
    sy = 0.0
    sxx = 0.0
    syy = 0.0
    sxy = 0.0

    sd2 = 0.0
    sad = 0.0

    ref_min = float("inf")
    ref_max = -float("inf")

    for start in range(0, npix, chunk_pix):
        end = min(start + chunk_pix, npix)

        y0 = np.asarray(origin[start:end], dtype=np.float64)
        x0 = np.asarray(recon[start:end], dtype=np.float64)
        x0 = transform_recon(x0, recon_mode)

        valid = np.isfinite(y0) & np.isfinite(x0) & (y0 > 0) & (x0 > 0)
        reg = region_mask_for_chunk(nside, order, start, end, region, band_deg)
        mask = valid & reg

        if not np.any(mask):
            continue

        y = safe_log10(y0[mask], eps)
        x_raw = safe_log10(x0[mask], eps)
        x = a * x_raw + b

        diff = x - y

        n = x.size
        n_total += n

        sx += float(np.sum(x))
        sy += float(np.sum(y))
        sxx += float(np.sum(x * x))
        syy += float(np.sum(y * y))
        sxy += float(np.sum(x * y))

        sd2 += float(np.sum(diff * diff))
        sad += float(np.sum(np.abs(diff)))

        ref_min = min(ref_min, float(np.min(y)))
        ref_max = max(ref_max, float(np.max(y)))

        print(f"[metric:{variant}:{region}] processed {end}/{npix}", flush=True)

    if n_total <= 1:
        raise RuntimeError(f"No valid pixels for region={region}")

    cov_xy = sxy - sx * sy / n_total
    var_x = sxx - sx * sx / n_total
    var_y = syy - sy * sy / n_total

    pearson = cov_xy / math.sqrt(max(var_x * var_y, 1e-300))
    cosine = sxy / math.sqrt(max(sxx * syy, 1e-300))

    rmse = math.sqrt(sd2 / n_total)
    mae = sad / n_total
    nrmse = rmse / max(ref_max - ref_min, 1e-30)

    return MetricResult(
        region=region,
        variant=variant,
        n=n_total,
        a=a,
        b=b,
        pearson=pearson,
        nrmse=nrmse,
        rmse=rmse,
        mae=mae,
        cosine=cosine,
        ref_min=ref_min,
        ref_max=ref_max,
    )



def fit_linear_signed_stream(
    origin: np.memmap,
    recon: np.memmap,
    nside: int,
    order: str,
    region: str,
    band_deg: float,
    chunk_pix: int,
):
    """Fit origin ~= a * recon + b in the original signed linear domain.

    Unlike the log-domain calibration, this function keeps the sign of the
    reconstructed map and only requires finite pixels. This is useful because
    raw adjoint reconstructions can contain negative sidelobe-dominated values
    before deconvolution/restoration.
    """
    npix = len(origin)

    n_total = 0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0

    nonfinite_origin = 0
    nonfinite_recon = 0
    nonpositive_recon = 0

    for start in range(0, npix, chunk_pix):
        end = min(start + chunk_pix, npix)

        y0 = np.asarray(origin[start:end], dtype=np.float64)
        x0 = np.asarray(recon[start:end], dtype=np.float64)

        nonfinite_origin += int(np.sum(~np.isfinite(y0)))
        nonfinite_recon += int(np.sum(~np.isfinite(x0)))
        nonpositive_recon += int(np.sum(np.isfinite(x0) & (x0 <= 0)))

        valid = np.isfinite(y0) & np.isfinite(x0)
        reg = region_mask_for_chunk(nside, order, start, end, region, band_deg)
        mask = valid & reg

        if not np.any(mask):
            continue

        y = y0[mask]
        x = x0[mask]

        n = x.size
        n_total += n
        sx += float(np.sum(x))
        sy += float(np.sum(y))
        sxx += float(np.sum(x * x))
        sxy += float(np.sum(x * y))

        print(f"[fit:signed-linear] processed {end}/{npix}", flush=True)

    if n_total <= 1:
        raise RuntimeError(f"No valid pixels for signed linear calibration region={region}")

    denom = sxx - sx * sx / n_total
    if abs(denom) < 1e-300:
        raise RuntimeError("Signed linear calibration failed: denominator is too small.")

    a = (sxy - sx * sy / n_total) / denom
    b = sy / n_total - a * sx / n_total

    info = {
        "calib_region": region,
        "n_calib": n_total,
        "a": a,
        "b": b,
        "nonfinite_origin": nonfinite_origin,
        "nonfinite_recon": nonfinite_recon,
        "nonpositive_recon": nonpositive_recon,
        "nonpositive_recon_ratio": nonpositive_recon / max(npix, 1),
    }

    return a, b, info


def compute_metrics_signed_linear_stream(
    origin: np.memmap,
    recon: np.memmap,
    nside: int,
    order: str,
    region: str,
    band_deg: float,
    chunk_pix: int,
    a: float,
    b: float,
    variant: str,
) -> SignedLinearMetricResult:
    """Compute signed linear-domain metrics over all finite pixels.

    The reconstructed map is not clipped and not converted to abs values.
    Calibration, if supplied, is x = a * recon + b. This metric therefore
    complements log-domain metrics that can only include positive pixels.
    """
    npix = len(origin)

    n_total = 0

    sx = 0.0
    sy = 0.0
    sxx = 0.0
    syy = 0.0
    sxy = 0.0

    sd2 = 0.0
    sad = 0.0

    ref_min = float("inf")
    ref_max = -float("inf")
    recon_min = float("inf")
    recon_max = -float("inf")

    nonpositive_recon = 0

    for start in range(0, npix, chunk_pix):
        end = min(start + chunk_pix, npix)

        y0 = np.asarray(origin[start:end], dtype=np.float64)
        x_raw = np.asarray(recon[start:end], dtype=np.float64)

        valid = np.isfinite(y0) & np.isfinite(x_raw)
        reg = region_mask_for_chunk(nside, order, start, end, region, band_deg)
        mask = valid & reg

        if not np.any(mask):
            continue

        y = y0[mask]
        x_unscaled = x_raw[mask]
        x = a * x_unscaled + b
        diff = x - y

        n = x.size
        n_total += n

        sx += float(np.sum(x))
        sy += float(np.sum(y))
        sxx += float(np.sum(x * x))
        syy += float(np.sum(y * y))
        sxy += float(np.sum(x * y))

        sd2 += float(np.sum(diff * diff))
        sad += float(np.sum(np.abs(diff)))

        ref_min = min(ref_min, float(np.min(y)))
        ref_max = max(ref_max, float(np.max(y)))
        recon_min = min(recon_min, float(np.min(x)))
        recon_max = max(recon_max, float(np.max(x)))
        nonpositive_recon += int(np.sum(x_unscaled <= 0))

        print(f"[metric:signed-linear:{variant}:{region}] processed {end}/{npix}", flush=True)

    if n_total <= 1:
        raise RuntimeError(f"No valid pixels for signed linear metrics region={region}")

    cov_xy = sxy - sx * sy / n_total
    var_x = sxx - sx * sx / n_total
    var_y = syy - sy * sy / n_total

    pearson = cov_xy / math.sqrt(max(var_x * var_y, 1e-300))
    cosine = sxy / math.sqrt(max(sxx * syy, 1e-300))

    rmse = math.sqrt(sd2 / n_total)
    mae = sad / n_total
    nrmse = rmse / max(ref_max - ref_min, 1e-300)
    rel_l2 = math.sqrt(sd2 / max(syy, 1e-300))

    return SignedLinearMetricResult(
        region=region,
        variant=variant,
        n=n_total,
        a=a,
        b=b,
        pearson=pearson,
        nrmse=nrmse,
        rel_l2=rel_l2,
        rmse=rmse,
        mae=mae,
        cosine=cosine,
        ref_min=ref_min,
        ref_max=ref_max,
        recon_min=recon_min,
        recon_max=recon_max,
        nonpositive_recon=nonpositive_recon,
        nonpositive_recon_ratio=nonpositive_recon / max(n_total, 1),
    )


def write_signed_linear_metrics_csv(path: str, rows):
    fieldnames = [
        "region",
        "variant",
        "n",
        "a",
        "b",
        "pearson",
        "nrmse",
        "rel_l2",
        "rmse",
        "mae",
        "cosine",
        "ref_min",
        "ref_max",
        "recon_min",
        "recon_max",
        "nonpositive_recon",
        "nonpositive_recon_ratio",
    ]

    with open(path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow({
                "region": r.region,
                "variant": r.variant,
                "n": r.n,
                "a": r.a,
                "b": r.b,
                "pearson": r.pearson,
                "nrmse": r.nrmse,
                "rel_l2": r.rel_l2,
                "rmse": r.rmse,
                "mae": r.mae,
                "cosine": r.cosine,
                "ref_min": r.ref_min,
                "ref_max": r.ref_max,
                "recon_min": r.recon_min,
                "recon_max": r.recon_max,
                "nonpositive_recon": r.nonpositive_recon,
                "nonpositive_recon_ratio": r.nonpositive_recon_ratio,
            })


def print_signed_linear_latex_table(rows):
    print("\nSigned linear-domain LaTeX table rows:")
    print(r"\begin{table}[t]")
    print(r"\caption{Signed linear-domain image-fidelity metrics for the raw 3D reconstruction after linear calibration.}")
    print(r"\label{tab:image_fidelity_signed_linear}")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"Region & Pearson & NRMSE & Rel. L2 & RMSE & Cosine \\")
    print(r"\midrule")

    for r in rows:
        if r.variant != "linear_calibrated":
            continue

        region_name = "Full sky" if r.region == "full" else r"Supported band"
        print(
            f"{region_name} & "
            f"{r.pearson:.4f} & "
            f"{r.nrmse:.4e} & "
            f"{r.rel_l2:.4e} & "
            f"{r.rmse:.4e} & "
            f"{r.cosine:.4f} \\\\" 
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def write_metrics_csv(path: str, rows):
    fieldnames = [
        "region",
        "variant",
        "n",
        "a",
        "b",
        "pearson",
        "nrmse",
        "rmse",
        "mae",
        "cosine",
        "ref_min",
        "ref_max",
    ]

    with open(path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow({
                "region": r.region,
                "variant": r.variant,
                "n": r.n,
                "a": r.a,
                "b": r.b,
                "pearson": r.pearson,
                "nrmse": r.nrmse,
                "rmse": r.rmse,
                "mae": r.mae,
                "cosine": r.cosine,
                "ref_min": r.ref_min,
                "ref_max": r.ref_max,
            })


def print_latex_table(rows):
    print("\nLaTeX table rows:")
    print(r"\begin{table}[t]")
    print(r"\caption{Image-fidelity metrics for the raw 3D reconstruction in log scale after linear calibration.}")
    print(r"\label{tab:image_fidelity}")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Region & Pearson & NRMSE & RMSE & Cosine \\")
    print(r"\midrule")

    for r in rows:
        if r.variant != "calibrated":
            continue

        region_name = "Full sky" if r.region == "full" else r"Supported band"
        print(
            f"{region_name} & "
            f"{r.pearson:.4f} & "
            f"{r.nrmse:.4e} & "
            f"{r.rmse:.4e} & "
            f"{r.cosine:.4f} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def load_plot_map(path: str, nside: int, order: str, plot_nside: int) -> np.ndarray:
    mm = open_raw_map(path, nside)
    m = np.array(mm, dtype=np.float32, copy=True)

    if plot_nside != nside:
        order_name = "NESTED" if order == "nest" else "RING"
        m = hp.ud_grade(
            m,
            nside_out=plot_nside,
            order_in=order_name,
            order_out=order_name,
            power=0,
        ).astype(np.float32)

    return m


def save_figure(base_path: str, save_pdf: bool):
    plt.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
    if save_pdf:
        plt.savefig(base_path + ".pdf", bbox_inches="tight")
    plt.close()


def plot_maps(
    origin_path: str,
    recon_path: str,
    nside: int,
    order: str,
    plot_nside: int,
    out_dir: str,
    freq_label: str,
    band_deg: float,
    eps: float,
    recon_mode: str,
    a: float,
    b: float,
    origin_percentile,
    save_pdf: bool,
):
    print("\n[plot] Loading maps for visualization...")
    origin = load_plot_map(origin_path, nside, order, plot_nside)
    recon = load_plot_map(recon_path, nside, order, plot_nside)
    recon = transform_recon(recon.astype(np.float64), recon_mode).astype(np.float32)

    nest = order == "nest"

    valid_origin = origin > 0
    valid_recon = recon > 0

    origin_log = safe_log10(origin.astype(np.float64), eps)
    recon_log_raw = safe_log10(recon.astype(np.float64), eps)
    recon_log_cal = a * recon_log_raw + b

    valid = valid_origin & valid_recon & np.isfinite(origin_log) & np.isfinite(recon_log_cal)

    vmin, vmax = np.percentile(origin_log[valid], origin_percentile)
    print(f"[plot] Shared visualization range from origin percentiles: vmin={vmin}, vmax={vmax}")

    # 1) Input vs raw reconstruction with shared color range
    fig = plt.figure(figsize=(15, 5))
    hp.mollview(
        origin_log,
        nest=nest,
        fig=fig.number,
        sub=(1, 2, 1),
        min=vmin,
        max=vmax,
        title=f"(a) Input sky map, {freq_label}",
        cmap=plt.cm.viridis,
    )
    hp.mollview(
        recon_log_raw,
        nest=nest,
        fig=fig.number,
        sub=(1, 2, 2),
        min=vmin,
        max=vmax,
        title=f"(b) Raw 3D reconstruction",
        cmap=plt.cm.viridis,
    )
    save_figure(os.path.join(out_dir, "fig_input_vs_raw_recon_shared_range"), save_pdf)

    # 2) Input vs calibrated reconstruction with shared color range
    fig = plt.figure(figsize=(15, 5))
    hp.mollview(
        origin_log,
        nest=nest,
        fig=fig.number,
        sub=(1, 2, 1),
        min=vmin,
        max=vmax,
        title=f"(a) Input sky map, {freq_label}",
        cmap=plt.cm.viridis,
    )
    hp.mollview(
        recon_log_cal,
        nest=nest,
        fig=fig.number,
        sub=(1, 2, 2),
        min=vmin,
        max=vmax,
        title=f"(b) Calibrated raw reconstruction",
        cmap=plt.cm.viridis,
    )
    save_figure(os.path.join(out_dir, "fig_input_vs_calibrated_recon_shared_range"), save_pdf)

    # 3) Residual map
    residual = recon_log_cal - origin_log
    residual[~valid] = hp.UNSEEN

    abs_lim = np.percentile(np.abs(residual[valid]), 99.0)
    abs_lim = max(abs_lim, 1e-6)

    hp.mollview(
        residual,
        nest=nest,
        min=-abs_lim,
        max=abs_lim,
        title=f"Residual map: calibrated reconstruction - input, {freq_label}",
        cmap=plt.cm.coolwarm,
    )
    save_figure(os.path.join(out_dir, "fig_residual_full_sky"), save_pdf)

    # 4) Band-limited residual map
    npix_plot = hp.nside2npix(plot_nside)
    pix = np.arange(npix_plot, dtype=np.int64)
    theta, _ = hp.pix2ang(plot_nside, pix, nest=nest)
    lat = np.pi / 2.0 - theta
    band = np.abs(lat) <= np.deg2rad(band_deg)

    residual_band = np.array(residual, copy=True)
    residual_band[~band] = hp.UNSEEN

    hp.mollview(
        residual_band,
        nest=nest,
        min=-abs_lim,
        max=abs_lim,
        title=f"Residual in supported band |lat| <= {band_deg:g} deg",
        cmap=plt.cm.coolwarm,
    )
    save_figure(os.path.join(out_dir, "fig_residual_supported_band"), save_pdf)

    # 5) Supported-band mask visualization
    mask_map = np.zeros(npix_plot, dtype=np.float32)
    mask_map[band] = 1.0

    hp.mollview(
        mask_map,
        nest=nest,
        min=0,
        max=1,
        title=f"Evaluation band mask: |lat| <= {band_deg:g} deg",
        cmap=plt.cm.viridis,
    )
    save_figure(os.path.join(out_dir, "fig_supported_band_mask"), save_pdf)

    # Save plotting configuration
    plot_meta = {
        "plot_nside": plot_nside,
        "shared_vmin": float(vmin),
        "shared_vmax": float(vmax),
        "origin_percentile": origin_percentile,
        "residual_abs_lim_99pct": float(abs_lim),
        "band_deg": band_deg,
        "calibration_a": float(a),
        "calibration_b": float(b),
        "domain": "log10",
    }

    with open(os.path.join(out_dir, "plot_config.json"), "w", encoding="utf-8") as fw:
        json.dump(plot_meta, fw, indent=2, ensure_ascii=False)

    print("[plot] Figures saved to:", out_dir)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    npix = expected_npix(args.nside)
    check_file_size(args.origin, npix)
    check_file_size(args.recon, npix)

    origin = open_raw_map(args.origin, args.nside)
    recon = open_raw_map(args.recon, args.nside)

    print("Input information")
    print("  origin:", args.origin)
    print("  recon :", args.recon)
    print("  nside :", args.nside)
    print("  npix  :", npix)
    print("  order :", args.order)
    print("  band  :", f"|lat| <= {args.band_deg} deg")
    print("  recon transform:", args.recon_transform)

    # Fit linear calibration in log10 domain
    a, b, calib_info = fit_linear_log_stream(
        origin=origin,
        recon=recon,
        nside=args.nside,
        order=args.order,
        region=args.calib_region,
        band_deg=args.band_deg,
        chunk_pix=args.chunk_pix,
        eps=args.log_eps,
        recon_mode=args.recon_transform,
    )

    print("\nLinear calibration in log10 domain:")
    print(f"  log_origin ~= a * log_recon + b")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  calibration region = {args.calib_region}")
    print(f"  n_calib = {calib_info['n_calib']}")
    print(f"  nonpositive_origin = {calib_info['nonpositive_origin']}")
    print(f"  nonpositive_recon  = {calib_info['nonpositive_recon']}")

    # Metrics before and after calibration
    rows = []

    for region in ["full", "band"]:
        rows.append(
            compute_metrics_log_stream(
                origin=origin,
                recon=recon,
                nside=args.nside,
                order=args.order,
                region=region,
                band_deg=args.band_deg,
                chunk_pix=args.chunk_pix,
                eps=args.log_eps,
                recon_mode=args.recon_transform,
                a=1.0,
                b=0.0,
                variant="raw",
            )
        )

        rows.append(
            compute_metrics_log_stream(
                origin=origin,
                recon=recon,
                nside=args.nside,
                order=args.order,
                region=region,
                band_deg=args.band_deg,
                chunk_pix=args.chunk_pix,
                eps=args.log_eps,
                recon_mode=args.recon_transform,
                a=a,
                b=b,
                variant="calibrated",
            )
        )

    csv_path = os.path.join(args.out_dir, "metrics_log10.csv")
    write_metrics_csv(csv_path, rows)

    # Signed linear-domain metrics. These keep negative reconstructed values
    # instead of discarding them before log10, so they complement the log-domain
    # metrics above.
    signed_rows = []

    if args.signed_linear_calib_region == "none":
        lin_a, lin_b = 1.0, 0.0
        signed_calib_info = {
            "calib_region": "none",
            "n_calib": 0,
            "a": lin_a,
            "b": lin_b,
        }
    else:
        lin_a, lin_b, signed_calib_info = fit_linear_signed_stream(
            origin=origin,
            recon=recon,
            nside=args.nside,
            order=args.order,
            region=args.signed_linear_calib_region,
            band_deg=args.band_deg,
            chunk_pix=args.chunk_pix,
        )

    print("\nLinear calibration in signed linear domain:")
    print("  origin ~= a * recon + b")
    print(f"  a = {lin_a}")
    print(f"  b = {lin_b}")
    print(f"  calibration region = {args.signed_linear_calib_region}")
    print(f"  n_calib = {signed_calib_info.get('n_calib', 0)}")

    for region in ["full", "band"]:
        signed_rows.append(
            compute_metrics_signed_linear_stream(
                origin=origin,
                recon=recon,
                nside=args.nside,
                order=args.order,
                region=region,
                band_deg=args.band_deg,
                chunk_pix=args.chunk_pix,
                a=1.0,
                b=0.0,
                variant="linear_raw",
            )
        )

        signed_rows.append(
            compute_metrics_signed_linear_stream(
                origin=origin,
                recon=recon,
                nside=args.nside,
                order=args.order,
                region=region,
                band_deg=args.band_deg,
                chunk_pix=args.chunk_pix,
                a=lin_a,
                b=lin_b,
                variant="linear_calibrated",
            )
        )

    signed_csv_path = os.path.join(args.out_dir, "metrics_signed_linear.csv")
    write_signed_linear_metrics_csv(signed_csv_path, signed_rows)

    summary = {
        "origin": args.origin,
        "recon": args.recon,
        "nside": args.nside,
        "order": args.order,
        "frequency_label": args.freq_label,
        "log_domain": {
            "calibration": calib_info,
            "metrics_csv": csv_path,
        },
        "signed_linear_domain": {
            "calibration": signed_calib_info,
            "metrics_csv": signed_csv_path,
        },
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fw:
        json.dump(summary, fw, indent=2, ensure_ascii=False)

    print("\nMetrics saved to:", csv_path)

    for r in rows:
        print(
            f"{r.variant:>10s} | {r.region:>4s} | "
            f"Pearson={r.pearson:.6f}, "
            f"NRMSE={r.nrmse:.6e}, "
            f"RMSE={r.rmse:.6e}, "
            f"MAE={r.mae:.6e}, "
            f"Cosine={r.cosine:.6f}, "
            f"N={r.n}"
        )

    print_latex_table(rows)

    print("\nSigned linear-domain metrics saved to:", signed_csv_path)
    for r in signed_rows:
        print(
            f"{r.variant:>18s} | {r.region:>4s} | "
            f"Pearson={r.pearson:.6f}, "
            f"NRMSE={r.nrmse:.6e}, "
            f"RelL2={r.rel_l2:.6e}, "
            f"RMSE={r.rmse:.6e}, "
            f"MAE={r.mae:.6e}, "
            f"Cosine={r.cosine:.6f}, "
            f"nonpositive_recon={r.nonpositive_recon} "
            f"({100.0 * r.nonpositive_recon_ratio:.2f}%), "
            f"N={r.n}"
        )

    print_signed_linear_latex_table(signed_rows)

    # Plot figures
    plot_maps(
        origin_path=args.origin,
        recon_path=args.recon,
        nside=args.nside,
        order=args.order,
        plot_nside=args.plot_nside,
        out_dir=args.out_dir,
        freq_label=args.freq_label,
        band_deg=args.band_deg,
        eps=args.log_eps,
        recon_mode=args.recon_transform,
        a=a,
        b=b,
        origin_percentile=args.origin_percentile,
        save_pdf=args.save_pdf,
    )


if __name__ == "__main__":
    main()