#!/usr/bin/env python3
"""
Build standardized raw binary sky-map products from Shanghai Astronomical Observatory
HEALPix HDF5 sky maps.

Canonical convention used by this script:
  - input maps:  NSIDE=4096, RING order, ecliptic coordinate system
  - output bins: raw little-endian float32 arrays, no header, linear brightness B
  - pixel area is NOT premultiplied into the bin; the CUDA forward operator applies
    pix_area = 4*pi/Npix at runtime.

Typical products:
  1) native bins for 1--10 MHz maps
  2) a derived 30 MHz map fitted from 1--10 MHz maps in log-frequency space
  3) NSIDE=512 degraded maps for validation/debugging
  4) NSIDE=16384 nearest-parent stress maps for scalability tests
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import healpy as hp
import numpy as np


@dataclass
class SkyMeta:
    product: str
    source: str
    dataset: str
    frequency_mhz: float | None
    source_frequencies_mhz: list[float] | None
    nside: int
    npix: int
    order: str
    coordinate: str
    dtype: str
    format: str
    pixel_area_policy: str
    generation_method: str
    usage: str
    note: str


def parse_freq_file(items: Iterable[str]) -> List[Tuple[float, Path]]:
    """Parse CLI items like '1:/path/1MHz.hdf5'."""
    out: list[tuple[float, Path]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Input item must be FREQ_MHZ:PATH, got {item!r}")
        freq_s, path_s = item.split(":", 1)
        out.append((float(freq_s), Path(path_s)))
    out.sort(key=lambda x: x[0])
    return out


def freq_tag(freq_mhz: float) -> str:
    """Return file tag used by CUDA loader: 1 -> 1M, 10 -> 10M, 30 -> 30M."""
    if abs(freq_mhz - round(freq_mhz)) < 1e-9:
        return f"{int(round(freq_mhz))}M"
    return (f"{freq_mhz:g}M").replace(".", "p")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def inspect_hdf5_dataset(path: Path, dataset: str) -> tuple[int, int, np.dtype]:
    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset {dataset!r} not found in {path}. Keys={list(f.keys())}")
        d = f[dataset]
        if d.ndim != 1:
            raise ValueError(f"Dataset {dataset!r} in {path} must be 1-D, got shape={d.shape}")
        npix = int(d.shape[0])
        nside = int(hp.npix2nside(npix))
        return nside, npix, d.dtype


def save_native_bin(
    h5_path: Path,
    dataset: str,
    out_dir: Path,
    out_name: str,
    frequency_mhz: float,
    coordinate: str,
    order: str,
    chunk_pix: int,
) -> Path:
    """Save HDF5 dataset as raw little-endian float32 bin without premultiplying area."""
    ensure_dir(out_dir)
    out_bin = out_dir / out_name

    nside, npix, dtype = inspect_hdf5_dataset(h5_path, dataset)
    print(f"[native] {frequency_mhz:g} MHz: {h5_path}::{dataset}")
    print(f"         nside={nside}, npix={npix}, dtype={dtype}, order={order}")

    mm = np.memmap(out_bin, dtype="<f4", mode="w+", shape=(npix,))
    with h5py.File(h5_path, "r") as f:
        d = f[dataset]
        for start in range(0, npix, chunk_pix):
            end = min(start + chunk_pix, npix)
            arr = np.asarray(d[start:end], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            mm[start:end] = arr
            print(f"         wrote {end}/{npix} pixels", flush=True)
    mm.flush()
    del mm

    expected = npix * 4
    actual = out_bin.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Size mismatch for {out_bin}: got {actual}, expected {expected}")

    meta = SkyMeta(
        product=out_bin.name,
        source=str(h5_path),
        dataset=dataset,
        frequency_mhz=frequency_mhz,
        source_frequencies_mhz=None,
        nside=nside,
        npix=npix,
        order=order.upper(),
        coordinate=coordinate,
        dtype="float32 little-endian",
        format="raw binary, no header",
        pixel_area_policy="not premultiplied; CUDA applies pix_area=4*pi/Npix at runtime",
        generation_method="direct HDF5 dataset export with NaN/Inf cleanup",
        usage="scientific workload if source HDF5 is original simulation",
        note="linear sky brightness B; do not save log10(B); visualization may use log10 only",
    )
    write_json(out_bin.with_suffix(out_bin.suffix + ".json"), asdict(meta))
    return out_bin


def fit_derived_frequency_map(
    inputs: list[tuple[float, Path]],
    dataset: str,
    out_dir: Path,
    target_freq_mhz: float,
    coordinate: str,
    order: str,
    degree: int,
    chunk_pix: int,
    clip_beta: tuple[float, float] | None,
    save_fit_maps: bool,
) -> Path:
    """
    Fit log(B) as a function of log(nu) per pixel and extrapolate/interpolate
    to target_freq_mhz. Default degree=1 corresponds to an effective spectral index.
    """
    if degree < 1 or degree > 2:
        raise ValueError("Only degree=1 or degree=2 is supported in this script.")

    nsides = []
    npixes = []
    for freq, path in inputs:
        nside, npix, _ = inspect_hdf5_dataset(path, dataset)
        nsides.append(nside)
        npixes.append(npix)
    if len(set(nsides)) != 1 or len(set(npixes)) != 1:
        raise ValueError(f"All input maps must have same NSIDE/Npix, got nsides={nsides}, npixes={npixes}")

    nside = nsides[0]
    npix = npixes[0]
    freqs = np.asarray([f for f, _ in inputs], dtype=np.float64)
    x = np.log(freqs)
    xt = math.log(target_freq_mhz)

    # Design matrix columns: [1, x] or [1, x, x^2]
    if degree == 1:
        X = np.stack([np.ones_like(x), x], axis=1)
        xt_vec = np.asarray([1.0, xt], dtype=np.float64)
    else:
        X = np.stack([np.ones_like(x), x, x * x], axis=1)
        xt_vec = np.asarray([1.0, xt, xt * xt], dtype=np.float64)
    P = np.linalg.pinv(X)  # shape: coeff_count x n_freq

    ensure_dir(out_dir)
    tag = freq_tag(target_freq_mhz)
    out_bin = out_dir / f"B_{tag}.bin"
    out_map = np.memmap(out_bin, dtype="<f4", mode="w+", shape=(npix,))

    beta_bin = out_dir / f"effective_beta_{freq_tag(freqs[0])}_{freq_tag(freqs[-1])}_to_{tag}.bin"
    rmse_bin = out_dir / f"fit_log_rmse_{freq_tag(freqs[0])}_{freq_tag(freqs[-1])}_to_{tag}.bin"
    beta_map = np.memmap(beta_bin, dtype="<f4", mode="w+", shape=(npix,)) if save_fit_maps else None
    rmse_map = np.memmap(rmse_bin, dtype="<f4", mode="w+", shape=(npix,)) if save_fit_maps else None

    print(f"[derive] target={target_freq_mhz:g} MHz from freqs={freqs.tolist()}, degree={degree}")
    print(f"         nside={nside}, npix={npix}, out={out_bin}")

    handles = [h5py.File(path, "r") for _, path in inputs]
    try:
        datasets = [h[dataset] for h in handles]
        eps = 1e-30
        for start in range(0, npix, chunk_pix):
            end = min(start + chunk_pix, npix)
            # Y: n_freq x chunk
            stack = []
            for d in datasets:
                arr = np.asarray(d[start:end], dtype=np.float64)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                stack.append(np.log(np.maximum(arr, eps)))
            Y = np.stack(stack, axis=0)

            coeff = P @ Y  # coeff_count x chunk
            log_target = xt_vec @ coeff
            target = np.exp(log_target)
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            target = np.maximum(target, 0.0)
            out_map[start:end] = target.astype(np.float32)

            # For degree=1, beta is coeff[1]. For degree=2, effective beta at target x is d logB/d lognu.
            if save_fit_maps:
                if degree == 1:
                    beta = coeff[1]
                else:
                    beta = coeff[1] + 2.0 * coeff[2] * xt
                if clip_beta is not None:
                    beta_to_store = np.clip(beta, clip_beta[0], clip_beta[1])
                else:
                    beta_to_store = beta
                beta_map[start:end] = beta_to_store.astype(np.float32)

                pred = X @ coeff
                rmse = np.sqrt(np.mean((pred - Y) ** 2, axis=0))
                rmse_map[start:end] = rmse.astype(np.float32)

            print(f"         derived {end}/{npix} pixels", flush=True)
    finally:
        for h in handles:
            h.close()

    out_map.flush()
    del out_map
    if beta_map is not None:
        beta_map.flush()
        del beta_map
    if rmse_map is not None:
        rmse_map.flush()
        del rmse_map

    expected = npix * 4
    actual = out_bin.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Size mismatch for {out_bin}: got {actual}, expected {expected}")

    meta = SkyMeta(
        product=out_bin.name,
        source="; ".join([f"{f:g}MHz:{p}" for f, p in inputs]),
        dataset=dataset,
        frequency_mhz=target_freq_mhz,
        source_frequencies_mhz=[float(f) for f in freqs],
        nside=nside,
        npix=npix,
        order=order.upper(),
        coordinate=coordinate,
        dtype="float32 little-endian",
        format="raw binary, no header",
        pixel_area_policy="not premultiplied; CUDA applies pix_area=4*pi/Npix at runtime",
        generation_method=(
            f"per-pixel log-log polynomial fit of degree {degree}: "
            "log(B_nu)=a0+a1*log(nu)[+a2*log(nu)^2], then evaluated at target frequency"
        ),
        usage="derived frequency-scaled workload; suitable for controlled experiments, not an independent SAO simulation",
        note=(
            "Uses existing same-coordinate same-order SAO maps. If absorption produces strong curvature, "
            "degree=1 is an effective spectral-index extrapolation; degree=2 should be treated as sensitivity analysis."
        ),
    )
    write_json(out_bin.with_suffix(out_bin.suffix + ".json"), asdict(meta))

    if save_fit_maps:
        fit_meta = {
            "source_frequencies_mhz": [float(f) for f in freqs],
            "target_frequency_mhz": target_freq_mhz,
            "degree": degree,
            "beta_file": beta_bin.name,
            "rmse_file": rmse_bin.name,
            "beta_definition": "degree=1: slope dlogB/dlognu; degree=2: derivative at target log-frequency",
            "rmse_definition": "RMSE in natural-log brightness over fitted frequencies",
        }
        write_json(out_dir / "fit_products.json", fit_meta)

    return out_bin


def degrade_ring_bin(
    in_bin: Path,
    nside_in: int,
    out_dir: Path,
    out_name: str,
    nside_out: int,
    frequency_mhz: float,
    coordinate: str,
    chunk_note_source: str,
) -> Path:
    ensure_dir(out_dir)
    m = np.fromfile(in_bin, dtype="<f4")
    expected = hp.nside2npix(nside_in)
    if m.size != expected:
        raise RuntimeError(f"Input size mismatch: {in_bin}: got {m.size}, expected {expected}")

    print(f"[degrade] {in_bin} nside {nside_in} -> {nside_out}")
    out = hp.ud_grade(
        m,
        nside_out=nside_out,
        order_in="RING",
        order_out="RING",
        power=0,
        dtype=np.float32,
    )
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("<f4", copy=False)
    out_bin = out_dir / out_name
    out.tofile(out_bin)

    meta = SkyMeta(
        product=out_bin.name,
        source=str(in_bin),
        dataset="raw bin",
        frequency_mhz=frequency_mhz,
        source_frequencies_mhz=None,
        nside=nside_out,
        npix=int(out.size),
        order="RING",
        coordinate=coordinate,
        dtype="float32 little-endian",
        format="raw binary, no header",
        pixel_area_policy="not premultiplied; CUDA applies pix_area=4*pi/Npix at runtime",
        generation_method="healpy.ud_grade with power=0; intensity-preserving degradation by averaging",
        usage="small-scale validation/debugging workload",
        note=chunk_note_source,
    )
    write_json(out_bin.with_suffix(out_bin.suffix + ".json"), asdict(meta))
    return out_bin


def upsample_ring_nearest_parent_chunked(
    in_bin: Path,
    nside_in: int,
    out_dir: Path,
    out_name: str,
    nside_out: int,
    frequency_mhz: float,
    coordinate: str,
    chunk_pix: int,
    source_note: str,
) -> Path:
    ensure_dir(out_dir)
    m_in = np.fromfile(in_bin, dtype="<f4")
    expected_in = hp.nside2npix(nside_in)
    if m_in.size != expected_in:
        raise RuntimeError(f"Input size mismatch: {in_bin}: got {m_in.size}, expected {expected_in}")

    npix_out = hp.nside2npix(nside_out)
    out_bin = out_dir / out_name
    print(f"[upsample-nearest] {in_bin} nside {nside_in} -> {nside_out}")
    print(f"                   output={out_bin}, size={npix_out*4/1024**3:.2f} GiB")

    out = np.memmap(out_bin, dtype="<f4", mode="w+", shape=(npix_out,))
    for start in range(0, npix_out, chunk_pix):
        end = min(start + chunk_pix, npix_out)
        pix = np.arange(start, end, dtype=np.int64)
        theta, phi = hp.pix2ang(nside_out, pix, nest=False)
        parent = hp.ang2pix(nside_in, theta, phi, nest=False)
        out[start:end] = m_in[parent]
        print(f"                   wrote {end}/{npix_out} pixels", flush=True)
    out.flush()
    del out

    actual = out_bin.stat().st_size
    expected = npix_out * 4
    if actual != expected:
        raise RuntimeError(f"Size mismatch for {out_bin}: got {actual}, expected {expected}")

    meta = SkyMeta(
        product=out_bin.name,
        source=str(in_bin),
        dataset="raw bin",
        frequency_mhz=frequency_mhz,
        source_frequencies_mhz=None,
        nside=nside_out,
        npix=int(npix_out),
        order="RING",
        coordinate=coordinate,
        dtype="float32 little-endian",
        format="raw binary, no header",
        pixel_area_policy="not premultiplied; CUDA applies pix_area=4*pi/Npix at runtime",
        generation_method="chunked nearest-parent upsampling in RING order; B_child=B_parent",
        usage="controlled scalability stress-test workload only",
        note=(
            source_note + " No new physical small-scale structure is introduced; "
            "total sky integral is preserved under pixel-area-weighted visibility integration."
        ),
    )
    write_json(out_bin.with_suffix(out_bin.suffix + ".json"), asdict(meta))
    return out_bin


def integral_check(bin_path: Path, nside: int) -> float:
    m = np.fromfile(bin_path, dtype="<f4").astype(np.float64)
    expected = hp.nside2npix(nside)
    if m.size != expected:
        raise RuntimeError(f"Integral check size mismatch: {bin_path}: got {m.size}, expected {expected}")
    return float(np.sum(m * (4.0 * math.pi / expected)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare SAO HEALPix sky-map binary products for CUDA pipeline.")
    ap.add_argument("--inputs", nargs="+", required=True, help="List of FREQ_MHZ:PATH items, e.g. 1:1MHz.hdf5 10:10MHz.hdf5")
    ap.add_argument("--dataset", default="skymap", help="HDF5 dataset name, default: skymap")
    ap.add_argument("--out-root", default="sky_sao_processed", help="Output root directory")
    ap.add_argument("--order", default="RING", choices=["RING"], help="Canonical output order. This script assumes RING.")
    ap.add_argument("--coordinate", default="ecliptic", help="Coordinate system label for metadata")
    ap.add_argument("--chunk-pix", type=int, default=4_000_000, help="Chunk size for HDF5 export and fitting")
    ap.add_argument("--make-native-bins", action="store_true", help="Export each input HDF5 map to B_xM.bin")
    ap.add_argument("--derive-target-freq", type=float, default=30.0, help="Target frequency for derived map, default 30 MHz")
    ap.add_argument("--derive-degree", type=int, default=1, choices=[1, 2], help="Log-log fit degree. Default 1 = effective spectral index")
    ap.add_argument("--skip-derived", action="store_true", help="Skip derived target-frequency map")
    ap.add_argument("--save-fit-maps", action="store_true", help="Save effective beta and log-RMSE maps for derived frequency fit")
    ap.add_argument("--clip-beta", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Only affects stored beta map, not target map")
    ap.add_argument("--make-nside512", action="store_true", help="Create NSIDE=512 degraded maps for selected products")
    ap.add_argument("--make-nside16384", action="store_true", help="Create NSIDE=16384 nearest-parent stress maps for selected products")
    ap.add_argument("--resample-freqs", nargs="*", type=float, default=None, help="Frequencies to resample. Default: target freq only if derived; otherwise last input freq")
    ap.add_argument("--upsample-chunk-pix", type=int, default=8_000_000, help="Chunk size for NSIDE=16384 upsampling")
    ap.add_argument("--integral-check", action="store_true", help="Print pixel-area-weighted sky integrals for generated/resampled bins")
    args = ap.parse_args()

    inputs = parse_freq_file(args.inputs)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    product_bins: dict[float, tuple[Path, int, str]] = {}

    # Validate all input maps.
    for f, p in inputs:
        nside, npix, dtype = inspect_hdf5_dataset(p, args.dataset)
        print(f"[inspect] {f:g} MHz -> {p}, dataset={args.dataset}, nside={nside}, npix={npix}, dtype={dtype}")

    if args.make_native_bins:
        for freq, path in inputs:
            tag = freq_tag(freq)
            out_dir = out_root / f"{tag}_n4096_ring_{args.coordinate}"
            out_bin = save_native_bin(
                h5_path=path,
                dataset=args.dataset,
                out_dir=out_dir,
                out_name=f"B_{tag}.bin",
                frequency_mhz=freq,
                coordinate=args.coordinate,
                order=args.order,
                chunk_pix=args.chunk_pix,
            )
            nside, _, _ = inspect_hdf5_dataset(path, args.dataset)
            product_bins[float(freq)] = (out_bin, nside, "native SAO simulation")

    derived_bin: Path | None = None
    if not args.skip_derived:
        tag = freq_tag(args.derive_target_freq)
        out_dir = out_root / f"{tag}_n4096_derived_from_{freq_tag(inputs[0][0])}_to_{freq_tag(inputs[-1][0])}_ring_{args.coordinate}"
        derived_bin = fit_derived_frequency_map(
            inputs=inputs,
            dataset=args.dataset,
            out_dir=out_dir,
            target_freq_mhz=args.derive_target_freq,
            coordinate=args.coordinate,
            order=args.order,
            degree=args.derive_degree,
            chunk_pix=args.chunk_pix,
            clip_beta=tuple(args.clip_beta) if args.clip_beta is not None else None,
            save_fit_maps=args.save_fit_maps,
        )
        nside, _, _ = inspect_hdf5_dataset(inputs[0][1], args.dataset)
        product_bins[float(args.derive_target_freq)] = (derived_bin, nside, f"derived degree-{args.derive_degree} log-log fit")

    # If native bins were not requested but resampling is requested, create required native bins on demand.
    resample_freqs = args.resample_freqs
    if resample_freqs is None:
        resample_freqs = [float(args.derive_target_freq)] if not args.skip_derived else [float(inputs[-1][0])]

    if args.make_nside512 or args.make_nside16384:
        for rf in resample_freqs:
            if rf not in product_bins:
                # Try to export this frequency from input list.
                match = [(f, p) for f, p in inputs if abs(f - rf) < 1e-9]
                if not match:
                    raise ValueError(f"Cannot resample {rf:g} MHz: product not generated and no matching input HDF5.")
                freq, path = match[0]
                tag = freq_tag(freq)
                out_dir = out_root / f"{tag}_n4096_ring_{args.coordinate}"
                out_bin = save_native_bin(
                    h5_path=path,
                    dataset=args.dataset,
                    out_dir=out_dir,
                    out_name=f"B_{tag}.bin",
                    frequency_mhz=freq,
                    coordinate=args.coordinate,
                    order=args.order,
                    chunk_pix=args.chunk_pix,
                )
                nside, _, _ = inspect_hdf5_dataset(path, args.dataset)
                product_bins[float(freq)] = (out_bin, nside, "native SAO simulation")

            src_bin, src_nside, src_note = product_bins[float(rf)]
            tag = freq_tag(rf)
            if src_nside != 4096:
                print(f"[warn] Source nside for {rf:g} MHz is {src_nside}, not 4096.")
            if args.make_nside512:
                out_dir = out_root / f"{tag}_n512_from{src_nside}_ring_{args.coordinate}"
                out_bin = degrade_ring_bin(
                    in_bin=src_bin,
                    nside_in=src_nside,
                    out_dir=out_dir,
                    out_name=f"B_{tag}.bin",
                    nside_out=512,
                    frequency_mhz=rf,
                    coordinate=args.coordinate,
                    chunk_note_source=src_note,
                )
                if args.integral_check:
                    print(f"[integral] {src_bin} nside={src_nside}: {integral_check(src_bin, src_nside):.8e}")
                    print(f"[integral] {out_bin} nside=512: {integral_check(out_bin, 512):.8e}")

            if args.make_nside16384:
                out_dir = out_root / f"{tag}_n16384_from{src_nside}_ring_{args.coordinate}_stress"
                out_bin = upsample_ring_nearest_parent_chunked(
                    in_bin=src_bin,
                    nside_in=src_nside,
                    out_dir=out_dir,
                    out_name=f"B_{tag}.bin",
                    nside_out=16384,
                    frequency_mhz=rf,
                    coordinate=args.coordinate,
                    chunk_pix=args.upsample_chunk_pix,
                    source_note=src_note,
                )
                if args.integral_check:
                    print(f"[integral] {src_bin} nside={src_nside}: {integral_check(src_bin, src_nside):.8e}")
                    print(f"[integral] {out_bin} nside=16384: {integral_check(out_bin, 16384):.8e}")

    print("\nAll requested products are complete.")


if __name__ == "__main__":
    main()
