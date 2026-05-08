import os
import json
import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def save_hdf5_skymap_as_ring_bin(
    h5_path: str,
    dataset: str,
    out_dir: str,
    out_name: str,
    frequency_mhz: float,
    coordinate: str = "ecliptic",
    plot_check: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        print("HDF5 keys:", list(f.keys()))
        m = f[dataset][:]

    m = np.asarray(m)
    npix = m.size
    nside = hp.npix2nside(npix)

    print("\nInput")
    print("  file:", h5_path)
    print("  dataset:", dataset)
    print("  nside:", nside)
    print("  npix:", npix)
    print("  dtype:", m.dtype)
    print("  min:", np.nanmin(m))
    print("  max:", np.nanmax(m))
    print("  mean:", np.nanmean(m))

    # 不保存 log10，不归一化。只清理异常值。
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    m = m.astype("<f4", copy=False)

    out_bin = os.path.join(out_dir, out_name)
    m.tofile(out_bin)

    expected_bytes = hp.nside2npix(nside) * 4
    actual_bytes = os.path.getsize(out_bin)

    print("\nOutput")
    print("  bin:", out_bin)
    print("  expected bytes:", expected_bytes)
    print("  actual bytes:", actual_bytes)

    if expected_bytes != actual_bytes:
        raise RuntimeError("Output bin size mismatch.")

    meta = {
        "source_hdf5": h5_path,
        "dataset": dataset,
        "frequency_mhz": frequency_mhz,
        "nside": int(nside),
        "npix": int(npix),
        "order": "RING",
        "coordinate": coordinate,
        "dtype": "float32 little-endian",
        "format": "raw binary, no header",
        "unit_note": "linear sky brightness; log10 is only for visualization",
        "derived_from": "original Shanghai Astronomical Observatory simulated map",
    }

    meta_path = out_bin + ".json"
    with open(meta_path, "w", encoding="utf-8") as fw:
        json.dump(meta, fw, indent=2, ensure_ascii=False)

    print("  meta:", meta_path)

    if plot_check:
        hp.mollview(
            np.log10(np.maximum(m, 1e-30)),
            nest=False,
            cmap=plt.cm.jet,
            title=f"{frequency_mhz:g} MHz, nside={nside}, RING, {coordinate}",
        )
        hp.graticule()
        plt.show()


if __name__ == "__main__":
    save_hdf5_skymap_as_ring_bin(
        h5_path="1.0MHz_with_absorption.hdf5",
        dataset="skymap",
        out_dir="sky_sao/1M_n4096_ring_ecliptic",
        out_name="B_1M.bin",
        frequency_mhz=1.0,
    )

    save_hdf5_skymap_as_ring_bin(
        h5_path="10.0MHz_with_absorption_split.hdf5",
        dataset="skymap",
        out_dir="sky_sao/10M_n4096_ring_ecliptic",
        out_name="B_10M.bin",
        frequency_mhz=10.0,
    )