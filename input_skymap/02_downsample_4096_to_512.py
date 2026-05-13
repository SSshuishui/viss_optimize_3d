import os
import json
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def downsample_ring_bin(
    in_bin: str,
    nside_in: int,
    out_dir: str,
    out_name: str,
    nside_out: int,
    frequency_mhz: float,
    coordinate: str = "ecliptic",
    plot_check: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    m = np.fromfile(in_bin, dtype="<f4")
    expected_in = hp.nside2npix(nside_in)

    if len(m) != expected_in:
        raise RuntimeError(f"Input size mismatch: got {len(m)}, expected {expected_in}")

    print("\nInput")
    print("  bin:", in_bin)
    print("  nside:", nside_in)
    print("  npix:", len(m))
    print("  min/max/mean:", np.min(m), np.max(m), np.mean(m))

    # 对亮温 / 面亮度图，降采样用 power=0，即子像素均值。
    out = hp.ud_grade(
        m,
        nside_out=nside_out,
        order_in="RING",
        order_out="RING",
        power=0,
        dtype=np.float32,
    )

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = out.astype("<f4", copy=False)

    out_bin = os.path.join(out_dir, out_name)
    out.tofile(out_bin)

    expected_out = hp.nside2npix(nside_out) * 4
    actual_bytes = os.path.getsize(out_bin)

    print("\nOutput")
    print("  bin:", out_bin)
    print("  nside:", nside_out)
    print("  npix:", len(out))
    print("  size MB:", actual_bytes / 1024**2)
    print("  min/max/mean:", np.min(out), np.max(out), np.mean(out))

    if actual_bytes != expected_out:
        raise RuntimeError("Output bin size mismatch.")

    meta = {
        "source_bin": in_bin,
        "frequency_mhz": frequency_mhz,
        "source_nside": nside_in,
        "nside": nside_out,
        "order": "RING",
        "coordinate": coordinate,
        "dtype": "float32 little-endian",
        "format": "raw binary, no header",
        "resampling": "healpy.ud_grade, power=0, intensity-preserving average",
        "derived_from": "Shanghai Astronomical Observatory nside=4096 simulated map",
    }

    with open(out_bin + ".json", "w", encoding="utf-8") as fw:
        json.dump(meta, fw, indent=2, ensure_ascii=False)

    if plot_check:
        hp.mollview(
            np.log10(np.maximum(out, 1e-30)),
            nest=False,
            cmap=plt.cm.jet,
            title=f"{frequency_mhz:g} MHz, nside={nside_out}, from 4096, RING",
        )
        hp.graticule()
        plt.show()


if __name__ == "__main__":
    downsample_ring_bin(
        in_bin="sky_sao/1M_n4096_ring_ecliptic/B_1M.bin",
        nside_in=4096,
        out_dir="sky_sao/1M_n512_from4096_ring_ecliptic",
        out_name="B_1M.bin",
        nside_out=512,
        frequency_mhz=1.0,
    )

    downsample_ring_bin(
        in_bin="sky_sao/10M_n4096_ring_ecliptic/B_10M.bin",
        nside_in=4096,
        out_dir="sky_sao/10M_n512_from4096_ring_ecliptic",
        out_name="B_10M.bin",
        nside_out=512,
        frequency_mhz=10.0,
    )