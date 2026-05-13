import os
import json
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def upsample_ring_nearest_chunked(
    in_bin: str,
    nside_in: int,
    out_dir: str,
    out_name: str,
    nside_out: int,
    frequency_mhz: float,
    coordinate: str = "ecliptic",
    chunk_pix: int = 8_000_000,
    plot_check: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    m_in = np.fromfile(in_bin, dtype="<f4")
    expected_in = hp.nside2npix(nside_in)

    if len(m_in) != expected_in:
        raise RuntimeError(f"Input size mismatch: got {len(m_in)}, expected {expected_in}")

    npix_out = hp.nside2npix(nside_out)
    out_bin = os.path.join(out_dir, out_name)

    print("\nInput")
    print("  bin:", in_bin)
    print("  nside:", nside_in)
    print("  npix:", len(m_in))
    print("  min/max/mean:", np.min(m_in), np.max(m_in), np.mean(m_in))

    print("\nOutput")
    print("  bin:", out_bin)
    print("  nside:", nside_out)
    print("  npix:", npix_out)
    print("  expected GB:", npix_out * 4 / 1024**3)
    print("  method: chunked nearest-parent resampling in RING order")

    out = np.memmap(out_bin, dtype="<f4", mode="w+", shape=(npix_out,))

    for start in range(0, npix_out, chunk_pix):
        end = min(start + chunk_pix, npix_out)
        pix_out = np.arange(start, end, dtype=np.int64)

        theta, phi = hp.pix2ang(nside_out, pix_out, nest=False)

        # 找到每个高分辨率像素中心在低分辨率图上的父像素。
        pix_parent = hp.ang2pix(nside_in, theta, phi, nest=False)

        out[start:end] = m_in[pix_parent].astype("<f4", copy=False)

        print(f"  written {end}/{npix_out} pixels ({100.0 * end / npix_out:.2f}%)", flush=True)

    out.flush()
    del out

    actual_bytes = os.path.getsize(out_bin)
    expected_bytes = npix_out * 4

    if actual_bytes != expected_bytes:
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
        "resampling": "chunked nearest-parent upsampling; no new physical small-scale structure",
        "usage": "controlled scalability stress-test workload only",
        "derived_from": "Shanghai Astronomical Observatory nside=4096 simulated map",
    }

    with open(out_bin + ".json", "w", encoding="utf-8") as fw:
        json.dump(meta, fw, indent=2, ensure_ascii=False)

    print("\nDone")
    print("  actual GB:", actual_bytes / 1024**3)
    print("  meta:", out_bin + ".json")

    if plot_check:
        # 16384 全图画图没必要全部高分辨率显示，可以降到 1024 看结构。
        m_big = np.memmap(out_bin, dtype="<f4", mode="r", shape=(npix_out,))
        m_show = hp.ud_grade(
            np.asarray(m_big),
            nside_out=1024,
            order_in="RING",
            order_out="RING",
            power=0,
            dtype=np.float32,
        )
        hp.mollview(
            np.log10(np.maximum(m_show, 1e-30)),
            nest=False,
            cmap=plt.cm.jet,
            title=f"{frequency_mhz:g} MHz, nside={nside_out} stress map shown at nside=1024",
        )
        hp.graticule()
        plt.show()


if __name__ == "__main__":
    upsample_ring_nearest_chunked(
        in_bin="sky_sao/10M_n4096_ring_ecliptic/B_10M.bin",
        nside_in=4096,
        out_dir="sky_sao/10M_n16384_from4096_ring_ecliptic",
        out_name="B_10M.bin",
        nside_out=16384,
        frequency_mhz=10.0,
        chunk_pix=8_000_000,
    )