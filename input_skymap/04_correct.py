import numpy as np
import healpy as hp

def sky_integral(path, nside):
    B = np.fromfile(path, dtype=np.float32).astype(np.float64)
    pix_area = 4*np.pi / hp.nside2npix(nside)
    return np.sum(B * pix_area)

print("4096 :", sky_integral("sky_sao/10M_n4096_ring_ecliptic/B_10M.bin", 4096))
print("512  :", sky_integral("sky_sao/10M_n512_from4096_ring_ecliptic/B_10M.bin", 512))
print("16384:", sky_integral("sky_sao/10M_n16384_from4096_ring_ecliptic/B_10M.bin", 16384))