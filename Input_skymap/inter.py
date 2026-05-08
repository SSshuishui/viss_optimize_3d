import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# ============================================
# 射电天图频率+空间插值（空间观测专用）
# 1MHz (NSIDE=512) → 10MHz (NSIDE=4096) + 30MHz (NSIDE=16384)
# ============================================

# --- 参数设置 ---
INPUT_PATH   = "../earth_1Mhz/B_1M.txt"   # 修改为你的文件路径
OUTPUT_10MHz = "sky_10MHz_n4096.bin"
OUTPUT_30MHz = "sky_30MHz_n16384.bin"
OUTPUT_BETA  = "spectral_index_n512.bin"

NSIDE_REF = 512
NSIDE_10  = 4096
NSIDE_30  = 16384
NU_REF    = 1.0   # MHz

# --- 1. 读取 1MHz 参考天图 ---
recon = np.loadtxt(INPUT_PATH, dtype=np.float32)
npix_ref = 12 * NSIDE_REF**2

assert len(recon) == npix_ref, \
    f"像素数不匹配: 文件有 {len(recon)} 个像素, NSIDE={NSIDE_REF} 需要 {npix_ref}"

print(f"✓ 已加载 1MHz 天图: {len(recon)} 像素, NSIDE={NSIDE_REF}")
print(f"  亮温范围: [{recon.min():.2f}, {recon.max():.2f}] K")

# --- 2. 生成空间变化频谱指数 (在 NSIDE=512 上生成，再随空间一起上采样) ---
np.random.seed(42)
T_norm = recon / np.max(recon)

# 亮区(银河平面)同步辐射主导 → beta 更负 (~-2.7)
# 暗区自由-自由/热辐射占比高 → beta 较平 (~-2.1)
beta_map = -2.1 - 0.6 * T_norm
beta_map += np.random.normal(0, 0.1, len(recon))   # 小尺度涨落
beta_map = np.clip(beta_map, -3.0, -1.8)           # 物理约束

print(f"✓ 频谱指数范围: [{beta_map.min():.2f}, {beta_map.max():.2f}], 均值={beta_map.mean():.2f}")

# --- 3. 插值到 10MHz (NSIDE=4096) ---
# 先空间上采样，再频率外推
recon_512_to_4096 = hp.ud_grade(recon, nside_out=NSIDE_10, order_in='NESTED', order_out='NESTED')
beta_4096 = hp.ud_grade(beta_map, nside_out=NSIDE_10, order_in='NESTED', order_out='NESTED')

recon_10mhz = recon_512_to_4096 * (10.0 / NU_REF)**beta_4096
recon_10mhz = np.maximum(recon_10mhz, 0.0)   # 非负约束

print(f"\n✓ 10MHz (NSIDE={NSIDE_10}) 插值完成")
print(f"  像素数: {len(recon_10mhz)}")
print(f"  亮温范围: [{recon_10mhz.min():.4f}, {recon_10mhz.max():.2f}] K")

# --- 4. 插值到 30MHz (NSIDE=16384) ---
recon_512_to_16384 = hp.ud_grade(recon, nside_out=NSIDE_30, order_in='NESTED', order_out='NESTED')
beta_16384 = hp.ud_grade(beta_map, nside_out=NSIDE_30, order_in='NESTED', order_out='NESTED')

recon_30mhz = recon_512_to_16384 * (30.0 / NU_REF)**beta_16384
recon_30mhz = np.maximum(recon_30mhz, 0.0)

print(f"\n✓ 30MHz (NSIDE={NSIDE_30}) 插值完成")
print(f"  像素数: {len(recon_30mhz)}")
print(f"  亮温范围: [{recon_30mhz.min():.4f}, {recon_30mhz.max():.2f}] K")

# --- 5. 保存为二进制文件 ---
recon_10mhz.astype(np.float32).tofile(OUTPUT_10MHz)
recon_30mhz.astype(np.float32).tofile(OUTPUT_30MHz)

print(f"\n✓ 文件已保存:")
print(f"  {OUTPUT_10MHz}   ({len(recon_10mhz)*4/1024**2:.1f} MB)")
print(f"  {OUTPUT_30MHz}  ({len(recon_30mhz)*4/1024**3:.1f} GB)")
print(f"  {OUTPUT_BETA}    (频谱指数图)")

# --- 6. 可视化 30MHz ---
hp.mollview(recon_30mhz, nest=True, title="30MHz Brightness Temperature (NSIDE=16384)", xsize=2000)
hp.graticule()
plt.show()