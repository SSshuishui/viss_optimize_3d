#!/usr/bin/env bash
set -euo pipefail

# build
# nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp dcf_mb_gen.cu -o dcf_mb_gen
# nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp main_3d_viss_recon.cu -o main_3d_viss_recon

# ------------------------------------------------------------
# step 1: generate stable dcf+mb once and save to bin
# ------------------------------------------------------------
./dcf_mb_gen \
  --btag=10M \
  --dcf_days=450 \
  --segs=10 \
  --bl_max=100000 \
  --gpus=2,3 \
  --gen_gpu_index=0 \
  --orbit_seed=42 \
  --out=dcf_mb_10M_days450_seed42.bin

# ------------------------------------------------------------
# step 2: load B + dcf/mb bin, then run viss + 3D recon
# ------------------------------------------------------------
./main_3d_viss_recon \
  --btag=10M \
  --nside=4096 \
  --day=1 \
  --day_start=1 \
  --day_count=1 \
  --segs=10 \
  --blockage=1 \
  --sky_dir=../earth_10Mhz \
  --out_dir=../out10M_3d_viss_recon_tile/ \
  --gpus=2,3 \
  --gen_gpu_index=0 \
  --B_mode=txt \
  --C_mode=bin \
  --orbit_seed=42 \
  --dcf_bin=dcf_mb_10M_days450_seed42.bin
