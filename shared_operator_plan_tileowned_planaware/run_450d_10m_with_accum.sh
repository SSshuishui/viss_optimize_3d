#!/usr/bin/env bash
set -euo pipefail

# 用法：在 shared_operator_plan_tileowned_planaware 目录下运行本脚本。
# 如果你的最终二进制仍叫 shared_operator_plan_stage2_balance，就保持下面命令不变；
# 如果已经改名为 shared_operator_plan_tileowned_planaware，请把 EXE 改掉。

EXE=./shared_operator_plan_tileowned_planaware

# OUT_DIR=./out10M
# mkdir -p "${OUT_DIR}"

# python3 ./run_planaware_450d_accumulate.py \
#   --out-dir "${OUT_DIR}" \
#   --btag 10M \
#   --nside 4096 \
#   --day-start 1 \
#   --day-count 450 \
#   --accum-file "${OUT_DIR}/C_accum_10M_days1_450.bin" \
#   --accum-mode atomic \
#   --delete-source \
#   --check-interval 5 \
#   --stable-seconds 3 \
#   --chunk-mb 256 \
#   --cmd \
#   "${EXE}" \
#     --btag=10M \
#     --nside=4096 \
#     --day_start=1 \
#     --day_count=450 \
#     --segs=10 \
#     --sky_dir=../input_skymap/sky_sao/10M_n4096_ring_ecliptic/B_10M \
#     --out_dir="${OUT_DIR}/" \
#     --gpus=1,3 \
#     --gen_gpu_index=0 \
#     --B_mode=bin \
#     --C_mode=bin \
#     --sky_order=ring \
#     --orbit_seed=42 \
#     --dcf_bin=dcf_mb_10M_days450_seed42.bin \
#     --viss_tile_pix=256


OUT_DIR=./out30M_n4096
mkdir -p "${OUT_DIR}"

python3 ./run_planaware_450d_accumulate.py \
  --out-dir "${OUT_DIR}" \
  --btag 30M \
  --nside 4096 \
  --day-start 1 \
  --day-count 450 \
  --accum-file "${OUT_DIR}/C_accum_30M_days1_450.bin" \
  --accum-mode atomic \
  --delete-source \
  --check-interval 5 \
  --stable-seconds 3 \
  --chunk-mb 256 \
  --cmd \
  "${EXE}" \
    --btag=30M \
    --nside=4096 \
    --day_start=1 \
    --day_count=450 \
    --segs=10 \
    --sky_dir=../input_skymap/sky_sao/30M_n4096_derived_from_1M_to_10M_ring_ecliptic/B_30M \
    --out_dir="${OUT_DIR}/" \
    --gpus=1,3 \
    --gen_gpu_index=0 \
    --B_mode=bin \
    --C_mode=bin \
    --sky_order=ring \
    --orbit_seed=42 \
    --dcf_bin=dcf_mb_30M_days450_seed42.bin \
    --viss_tile_pix=256


# OUT_DIR=./out1M
# mkdir -p "${OUT_DIR}"

# python3 ./run_planaware_450d_accumulate.py \
#   --out-dir "${OUT_DIR}" \
#   --btag 10M \
#   --nside 512 \
#   --day-start 1 \
#   --day-count 450 \
#   --accum-file "${OUT_DIR}/C_accum_1M_days1_450.bin" \
#   --accum-mode atomic \
#   --delete-source \
#   --check-interval 5 \
#   --stable-seconds 3 \
#   --chunk-mb 256 \
#   --cmd \
#   "${EXE}" \
#     --btag=1M \
#     --nside=512 \
#     --day_start=1 \
#     --day_count=450 \
#     --segs=10 \
#     --sky_dir=../input_skymap/sky_sao/1M_n512_from4096_ring_ecliptic/B_1M \
#     --out_dir="${OUT_DIR}/" \
#     --gpus=0,1 \
#     --gen_gpu_index=0 \
#     --B_mode=bin \
#     --C_mode=bin \
#     --sky_order=ring \
#     --orbit_seed=42 \
#     --dcf_bin=dcf_mb_1M_days450_seed42.bin \
#     --viss_tile_pix=256