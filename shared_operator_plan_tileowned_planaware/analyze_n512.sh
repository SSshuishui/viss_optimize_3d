# nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp shared_operator_plan_tileowned_planaware.cu -o shared_operator_plan_tileowned_planaware

# 1. 使用 nsys 分析整体。这里保留 CUDA trace，关闭 osrt / memory 细项，避免事件过密导致导出失败。
nsys profile -o ./out1M/pipeline_multigpu \
  --force-overwrite=true \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda \
  --cuda-memory-usage=false \
  ./shared_operator_plan_tileowned_planaware \
  --btag=1M \
  --nside=512 \
  --day_start=1 \
  --day_count=5 \
  --segs=10 \
  --sky_dir=../input_skymap/sky_sao/1M_n512_from4096_ring_ecliptic/B_1M \
  --out_dir=./out1M/ \
  --gpus=0,1 \
  --gen_gpu_index=0 \
  --B_mode=bin \
  --C_mode=bin \
  --sky_order=ring \
  --orbit_seed=42 \
  --dcf_bin=dcf_mb_1M_days450_seed42.bin \
  --viss_tile_pix=256


# 导出
nsys stats pipeline_multigpu.nsys-rep \
 --report cuda_gpu_kern_sum \
 --format csv \
 -o kernel_stats.csv


# 2. 用 NCU 分析热点 kernel（单GPU）
# # 只分析 viss kernel
# ncu --set full \
#  --kernel-name "*viss_partial*" \
#  -o viss_analysis \
#  ./main_3d_viss_recon --gpus=0 ...

# # 只分析 recon kernel
# ncu --set full \
#  --kernel-name "*recon_3d*" \
#  -o recon_analysis \
#  ./main_3d_viss_recon --gpus=0 ...
