
# 1. 使用nsys分析看整体
# nsys profile -o ./out10M/pipeline_multigpu \
#   --force-overwrite=true \
#   --sample=none \
#   --cpuctxsw=none \
#   --trace=cuda \
#   --cuda-memory-usage=false \
#   ./shared_operator_plan_tileowned_planaware \
#   --btag=10M \
#   --nside=4096 \
#   --day_start=1 \
#   --day_count=3 \
#   --segs=10 \
#   --sky_dir=../input_skymap/sky_sao/10M_n4096_ring_ecliptic/B_10M \
#   --out_dir=./out10M/ \
#   --gpus=1,3 \
#   --gen_gpu_index=0 \
#   --B_mode=bin \
#   --C_mode=bin \
#   --sky_order=ring \
#   --orbit_seed=42 \
#   --dcf_bin=dcf_mb_10M_days450_seed42.bin \
#   --viss_tile_pix=256



# 2. 用 NCU 分析热点 kernel（单GPU）
# mkdir -p ./out10M/ncu

# sudo ncu \
#   --force-overwrite \
#   --set full \
#   --kernel-name regex:.*recon_3d_direct_planaware_tileowned_real.* \
#   --launch-skip 5 \
#   --launch-count 1 \
#   --export ./out10M/ncu/recon_planaware_day1_seg1 \
#   ./shared_operator_plan_tileowned_planaware \
#     --btag=10M \
#     --nside=4096 \
#     --day_start=1 \
#     --day_count=1 \
#     --segs=10 \
#     --sky_dir=../input_skymap/sky_sao/10M_n4096_ring_ecliptic/B_10M \
#     --out_dir=./out10M/ \
#     --gpus=1,3 \
#     --gen_gpu_index=0 \
#     --B_mode=bin \
#     --C_mode=bin \
#     --sky_order=ring \
#     --orbit_seed=42 \
#     --dcf_bin=dcf_mb_10M_days450_seed42.bin \
#     --viss_tile_pix=256



mkdir -p ./out10M/viss_ncu

sudo ncu \
  --force-overwrite \
  --replay-mode kernel \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,launch__registers_per_thread,launch__shared_mem_per_block \
  --kernel-name regex:.*viss_partial_all_halfsym_tilecone.* \
  --launch-count 1 \
  --export ./out10M/viss_ncu/viss_keymetrics \
  ./shared_operator_plan_tileowned_planaware \
    --btag=10M \
    --nside=4096 \
    --day_start=1 \
    --day_count=1 \
    --segs=1 \
    --sky_dir=../input_skymap/sky_sao/10M_n4096_ring_ecliptic/B_10M \
    --out_dir=./out10M/ \
    --gpus=1 \
    --gen_gpu_index=0 \
    --B_mode=bin \
    --C_mode=bin \
    --sky_order=ring \
    --orbit_seed=42 \
    --dcf_bin=dcf_mb_10M_days450_seed42.bin \
    --viss_tile_pix=256