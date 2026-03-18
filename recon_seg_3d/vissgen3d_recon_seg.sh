# nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp vissgen3d_recon_seg.cu -o vissgen3d_recon_seg

./vissgen3d_recon_seg \
  --btag=10M \
  --nside=4096 \
  --start_day=1 \
  --end_day=1 \
  --dcf_start_day=1 \
  --dcf_end_day=450 \
  --in_dir=/data/zhaox/earth_10Mhz \
  --out_dir=./out10M_dcf/ \
  --gpus=0,1,2,3 \
  --uvw_max=450000 \
  --blockage=1