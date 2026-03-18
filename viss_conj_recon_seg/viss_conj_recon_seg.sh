# nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp viss_conj_recon_seg.cu -o viss_conj_recon_seg

./viss_conj_recon_seg \
  --btag=1M \
  --nside=512 \
  --day=1 \
  --dcf_days=1 \
  --in_dir=../earth_1Mhz \
  --sky_dir=../earth_1Mhz \
  --out_dir=../out3d_1M_seg/ \
  --gpus=0,1,2,3 \
  --uvw_max=4500000 \
  --write_viss=1