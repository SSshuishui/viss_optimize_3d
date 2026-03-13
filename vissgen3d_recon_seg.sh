# nvcc -O3 -lineinfo -Xcompiler -fopenmp -std=c++14 vissgen3d_recon_seg.cu -o vissgen3d_recon_seg

# blockage
./vissgen3d_recon_seg \
  --btag=10M --nside=4096 \
  --start_day=1 --end_day=1 \
  --in_dir=./earth_10Mhz --out_dir=./out_block/ \
  --gpus=0,1,2,3 --uvw_max=450000 --blockage=1

# noblockage
./vissgen3d_recon_seg \
  --btag=10M --nside=4096 \
  --start_day=1 --end_day=1 \
  --in_dir=./earth_10Mhz --out_dir=./out_noblock/ \
  --gpus=0,1,2,3 --uvw_max=450000 --blockage=0