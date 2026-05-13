nvcc -o pix2ang_ring pix2ang_ring.cu -Xcompiler -fopenmp



# 1M  512
# 10M 4096
# 30M 16384

./pix2ang_ring \
  --nside 512 \
  --out_theta './freq_1m/theta_heal.txt' \
  --out_phi './freq_1m/phi_heal.txt'

./pix2ang_ring \
  --nside 4096 \
  --out_theta './freq_10m/theta_heal.txt' \
  --out_phi './freq_10m/phi_heal.txt'

# ./pix2ang_ring \
#   --out_dir=./freq_30m \
#   --nside=16384 \
#   --gpu=0