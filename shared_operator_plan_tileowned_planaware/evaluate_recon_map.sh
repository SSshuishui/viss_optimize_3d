

# 1M nside512
python evaluate_recon_map.py \
  --origin ../input_skymap/sky_sao/1M_n512_from4096_ring_ecliptic/B_1M.bin \
  --recon out1M_n512/C_accum_1M_days1_450.bin \
  --nside 512 \
  --order ring \
  --freq-label "1 MHz" \
  --out-dir eval_1M_n512 \
  --band-deg 30 \
  --calib-region band \
  --plot-nside 512 \
  --save-pdf

# 10M nside4096
python evaluate_recon_map.py \
  --origin ../input_skymap/sky_sao/10M_n4096_ring_ecliptic/B_10M.bin \
  --recon out10M_n4096/C_accum_10M_days1_450.bin \
  --nside 4096 \
  --order ring \
  --freq-label "10 MHz" \
  --out-dir eval_10M_n4096 \
  --band-deg 30 \
  --calib-region band \
  --plot-nside 1024 \
  --save-pdf


# 30M nside4096
python evaluate_recon_map.py \
  --origin ../input_skymap/sky_sao/30M_n4096_derived_from_1M_to_10M_ring_ecliptic/B_30M.bin \
  --recon out30M_n4096/C_accum_30M_days1_450.bin \
  --nside 4096 \
  --order ring \
  --freq-label "30 MHz" \
  --out-dir eval_30M_n4096 \
  --band-deg 30 \
  --calib-region band \
  --plot-nside 1024 \
  --save-pdf


# # 30M nside16384
# python evaluate_recon_map.py \
#   --origin ../input_skymap/sky_sao/10M_n4096_ring_ecliptic/B_10M.bin \
#   --recon out30M_n16384/C_accum_30M_days1_50.bin \
#   --nside 16384 \
#   --order ring \
#   --freq-label "30 MHz" \
#   --out-dir eval_30M_n16384 \
#   --band-deg 30 \
#   --calib-region band \
#   --plot-nside 1024 \
#   --save-pdf