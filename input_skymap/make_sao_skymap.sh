python make_sao_sky_products.py \
  --inputs \
    1:shao/1.0MHz_with_absorption.hdf5 \
    2:shao/2.0MHz_with_absorption_split.hdf5 \
    3:shao/3.0MHz_with_absorption_split.hdf5 \
    4:shao/4.0MHz_with_absorption_split.hdf5 \
    5:shao/5.0MHz_with_absorption_split.hdf5 \
    6:shao/6.0MHz_with_absorption_split.hdf5 \
    7:shao/7.0MHz_with_absorption_split.hdf5 \
    8:shao/8.0MHz_with_absorption_split.hdf5 \
    9:shao/9.0MHz_with_absorption_split.hdf5 \
    10:shao/10.0MHz_with_absorption_split.hdf5 \
  --dataset skymap \
  --out-root sky_sao \
  --coordinate ecliptic \
  --derive-target-freq 30 \
  --derive-degree 1 \
  --make-nside16384 \
  --resample-freqs 30 \
  --integral-check