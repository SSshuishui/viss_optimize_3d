nvcc -O3 --use_fast_math -lineinfo -Xcompiler -fopenmp -std=c++17 orbit_gen.cu -o orbit_gen

./orbit_gen --only=1M --start=1 --end=1 --device=0 --seed=42

./orbit_gen --only=10M --start=1 --end=1 --device=0 --seed=42
