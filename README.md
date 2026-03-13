# viss_optimize_3d

传统三维天图成像方法

# 生成基线和相关数据

```bash
nvcc -o orbit_gen orbit_gen.cu -Xcompiler -fopenmp

bash orbit_gen.sh
```

支持三种观测频率（单位 Hz）：

- 1 MHz：`1e6`
- 10 MHz：`1e7`
- 30 MHz：`3e7`

程序会根据频率自动选择输出目录与文件后缀：

- 输出目录：`./earth_<tag>hz/`，其中 `<tag>` 为 `1M / 10M / 30M`
- 输出文件名：
  - `uvw{day}day<tag>.txt`
  - `xyza{day}day<tag>.txt`
  - `xyzb{day}day<tag>.txt`
  - `bll{day}day<tag>.txt`

例如（day=367，30MHz）：
- `./earth_30Mhz_cuda/uvw1day30M.txt`
- `./earth_30Mhz_cuda/xyza1day30M.txt`
- `./earth_30Mhz_cuda/xyzb1day30M.txt`
- `./earth_30Mhz_cuda/bll1day30M.txt`

> 说明：内部计算使用单精度 float，输出仍以 `%.15f` 格式写入文本。

## 多卡运行

30MHz 场景下，核心数组均为 float，单卡 4090（24GB）通常可以支持（显存占用远小于 24GB）。  
需要注意：30MHz 输出文本规模较大，写盘可能成为主要瓶颈（速度取决于磁盘性能）。

如需加速批量 day 生成，推荐多卡“按 day 范围切分”并行运行（最简单稳定，不需要跨卡通信）：

双卡示例：
```bash
CUDA_VISIBLE_DEVICES=0 ./orbit_gen --only=30M --start=1 --end=100 
CUDA_VISIBLE_DEVICES=1 ./orbit_gen --only=30M --start=101 --end=450
```



## 生成$\theta$ 和 $\phi$数据

### 编译
```bash
nvcc -o pix2ang_nest pix2ang_nest.cu -Xcompiler -fopenmp
```
### 运行
```bash
bash run_pix2fang_nest.sh 
```

## 运行模拟

### 基于thrust实现，不分段处理
```bash
编译
nvcc -o vissgen3d_blockage_thrust vissgen3d_blockage_thrust.cu -Xcompiler -fopenmp
nvcc -o vissgen3d_noblockage_thrust vissgen3d_noblockage_thrust.cu -Xcompiler -fopenmp
```
```bash
运行
CUDA_VISIBLE_DEVICES=0 ./vissgen3d_blockage_thrust \
CUDA_VISIBLE_DEVICES=0 ./vissgen3d_noblockage_thrust \
```

### 反演阶段分段处理
```bash
编译
nvcc -O3 -lineinfo -Xcompiler -fopenmp -std=c++14 vissgen3d_recon_seg.cu -o vissgen3d_recon_seg
```
```bash
运行
bash vissgen3d_recon_seg.sh
```
