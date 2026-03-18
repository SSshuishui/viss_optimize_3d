# viss_recon_tile

这一目录对应当前这一系列工作的渐进优化版本，也是推荐优先使用和继续扩展的版本。

核心特点：

- Viss 阶段使用前后共轭优化
- Viss 阶段进行 tile 分块
- Recon 阶段进行 tile 分块
- 可视为从基础三维方法逐步演化到面向并行优化与大规模计算的版本

## 适用场景

- 10MHz / 30MHz 等高分辨率大规模实验
- 多 GPU 场景
- 并行计算优化与论文实验

## 特点概述

相对于前几个目录，该版本进一步引入：

- tile 级空间划分
- 面向遮挡判断和反演计算的分块加速
- 更适合做高性能实现和工程化扩展

## 编译

```bash
nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp dcf_mb_gen.cu -o dcf_mb_gen
```

```bash
nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp main_3d_viss_recon.cu -o main_3d_viss_recon
```


## 运行

```bash
bash main_3d_viss_recon.sh
```


## 阅读建议

建议结合总目录 `README.md` 使用：

1. 先在总目录了解整体结构
2. 再回到本目录阅读和运行当前版本
3. 如需理解优化来源，可回看前几个目录的 README
