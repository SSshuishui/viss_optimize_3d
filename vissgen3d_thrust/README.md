# vissgen3d_thrust

基于 Thrust 的传统三维天图成像实现。

这一版本可以看作整个工程的基础版本：

- Viss 阶段采用传统三维实现
- 不进行 segment 分段处理
- 适合作为结果基准、功能验证版本或后续优化对照版本

## 适用场景

- 验证最基础的三维可见度模拟流程
- 与后续优化版本做精度和性能对照
- 小规模数据测试

### 1. 生成基线和相关数据

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

例如（day=1，30MHz）：

- `./earth_30Mhz_cuda/uvw1day30M.txt`
- `./earth_30Mhz_cuda/xyza1day30M.txt`
- `./earth_30Mhz_cuda/xyzb1day30M.txt`
- `./earth_30Mhz_cuda/bll1day30M.txt`

> 说明：内部计算使用单精度 `float`，输出仍以较高精度文本格式写盘。

### 2. 生成 `theta` 和 `phi` 数据

编译：

```bash
nvcc -o pix2ang_nest pix2ang_nest.cu -Xcompiler -fopenmp
```

运行：

```bash
bash run_pix2fang_nest.sh
```

## 多卡运行建议

30MHz 场景下，核心数组均为 `float`，单卡 4090（24GB）通常可以支持核心计算；但输出文本规模很大，写盘可能成为主要瓶颈，速度依赖磁盘性能。

如需加速批量 day 生成，推荐最简单稳定的方式：**按 day 范围切分，多卡并行运行**。这样不需要跨卡通信，易于管理。

双卡示例：

```bash
CUDA_VISIBLE_DEVICES=0 ./orbit_gen --only=30M --start=1 --end=100
CUDA_VISIBLE_DEVICES=1 ./orbit_gen --only=30M --start=101 --end=450
```

## 编译

```bash
nvcc -o vissgen3d_thrust vissgen3d_thrust.cu -Xcompiler -fopenmp
```

## 运行

考虑遮挡版本：

```bash
CUDA_VISIBLE_DEVICES=0 ./vissgen3d_thrust 1e6 1
```

不考虑遮挡版本：

```bash
CUDA_VISIBLE_DEVICES=0 ./vissgen3d_thrust 1e6 0
```

## 特点

- 实现直接
- 逻辑清晰
- 易于作为参考基线
- 对大规模场景不够友好，整体计算与内存压力较大

## 建议

如果目标是进一步做大规模三维重建或并行优化，建议在理解该目录实现后，继续阅读：

- `../recon_seg_3d/README.md`
