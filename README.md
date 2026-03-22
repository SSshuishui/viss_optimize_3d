# viss_optimize_3d

传统三维天图成像方法的渐进式优化工程。

本仓库围绕“可见度模拟（Viss）+ 图像反演（Recon）”两阶段流程，提供从基础版本到逐步优化版本的多套实现。不同目录对应不同优化阶段；使用某一套实现时，请优先进入对应目录阅读该目录下的 `README.md`。

## 目录说明

- [vissgen3d_thrust](`vissgen3d_thrust/`)
  - 基于 Thrust 的传统三维实现
  - 不分段处理
  - 适合作为基础版本或结果对照版本

- [recon_seg_3d](`recon_seg_3d/`)
  - 在传统三维实现基础上，对反演阶段进行分段处理
  - 适合在不改动 Viss 主逻辑的情况下缓解反演阶段压力

- [viss_conj_recon_seg](`viss_conj_recon_seg/`)
  - 在反演阶段分段基础上，进一步在 Viss 阶段利用基线前后对称 / 共轭关系进行加速
  - 适合研究“Viss 半对称 + Recon 分段”的组合收益

- [viss_recon_tile](`viss_recon_tile/`)
  - 在前述基础上，进一步对 Viss 与 Recon 两阶段都引入 tile 思想
  - 可以理解为当前这一系列工作的渐进优化版本
  - 适合做性能优化、并行计算加速与论文实验的主实现

- [vissfast](`vissfast/`)
  - viss阶段加入快速判断通道，提高速度（同[二维算法](https://github.com/SSshuishui/viss_optimize_ws)）

- [half_native](`half_native`)
  - 基线部分和后续全程只保留一半的内容，在数学上保持一致

- [system_viss_recon](`system_viss_recon`)
  - 去掉blockage的判断，recon阶段只考虑加入遮挡的逻辑
  - 把viss和recon两个阶段统一了，复用中间部分计算遮挡的逻辑，提高速度

## 使用建议

如果你是第一次接触本工程，建议按以下顺序阅读与测试：

1. [vissgen3d_thrust](`vissgen3d_thrust/`)
2. [recon_seg_3d](`recon_seg_3d/`)
3. [viss_conj_recon_seg](`viss_conj_recon_seg/`)
4. [viss_recon_tile](`viss_recon_tile/`)
5. [vissfast](`vissfast/`)
6. [half_native](`half_native`)
7. [system_viss_recon](`system_viss_recon`)

这样更容易理解每一步优化引入的位置、作用和收益。
