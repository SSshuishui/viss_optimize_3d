# viss_conj_recon_seg

在反演阶段分段处理的基础上，进一步对 Viss 阶段引入共轭 / 半对称优化。

这一版本利用了基线数据中的前后对称关系：

- 每组基线前一半与后一半互为相反数
- 可在 Viss 阶段只计算一半，另一半通过共轭恢复

## 适用场景

- 想验证“Viss 共轭优化 + Recon 分段”的组合收益
- 想在保持三维方法框架的同时，进一步降低 Viss 阶段计算量

## 核心思路

- Viss 阶段：利用前后共轭关系，只算半边
- Recon 阶段：按 segment 分段


## 基线数据
基线和 $\phi$、 $\theta$ 参考 [vissgen3d_thrust]('../vissgen3d_thrust')

## 编译

```bash
nvcc -O3 -lineinfo -Xcompiler -fopenmp -std=c++14 viss_conj_recon_seg.cu -o viss_conj_recon_seg
```

## 运行

```bash
bash viss_conj_recon_seg.sh
```

## 特点

- 比 [viss_recon_seg_3d](`../viss_recon_seg_3d/`) 更进一步
- Viss 阶段显著减少重复计算
- 仍主要属于传统三维框架下的渐进优化

## 建议

如果你的目标是继续做更强的并行优化或 tile 级别优化，建议继续阅读：

- [viss_recon_tile](`../viss_recon_tile/README.md`)
