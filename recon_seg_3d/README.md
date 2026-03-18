# recon_seg_3d

在传统三维方法基础上，对反演阶段（Recon）引入分段处理。

这一版本的核心思想是：

- Viss 阶段仍保持传统三维逻辑
- 按 chunk 分块处理，避免一次性处理全部时间序列带来的压力

## 适用场景

- 想在不大改 Viss 主流程的前提下，降低反演阶段的计算和内存压力
- 作为从基础实现走向优化实现的第一步

## 基线数据
基线和 $\phi$、 $\theta$ 参考 [vissgen3d_thrust]('../vissgen3d_thrust')

## 编译

```bash
nvcc -O3 -lineinfo -Xcompiler -fopenmp -std=c++14 vissgen3d_recon_seg.cu -o vissgen3d_recon_seg
```

## 运行

```bash
bash vissgen3d_recon_seg.sh
```

## 特点

- 保持三维反演框架
- 对 Recon 阶段做 segment 分段
- 相比不分段版本，更适合大规模数据
- 仍未利用 Viss 阶段的前后共轭 / 对称关系

## 与其他目录的关系

- 相比 [vissgen3d_thrust](`vissgen3d_thrust/`)：加入了反演阶段分段
- 相比 [viss_conj_recon_seg](`viss_conj_recon_seg/`)：还没有加入 Viss 阶段共轭优化

## 建议

如果你已经验证该目录版本可运行，下一步建议继续阅读：

- `../viss_conj_recon_seg/README.md`
