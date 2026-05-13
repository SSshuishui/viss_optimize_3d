# viss_optimize_3d

传统三维天图成像方法的渐进式优化工程。

本仓库围绕“可见度模拟（Viss）+ 图像反演（Recon）”两阶段流程，提供从基础版本到逐步优化版本的多套实现。不同目录对应不同优化阶段；使用某一套实现时，请优先进入对应目录阅读该目录下的 `README.md`。

## 目录说明

### 基础实现与渐进式优化

- [`vissgen3d_thrust`](vissgen3d_thrust/)
  - 基于 Thrust 的传统三维实现。
  - 不分段处理。
  - 适合作为基础版本或结果对照版本。

- [`recon_seg_3d`](recon_seg_3d/)
  - 在传统三维实现基础上，对反演阶段进行分段处理。
  - 适合在不改动 Viss 主逻辑的情况下缓解反演阶段压力。

- [`viss_conj_recon_seg`](viss_conj_recon_seg/)
  - 在反演阶段分段基础上，进一步在 Viss 阶段利用基线前后对称 / 共轭关系进行加速。
  - 适合研究“Viss 半对称 + Recon 分段”的组合收益。

- [`viss_recon_tile`](viss_recon_tile/)
  - 在前述基础上，进一步对 Viss 与 Recon 两阶段都引入 tile 思想。
  - 可以理解为当前这一系列工作的渐进优化基础版本。
  - 适合做性能优化、并行计算加速与论文实验的主实现之一。

### Viss 与算子复用优化  

- [`half_native`](half_native/)  （10MHz 约 187s，4×4090）
  - Viss 阶段加入快速判断通道，提高速度。
  - 思路与[二维算法](https://github.com/SSshuishui/viss_optimize_ws)中的快速判断通道一致。
  - 基线部分和后续流程只保留一半内容。
  - 利用基线共轭 / 半对称关系，在数学上保持结果一致。

- [`system_viss_recon`](system_viss_recon/)  （10MHz 约 178s，4×4090）
  - 去掉 Viss 阶段重复的 blockage 判断，将遮挡逻辑重点放在 Recon 阶段。
  - 统一 Viss 和 Recon 两阶段的中间算子组织方式，复用中间遮挡计算逻辑，提高速度。

- [`shared_operator_plan`](shared_operator_plan/)  （10MHz 约 156s，4×4090）
  - 同一份 hide / vv / mixed 分类不再在两阶段重复构建。
  - hide 更干净地跳过。
  - vv 更干净地进入快路径。
  - 中间准备层被 chunk 化并复用。

- [`shared_operator_plan_tasklist_overlap`](shared_operator_plan_tasklist_overlap/)  （10MHz 约 117s，4×4090）
  - 原“第二版 shared_operator_plan”建议命名为该目录名，避免与上一阶段重名。
  - Recon / adjoint 侧从“内核里一边判断一边做”推进为“先生成精确任务，再按任务执行”。
  - 按 baseline 子块分批处理，对每个 `tile × baseline-block` 精确分类为 hide / vv / mixed。
  - 使用 CUB `DeviceSelect::Flagged` 将 vv 任务和 mixed 任务分别压缩成任务列表，hide 不进入执行列表。
  - vv list 单独执行，mixed list 单独执行，避免所有 task 混在同一个 recon kernel 中再分支。
  - vv 专用主路径只处理全可见任务，不做点级 visibility 判断，直接进行 phase + weight + Viss 累加。
  - mixed kernel 内部再做一层 shared-memory task list，将任务块内 baseline 继续分成 block-local vv / block-local mixed / hidden，使 mixed kernel 中仍可走快路径的部分尽量走快路径。
  - 加入 double-buffered plan buffer，使 plan generation 与执行重叠：当前 plan 在 `compute_stream` 上执行，下一 plan 在 `reduce_stream` 上构建，通过 event 控制 buffer 复用。
  - 引入 plan-aware reduce / scatter：`pair_weight_half` 由 reducer GPU 只计算一次，其他 GPU 优先 peer copy，不能 peer copy 时再回退到本地计算。

### Stage-2 负载均衡与 Recon 热路径优化

- [`shared_operator_plan_load_balance`](shared_operator_plan_load_balance/)
  - 用于统计每个 GPU 上的负载。
  - 主要服务于 Stage-2 多 GPU 负载不均问题的定位。

- [`shared_operator_plan_stage2_balance`](shared_operator_plan_stage2_balance/)  （10MHz 约 107s，4×4090）
  - Stage-2 改成 block-cyclic 像素分区。
  - 将全图按固定大小的 Recon pixel block 切块，并按 round-robin 分给不同 GPU。
  - 这样每张 GPU 会拿到更混合的天空区域，而不是原来连续大块区域，从而缓解 vv 任务在 GPU 间分布不均的问题。
  - 默认参数为 `--recon_balance_block_tiles 256`。
    - 即每个 Stage-2 块大小为 `256 * RECON_TILE_PIX_HOST`。
    - 当前 `RECON_TILE_PIX_HOST=256`，所以默认一个负载均衡块是 65536 个像素。
  - Stage-2 使用独立的 Recon 像素数组：`d_l_recon`、`d_m_recon`、`d_n_recon`、`recon_n_chunk`。
  - Stage-1 仍然使用原来的连续 chunk，Stage-2 改成新的 block-cyclic 分区。
  - Stage-2 的 tile cone metadata 基于新分区重建。
  - 输出文件仍然保持全局像素顺序：按全局 block 顺序遍历，找到该 block 属于哪张 GPU，再从对应 GPU 的 `d_Cacc` 偏移拷出并写回。

- [`shared_operator_plan_recon_fastpath`](shared_operator_plan_recon_fastpath/)
  - 基于 `shared_operator_plan_stage2_balance` 的 Recon 热路径小优化版本。
  - 不改变 shared/operator-plan 总体框架，不切换 tile-owned recon，不改变 Stage-2 block-cyclic 负载均衡策略。
  - 在 Recon 前预先将 `pair_weight_half` 乘入 `Viss_half`，从 vv / mixed Recon 内层循环中移除 `pairw` 读取、分支判断和逐像素乘权操作。
  - 将 `isfinite(Viss)` 检查从 `pixel × baseline` 内层循环移动到预加权 kernel 中，只对每条 half-baseline 检查一次。
  - 默认保持 `OP_RECON_TASK_BL=128`、`OP_RECON_PLAN_BL_CHUNK=512`、`OP_RECON_TASK_TILE_PIX=256`，便于和上一版本公平对比。
  - 支持通过编译宏继续测试：`OP_RECON_TASK_BL_VALUE=256`、`OP_RECON_PLAN_BL_CHUNK_VALUE=1024/2048`。
  - 已观测 nsys kernel summary：
    - nside=512，1MHz：总 kernel time 约 `6.990s -> 6.048s`，Recon vv 约 `1.637s -> 0.891s`。
    - nside=4096，10MHz，3 天：总 kernel time 约 `1306.619s -> 984.615s`，Recon vv 约 `778.962s -> 458.478s`。
  - 适合作为当前 shared 主线之后的下一阶段稳定优化版本。

## 使用建议

如果你是第一次接触本工程，建议按以下顺序阅读与测试：

1. [`vissgen3d_thrust`](vissgen3d_thrust/)
2. [`recon_seg_3d`](recon_seg_3d/)
3. [`viss_conj_recon_seg`](viss_conj_recon_seg/)
4. [`viss_recon_tile`](viss_recon_tile/)
5. [`vissfast`](vissfast/)
6. [`half_native`](half_native/)
7. [`system_viss_recon`](system_viss_recon/)
8. [`shared_operator_plan`](shared_operator_plan/)
9. [`shared_operator_plan_tasklist_overlap`](shared_operator_plan_tasklist_overlap/)
10. [`shared_operator_plan_load_balance`](shared_operator_plan_load_balance/)
11. [`shared_operator_plan_stage2_balance`](shared_operator_plan_stage2_balance/)
12. [`shared_operator_plan_recon_fastpath`](shared_operator_plan_recon_fastpath/)

这样更容易理解每一步优化引入的位置、作用和收益。

## 当前建议主线

当前建议将以下版本作为后续实验主线：

```text
shared_operator_plan_recon_fastpath
```

如果该版本在更大规模 nside=16384 场景下仍然由 Recon 主导，再进一步研究 plan-aware tile-owned recon，而不是直接替换当前已经跑通的 shared/operator-plan 框架。
