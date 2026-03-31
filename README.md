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

- [vissfast](`vissfast/`)  (10Mhz 192s, 4*4090)
  - viss阶段加入快速判断通道，提高速度（同[二维算法](https://github.com/SSshuishui/viss_optimize_ws)）

- [half_native](`half_native`)  (10Mhz 187s, 4*4090)
  - 基线部分和后续全程只保留一半的内容，在数学上保持一致

- [system_viss_recon](`system_viss_recon`)  (10Mhz 178s, 4*4090)
  - 去掉blockage的判断，recon阶段只考虑加入遮挡的逻辑
  - 把viss和recon两个阶段统一了，复用中间部分计算遮挡的逻辑，提高速度

- [shared_operator_plan](`shared_operator_plan`)  (10Mhz 156s, 4*4090)
  - 同一份 hide/vv/mixed 分类不再两阶段重复做
  - hide 被更干净地跳过
  - vv 被更干净地走快路径
  - 中间准备层被 chunk 化并复用

- [shared_operator_plan](`shared_operator_plan`)  (10Mhz 117s, 4*4090)
  - 1. 把recon/adjoint 侧从“内核里一边判断一边做”推进成了“先生成精确任务，再按任务执行”
    按 baseline 子块分批处理
    对每个 tile × baseline-block 精确分类成：hide / vv / mixed， 用 CUB DeviceSelect::Flagged 把：vv 任务压成一个任务列表， mixed 任务压成一个任务列表， hide 直接不进入执行列表
    - 不是所有 task 都发进同一个 recon kernel 再分支，而是：
    vv list 单独执行
    mixed list 单独执行
    hide 根本不执行
  - 2. vv 专用主路径
    - vv kernel 只处理全可见任务：不做点级 visibility 判断， 直接 phase + weight + Viss 累加
    - mixed kernel 只处理混合任务，里面又做了一层 shared-memory task list： 先把这个任务块里的 baseline 再精确分成：（block-local vv / block-local mixed / hidden）
    然后：本地 vv 仍走无遮挡快路径 ｜ 本地 mixed 才做 point visibility check。 即使进入 mixed kernel，也尽量把里面仍可走快路径的部分抽出来。
  - 3. plan generation 与执行重叠
    - 加了 double-buffered plan buffer。 每个 GPU 上有两套 plan buffer： 当前 plan 在 compute_stream 上执行， 下一 plan 在 reduce_stream 上构建
    - 执行顺序变成：先为 chunk 0、chunk 1 预生成 plan， compute_stream 执行当前 chunk， 同时 reduce_stream 为下一轮 chunk 生成新 plan， 通过 event 做 buffer 复用
    - 这一步的意义是：plan build 不再完全串在执行前面，新加的“计划层”尽量被流水隐藏
  - 4. plan-aware reduce / scatter
    - 把 pair_weight_half 从“每张卡都各算一遍”改成了：
      - reducer GPU 只算一次
      - 其他 GPU：能 peer copy 就直接拷过去，不能 peer copy 再回退到本地算
    - 消掉了多卡上重复的权重生成，让 pair_weight_half 真正变成 shared operator plan 的一部分，而不是每卡各自现算的局部状态


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
