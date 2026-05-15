# shared_operator_plan_tileowned_planaware

第三阶段实验版本：在 `shared_operator_plan_tileowned_planaware` 基础上，引入 **plan-aware tile-owned Recon** 执行路径。

## 核心思路

该版本不推翻原来的 shared/operator-plan 框架，仍然保留：

- Stage-2 的 `hide / vv / mixed` 分类；
- `hide` 任务跳过；
- `vv` 快路径；
- `mixed` 中的局部二次分类；
- Stage-2 block-cyclic 像素负载均衡；
- Recon 热路径中的 `Viss_half` 预加权优化。

新增点是：

- 每个 baseline chunk 仍先生成 `vv_flags / mixed_flags`；
- 再把“至少包含一个非 hidden task 的 sky tile”压缩成 `active_tile_list`；
- Recon 执行时，一个 CUDA block 拥有一个 sky tile；
- block 内部遍历该 tile 在当前 baseline chunk 内的多个 baseline-block；
- 同一个 sky tile 在同一 chunk 内尽量连续处理，减少 `l/m/n` 重复加载和 `Cacc[pix]` 多次写回。

## 编译开关

默认启用 tile-owned plan-aware 路径：

```bash
-DOP_RECON_USE_TILEOWNED_PLAN=1
```

如需回退到上一阶段的 task-list 路径，可编译时关闭：

```bash
-DOP_RECON_USE_TILEOWNED_PLAN=0
```

其他可测参数仍然保留：

```bash
-DOP_RECON_TASK_BL_VALUE=256
-DOP_RECON_PLAN_BL_CHUNK_VALUE=1024
-DOP_RECON_PLAN_BL_CHUNK_VALUE=2048
```

## 建议测试顺序

1. 先跑 `nside=512, 1MHz`，确认结果与 `shared_operator_plan_tileowned_planaware` 基本一致；
2. 再跑 `nside=4096, 10MHz` 的 1 天或 3 天；
3. 对比：
   - `recon_3d_direct_planaware_tileowned_real` 总时间；
   - Stage-2 总时间；
   - 总 kernel time；
   - 输出图 `max/min/rmse`。

## 注意

这是第三阶段实验分支，不建议直接替换当前稳定主线。若该路径在 4096 / 16384 下收益明显，再进一步把 active tile 构建、vv/mixed 分类和 chunk 粒度做细化优化。
