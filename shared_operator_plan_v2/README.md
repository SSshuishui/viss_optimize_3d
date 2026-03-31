This package continues the shared-operator-plan path and focuses on four concrete follow-up optimizations:

1. Explicit task-list dispatch on the recon/adjoint side
   - For each baseline subchunk, build exact tile-block flags: hide / vv / mixed.
   - Compact vv tasks and mixed tasks into separate task lists with CUB DeviceSelect.
   - hide tasks are dropped completely.

2. Dedicated vv-only main path
   - Added a vv-only adjoint kernel with no per-point visibility checks.
   - Mixed tasks go to a separate exact fallback kernel.
   - Inside the mixed kernel, baseline-local vv/mixed indices are compacted into shared-memory task lists, so the vv subset still takes the fast path.

3. Plan generation overlapped with execution
   - Introduced double-buffered plan buffers per GPU.
   - While compute_stream executes the current plan, reduce_stream builds the next baseline-subchunk plan.
   - Host launches chunk execution only after the corresponding plan-ready event is complete.

4. Plan-aware pair-weight generation / scatter
   - pair_weight_half is now generated once on the reducer GPU.
   - Other GPUs receive it by peer copy when possible.
   - If peer scatter is unavailable, the code falls back to local computation on that GPU.

Important notes:
- This round keeps the existing Stage-1 forward kernel shape intact. That kernel already had an explicit vv fast path and hide skip path. The new work is concentrated on the side that benefited more from shared operator plans: the adjoint/recon side.
- The recon path now processes baseline subchunks of OP_RECON_PLAN_BL_CHUNK=512. This is a memory-conscious choice so the task-list buffers remain viable for 30M as well.
- Because this environment does not provide nvcc or GPUs, this package is prepared as a best-effort code update and has not been compiled or benchmarked here.
