# PR 前检查经验

贡献检查记录。后续更新或提交 PR 前，按改动涉及的范围核对；不能用这份清单代替当前源码、实机验证或仓库要求。

## Higgs v2 mask：边界、平台和 kernel 读写

来源：[vLLM-Omni #11119 的评审](https://github.com/vllm-project/vllm-omni/pull/11119#pullrequestreview-6055542476)。修正提交：[`b19471884`](https://github.com/EddieTzy/vllm-omni/commit/b19471884f41310fc1429357a5eca95bdcbcb0db)。记录日期：2026-11-25。

### 1. 向量化不能悄悄改变非法输入的行为

旧实现使用 token ID 索引张量，越界会抛错；改成 `v == eos` 等比较后，越界 ID 可能只产生全 false 的比较结果，静默继续，甚至无法正常收尾。

提交前检查：

- 索引改比较、掩码或自定义 kernel 时，同时检查合法输入等价性和非法输入失败方式。
- 配置中固定的边界约束应在配置或初始化时一次性校验，避免放进逐步执行的热路径。
- BOS/EOS 等 ID 至少覆盖负数、等于词表大小、超过词表大小；也测试合法边界及配置序列化往返。
- 本次两种 stream token × `-1/1026/1027` 的 6 个用例在旧配置均未抛错；修正后明确抛 `ValueError`。

### 2. `Tensor.is_cuda` 不等于 NVIDIA

PyTorch ROCm/HIP 张量也可能满足 `.is_cuda`。只看张量属性，会把 AMD 意外送入只在 NVIDIA 上验证的新路径。

提交前检查：

- 同时核对平台、依赖可用性与张量设备；本次使用 `current_platform.is_cuda() and HAS_TRITON and cb_logits.is_cuda`。
- 快路径的平台范围必须和测试的 skip 条件、CI 标记及 PR 支持声明一致。
- 在不支持快路径的平台上保留原实现，并延迟导入平台相关 kernel。
- 分发模拟测试应检查回退后的结果；模拟不能替代 AMD 实机测试。
- ROCm 继续使用原路径与 ROCm 获得新 kernel 加速，是两个不同声明。后者需要 AMD 实机等价性验证；若宣称性能收益，还需对应设备的性能证据。

### 3. 原地掩码只需写被禁止的位置

若允许的 logits 必须原样保留，不必把它们读出再写回。可用写掩码：

```python
tl.store(ptr, -float("inf"), (v < vocab_size) & ~allowed)
```

提交前检查：

- 写掩码同时约束有效地址与需修改的位置，不能只看是否减少代码行数。
- 覆盖 NaN/Inf、FP32/FP16/BF16、不同 delay 状态、strided 张量及 padding 不变、空 batch。
- 减少读写不能直接写成额外端到端加速，收益需要单独测量。

### 4. 评审修改后，证据必须跟随提交版本

- 本次修正后的定点测试为 RTX PRO 6000 上 `38 passed, 1 warning`，本地 pre-commit 检查通过。
- 既有端到端性能和完整音频配对来自上一版；kernel 修改后，应在 PR 中明确哪些已重测、哪些沿用历史结果，不能称新 head 已完成旧版全部验证。
- CI 必须看新 head 的结果，不能沿用旧 head 的绿色状态。
- 在对应 GitHub review thread 中说明修正提交和验证范围，保留给 reviewer 复查；无需再重复回复通知邮件。
