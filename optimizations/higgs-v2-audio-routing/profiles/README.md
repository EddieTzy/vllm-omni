# 路由前后 profile 证据

这是独立的诊断记录，不是新的无 profiler 服务吞吐结果。原始证据位于 [higgs-routing-profiling-evidence](https://github.com/EddieTzy/vllm-omni/tree/higgs-routing-profiling-evidence)，固定提交 `30aa5bc7d9007cbddbb9ca0cfbd7680526970f49`。

- [完整说明及复现](https://github.com/EddieTzy/vllm-omni/blob/30aa5bc7d9007cbddbb9ca0cfbd7680526970f49/benchmarks/higgs_audio_routing_profile/README.md)。
- [八份 Chrome trace、事件、日志、源码及脚本归档](https://github.com/EddieTzy/vllm-omni/blob/30aa5bc7d9007cbddbb9ca0cfbd7680526970f49/benchmarks/higgs_audio_routing_profile/evidence.tar.gz)，SHA256 `054a8fc25293978eadb226565304fc0f5f1ff240c1595fa264dc335660d6c3f9`。
- 本目录 [metrics.json](metrics.json) 是该固定版本的汇总副本，不是完整 trace。

四次启动 A/B/A2/B2；每次 32 warmup +32 measured，语料取既有中英交错列表前 32 条。带 trace 的结果与原先 64 measured 服务协议分开。

32-token forward 汇总丢弃该形状前 16 次，排除 trace forward 及其后两次；选中样本覆盖两波请求，并非仅 measured 波。每次另取两个真实 32 行全音频 forward 做 torch.profiler，八份 trace 均确认输入是音频 token。逐步计时使用同 stream CUDA event，延后读取，不在每个 forward 后同步。

forward host 中位数 base 34.639/35.248ms，优化 17.069/17.386ms。outside-forward remainder 中位数 base 16.942/16.933ms，优化 16.945/17.083ms；它包含 logits、sampling、runner/scheduler、诊断和等待，不能标成纯采样耗时。

codec shape 和输出长度未配对，因此 codec 各轮中位数只作描述，不能归因成 codec 加速。带 profiler 的 kernel 总和、CPU span 和无 profiler wall time不直接相加。此记录没有增加质量、TP 或其他硬件验证。
