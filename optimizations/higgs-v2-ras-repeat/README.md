# 批量 RAS 重复检测

重复检测变快，C32 吞吐下降 6.45%。

RTX PRO 6000，C32，基线包含 #11068 和 #11119 的模型改动。四项候选分别对照同一实现。各自 A/B/B2/A2，每轮独立启动。配对验证中 32/32 完整音频及逐帧码一致，200 个采样/RNG 事件一致。

| 指标 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 吞吐（音频秒/秒） | 24.4329 | 22.8563 | -6.45% |
| 平均 E2E（秒） | 3.6682 | 3.7123 | +1.20% |
| 平均请求 RTF | 0.8957 | 0.8822 | -1.52% |
| 首个 PCM P50（秒） | 3.5766 | 3.7144 | +3.85% |
| 首个 PCM P95（秒） | 5.0697 | 5.1306 | +1.20% |

[实现、基线、逐轮结果与限制](https://github.com/EddieTzy/vllm-omni/tree/4235cb5bfcb2ba8a3cb109d84d22a40c19f4a9b5/benchmarks/higgs_ras_repeat_experiment) · [单独的阶段 profile](https://github.com/EddieTzy/vllm-omni/tree/4235cb5bfcb2ba8a3cb109d84d22a40c19f4a9b5/benchmarks/higgs_ras_repeat_experiment/profile)
