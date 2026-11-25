# 保留 CPU 请求状态的 delay 更新融合

阶段耗时下降，C32 吞吐下降 11.01%，平均耗时增加 19.02%。

RTX PRO 6000，C32，基线包含 #11068 和 #11119 的模型改动。四项候选分别对照同一实现。各自 A/B/B2/A2，每轮独立启动。配对验证中 32/32 完整音频及逐帧码一致，200 个采样/RNG 事件一致。

| 指标 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 吞吐（音频秒/秒） | 25.7309 | 22.8977 | -11.01% |
| 平均 E2E（秒） | 3.3790 | 4.0216 | +19.02% |
| 平均请求 RTF | 0.8168 | 0.9339 | +14.34% |
| 首个 PCM P50（秒） | 3.3899 | 3.9165 | +15.54% |
| 首个 PCM P95（秒） | 4.4742 | 5.7393 | +28.27% |

[实现、基线、逐轮结果与限制](https://github.com/EddieTzy/vllm-omni/tree/7b27a036dbfd2fe97b5369c0100b74fee7e3c492/benchmarks/higgs_delay_state_experiment) · [单独的阶段 profile](https://github.com/EddieTzy/vllm-omni/tree/7b27a036dbfd2fe97b5369c0100b74fee7e3c492/benchmarks/higgs_delay_state_experiment/profile)
