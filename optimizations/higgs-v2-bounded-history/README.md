# 限制每个请求的音频历史

内部历史在 200 帧时由 12,800 B 降至 448 B。基线有 40.6 秒长音频，吞吐差异无法归因。

RTX PRO 6000，C32，基线包含 #11068 和 #11119 的模型改动。四项候选分别对照同一实现。各自 A/B/B2/A2，每轮独立启动。配对验证中 32/32 完整音频及逐帧码一致，200 个采样/RNG 事件一致。

| 指标 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 吞吐（音频秒/秒） | 14.0312 | 22.6846 | +61.67% |
| 平均 E2E（秒） | 4.2849 | 3.9511 | -7.79% |
| 平均请求 RTF | 0.9362 | 0.9351 | -0.13% |
| 首个 PCM P50（秒） | 3.9403 | 3.9324 | -0.20% |
| 首个 PCM P95（秒） | 5.4566 | 5.5188 | +1.14% |

[实现、基线、逐轮结果与限制](https://github.com/EddieTzy/vllm-omni/tree/8a33ea3776e29937bde8da11becb2d06811e257b/benchmarks/higgs_bounded_history_experiment) · [单独的阶段 profile](https://github.com/EddieTzy/vllm-omni/tree/8a33ea3776e29937bde8da11becb2d06811e257b/benchmarks/higgs_bounded_history_experiment/profile)
