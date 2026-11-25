# LM 音频模式 mask

平均耗时下降 3.21%，吞吐下降 2.69%，两次候选启动的吞吐波动较大。

RTX PRO 6000，C32，基线包含 #11068 和 #11119 的模型改动。四项候选分别对照同一实现。各自 A/B/B2/A2，每轮独立启动。配对验证中 32/32 完整音频及逐帧码一致，200 个采样/RNG 事件一致。

| 指标 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 吞吐（音频秒/秒） | 21.9579 | 21.3679 | -2.69% |
| 平均 E2E（秒） | 3.9301 | 3.8038 | -3.21% |
| 平均请求 RTF | 0.9239 | 0.8822 | -4.51% |
| 首个 PCM P50（秒） | 3.8314 | 3.7526 | -2.06% |
| 首个 PCM P95（秒） | 5.3932 | 5.3145 | -1.46% |

[实现、基线、逐轮结果与限制](https://github.com/EddieTzy/vllm-omni/tree/238147bf1b199b099b3e3a6c394f310c14a0c068/benchmarks/higgs_lm_mode_mask_experiment) · [单独的阶段 profile](https://github.com/EddieTzy/vllm-omni/tree/238147bf1b199b099b3e3a6c394f310c14a0c068/benchmarks/higgs_lm_mode_mask_experiment/profile)
