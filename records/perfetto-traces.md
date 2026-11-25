# Higgs Audio v2 的 Perfetto 时间线

已核对 6 项实验的 22 份 PyTorch Profiler Chrome JSON 时间线。每份文件的 `traceEvents` 都含 CPU 算子和 GPU kernel。下列下载链接固定到各自的 Git 提交。

| 实验 | 归档内的时间线 | 原始数据 |
| --- | --- | --- |
| 纯音频 norm/MLP 路由 | A × 2、B × 2、B2 × 2、A2 × 2 | [下载归档](https://raw.githubusercontent.com/EddieTzy/vllm-omni/30aa5bc7d9007cbddbb9ca0cfbd7680526970f49/benchmarks/higgs_audio_routing_profile/evidence.tar.gz) |
| BOS/EOS delay-pattern logits mask fusion | A × 1、B × 1、B2 × 1、A2 × 2 | [下载归档](https://raw.githubusercontent.com/EddieTzy/vllm-omni/77a2b9b1e704ea356b125a62d0e52f96ec96f9a2/benchmarks/higgs_delay_mask_experiment/profile_rtx_20261125/evidence.tar.gz) |
| 批量 RAS 重复检测 | A × 1、B × 1 | [下载归档](https://raw.githubusercontent.com/EddieTzy/vllm-omni/4235cb5bfcb2ba8a3cb109d84d22a40c19f4a9b5/benchmarks/higgs_ras_repeat_experiment/profile/evidence.tar.gz) |
| LM 音频模式 mask | A × 1、B × 1 | [下载归档](https://raw.githubusercontent.com/EddieTzy/vllm-omni/238147bf1b199b099b3e3a6c394f310c14a0c068/benchmarks/higgs_lm_mode_mask_experiment/profile/evidence.tar.gz) |
| 限制每个请求的音频历史 | A × 1、B × 1 | [下载归档](https://raw.githubusercontent.com/EddieTzy/vllm-omni/8a33ea3776e29937bde8da11becb2d06811e257b/benchmarks/higgs_bounded_history_experiment/profile/evidence.tar.gz) |
| 保留 CPU 请求状态的 delay 更新融合 | A × 2、B × 1 | [下载归档](https://raw.githubusercontent.com/EddieTzy/vllm-omni/7b27a036dbfd2fe97b5369c0100b74fee7e3c492/benchmarks/higgs_delay_state_experiment/profile/evidence.tar.gz) |

## 打开一份 trace

1. 下载对应的 `evidence.tar.gz` 并解压。
2. 在解压目录中找到 `A/trace-1.json` 和 `B/trace-1.json`。音频路由的压缩包在 A、B 目录外还有一层实验路径。
3. 在 [Perfetto UI](https://ui.perfetto.dev/) 选择 **Open trace file** 打开一个 JSON。可以另开一个标签页查看另一组，按同名 CPU 算子、CUDA kernel 或模型阶段对照。

A、A2 是基线，B、B2 是候选实现。每份 trace 覆盖采集到的诊断步；两端的动态合批和采样工作量可能不同。端到端吞吐和延迟使用各实验独立的无 profiler 服务结果。包中的 `*-meta.json`、`*-ops.json` 和事件 JSONL 供核对采样条件和计数。

## 早期实验的留存范围

| 实验 | 可核实的资料 |
| --- | --- |
| B200 Triton 状态融合 | [阶段计时汇总](https://github.com/EddieTzy/vllm-omni/blob/06549a28d920f033de1bce3d0a89307151974d51/benchmarks/higgs_decode_state_fusion_experiment/phase_profile.json)。现有归档中未找到原始时间线。 |
| B200 批量反馈 embedding | [历史实验记录](../optimizations/higgs-v2-packed-feedback/README.md)。现有归档中未找到原始时间线。 |

后续每个实验都在独立 perf 分支保留基线与候选的原始时间线、采集配置和阶段摘要，并将固定版本的下载入口登记在此页。若某轮采集失败或缺少时间线，在该实验记录里注明。仅把含有可解析 `traceEvents` 的文件计入时间线。
