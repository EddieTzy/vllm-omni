# EddieTzy vLLM-Omni 推理优化与贡献记录

按状态查看进展，再按模型和优化位置查看实现、实验与取舍。此 `docs/contribution-log` 分支用于记录；实现保存在各自的 perf 分支，上游 PR 保持独立。[项目原始 README](https://github.com/vllm-project/vllm-omni/blob/main/README.md)。

最后整理：2026-11-25（北京时间）。本次增加 Fish S2-Pro 首次完整 prefill 的 C32 流式克隆对照和六份 Perfetto 时间线。

最新进展：Higgs v2 音频路由已合并 [PR #11068](https://github.com/vllm-project/vllm-omni/pull/11068)，于北京时间 2026-11-25 01:58:56 进入上游 `main`；logits mask 融合已提交 [PR #11119](https://github.com/vllm-project/vllm-omni/pull/11119)，RTX PRO 6000 C32 吞吐 +19.44%、平均端到端耗时 -14.10%、RTF 1.133→.996。旧路由优化的新卡补充对照为 C32 吞吐 +85.72%、RTF 1.845→1.150，单独归档。Triton 状态融合已完成实际采样路径接入和配对验证，但增量服务收益较小，暂不单独提交状态融合 PR。批量反馈嵌入实验未形成稳定收益，保留不采用记录。

## CosyVoice3 实验

[四个 perf 分支和 11 份可直接下载的 Perfetto 时间线](records/cosyvoice3.md)。普通合批流量与显式 seed 流量分别记录；已有开关的收益与新增原型的结果分别归档。

## 状态总览

| 分类 | 数量 | 归类规则 |
| --- | ---: | --- |
| [已合并](#已合并) | 1 | 已核实进入上游 |
| [已提 PR](#已提-pr) | 1 | 已创建且仍开放，不能记成已合并 |
| 待验收 | 1 | 有界历史已完成独立分支回归 |
| [已测有收益或取舍](#已测有收益或取舍) | 2 | 已有实测信号，同时明确回退和限制 |
| [无稳定收益或暂不采用](#无稳定收益或暂不采用) | 5 | 保留负面结果和停止原因 |
| [CosyVoice3 独立实验归档](records/cosyvoice3.md) | 4 | 含路径诊断、已有选项和原型对照 |
| 已关闭或归档 | 0 | 已关闭 PR 或被替代的历史版本 |

## 已合并

上游合并提交：[`14a01235186941ad7a26ce258a4ca27c45cb6ebf`](https://github.com/vllm-project/vllm-omni/commit/14a01235186941ad7a26ce258a4ca27c45cb6ebf)。

| 模型 | 改动 | 已观测结果与范围 | 入口 |
| --- | --- | --- | --- |
| Higgs Audio v2 | 统一纯音频 norm/MLP 路由 | B200 历史 C32 +50.8%；RTX PRO 6000 补充 C32 +85.72%、RTF 1.845→1.150；受控 32/32 完整音频相同。带 profiler 的证据单独归档。 | [PR #11068](https://github.com/vllm-project/vllm-omni/pull/11068) · [perf 分支](https://github.com/EddieTzy/vllm-omni/tree/perf/higgs-v2-audio-routing) · [详情](optimizations/higgs-v2-audio-routing/README.md) |

## 已提 PR

| 模型 | 改动 | 已观测结果与范围 | 入口 |
| --- | --- | --- | --- |
| Higgs Audio v2 | BOS/EOS logits mask Triton 融合 | RTX PRO 6000，C32 吞吐 +19.44%、平均耗时 -14.10%、RTF 1.133→.996；29 项测试通过，受控 32/32 音频相同，尚未合并。 | [PR #11119](https://github.com/vllm-project/vllm-omni/pull/11119) · [perf 分支](https://github.com/EddieTzy/vllm-omni/tree/perf/higgs-v2-logits-mask) · [详情](optimizations/higgs-v2-logits-mask/README.md) |

## 待验收

| 模型 | 改动 | 结果 | 入口 |
| --- | --- | --- | --- |
| Higgs Audio v2 | 限制每个请求的音频历史 | 内部历史在 200 帧时由 12,800 B 降至 448 B。基线有 40.6 秒长音频，吞吐差异无法归因。 | [详情](optimizations/higgs-v2-bounded-history/README.md) |

## 已测有收益或取舍

| 模型 | 改动 | 已观测结果与决定 | 入口 |
| --- | --- | --- | --- |
| Higgs Audio v2 | Triton delay/EOS/历史状态融合 | B200，基线已含音频路由优化；C32 吞吐额外 +1.64%，平均耗时 -2.01%，首包 P95 +2.43%。增益小且存在尾延迟回退，作为实验保留，尚未提 PR。 | [perf 分支](https://github.com/EddieTzy/vllm-omni/tree/perf/higgs-v2-triton-state) · [详情](optimizations/higgs-v2-triton-state/README.md) |
| Higgs Audio v2 | LM 音频模式 mask | 平均耗时下降 3.21%，吞吐下降 2.69%，两次候选启动的吞吐波动较大。 | [详情](optimizations/higgs-v2-audio-mode-mask/README.md) |

## 无稳定收益或暂不采用

| 模型 | 改动 | 已观测结果与决定 | 入口 |
| --- | --- | --- | --- |
| Higgs Audio v2 | 批量反馈 embedding lookup | 组件变快，但历史服务初筛 C32 吞吐约 -2.45%，C1 约 -1.15%，C16 无一致改善；停止单独 PR。代码和完整原始材料尚未发布为 perf 分支。 | [详情与证据限制](optimizations/higgs-v2-packed-feedback/README.md) |
| Higgs Audio v2 | 批量 RAS 重复检测 | 重复检测变快，C32 吞吐下降 6.45%。 | [详情](optimizations/higgs-v2-ras-repeat/README.md) |
| Higgs Audio v2 | 保留 CPU 请求状态的 delay 更新融合 | 阶段耗时下降，C32 吞吐下降 11.01%，平均耗时增加 19.02%。 | [详情](optimizations/higgs-v2-delay-state/README.md) |
| Higgs Audio v3 | 流式 codec 输入校验 | C32 校验同步由两次减为一次。吞吐 36.85 → 35.11 音频秒/秒，首包 P95 373 → 402 毫秒。保留实验，暂不提交。 | [详情](optimizations/higgs-v3-codec-validation/README.md) |
| Fish S2-Pro | 首次完整 prefill embedding | RTX PRO 6000，C32 吞吐 23.47 → 23.34 音频秒/秒（−0.56%），首包 P95 增加 3.50%。三组 trace 均消除了完整 prefill 的双向拷贝，保留实验。 | [详情](optimizations/fish-s2-mrv2-prefill-embeddings/README.md) |

## 详细记录

[Perfetto 原始时间线下载与打开方法](records/perfetto-traces.md)

- [每项优化目录](optimizations/README.md)：动机、代码位置、原理、基线、profile、性能、正确性和取舍。
- [记录模板](optimizations/TEMPLATE.md) · [RTF、cost、首包与 profile 口径](records/performance-methodology.md)。
- [Higgs v2 模型记录](records/higgs-audio-v2.md) · [结构化索引](records/entries.json)。
- [路由 profiling 证据分支](https://github.com/EddieTzy/vllm-omni/tree/higgs-routing-profiling-evidence)：八份真实 CPU/CUDA trace、阶段统计和复现资料。它不替代原始无 profiler 性能对照。

## 维护规则

一项优化一个独立实现分支和详细目录，记录精确 base/head、叠加依赖、硬件、权重、软件、请求集、预热与重复顺序。没有数据的字段标记未测，不填零，不由吞吐倒算请求 RTF 或耗时。

性能分支、已提交 PR、已合并 PR 是不同状态；正确性通过不等于质量全面合格，组件加速也不等于服务收益。保留全部重复轮次、失败、回退和无收益记录；不同实验的收益不能直接相加。

公开记录仅包含已授权的优化摘要、公开源码引用及汇总证据，不上传权重、音频、凭据或私有项目源码。后续 GPU 工作遵守用户当次安排；记录缺项不代表自动启动补测。

结构参考 [sglang-omni 的同名记录分支](https://github.com/EddieTzy/sglang-omni/tree/docs/contribution-log)。这里的实测与 SGLang 记录分开维护。AI assistance：Codex 协助实现、实验和记录整理；不虚构人工审查或听测结论。
