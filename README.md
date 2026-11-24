# EddieTzy vLLM-Omni 推理优化与贡献记录

按状态查看进展，再按模型和优化位置查看实现、实验与取舍。此 `docs/contribution-log` 分支用于记录；实现保存在各自的 perf 分支，上游 PR 保持独立。[项目原始 README](https://github.com/vllm-project/vllm-omni/blob/main/README.md)。

最后整理：2026-11-24（北京时间）。本次仅核对 GitHub 状态和既有资料，没有连接 GPU、重新测试或补测并发档位。

最新进展：Higgs v2 音频路由已提交 [PR #11068](https://github.com/vllm-project/vllm-omni/pull/11068)，仍开放、未合并；Triton 状态融合已完成实际采样路径接入和配对验证，但增量服务收益较小，暂不提交第二个 PR。批量反馈嵌入实验未形成稳定收益，保留不采用记录。

## 状态总览

| 分类 | 数量 | 归类规则 |
| --- | ---: | --- |
| 已合并 | 0 | 已核实进入上游 |
| [已提 PR](#已提-pr) | 1 | 已创建且仍开放，不能记成已合并 |
| 进行中或待验证 | 0 | 尚未登记为独立实现的新方向不计数 |
| [已测有收益或取舍](#已测有收益或取舍) | 1 | 已有实测信号，同时明确回退和限制 |
| [无稳定收益或暂不采用](#无稳定收益或暂不采用) | 1 | 保留负面结果和停止原因 |
| 已关闭或归档 | 0 | 已关闭 PR 或被替代的历史版本 |

## 已提 PR

| 模型 | 改动 | 已观测结果与范围 | 入口 |
| --- | --- | --- | --- |
| Higgs Audio v2 | 统一纯音频 norm/MLP 路由 | B200、指定 eager MRv2 配置，C32 吞吐 +50.8%；受控 32/32 完整音频相同。带 profiler 的证据单独归档。 | [PR #11068](https://github.com/vllm-project/vllm-omni/pull/11068) · [perf 分支](https://github.com/EddieTzy/vllm-omni/tree/perf/higgs-v2-audio-routing) · [详情](optimizations/higgs-v2-audio-routing/README.md) |

## 已测有收益或取舍

| 模型 | 改动 | 已观测结果与决定 | 入口 |
| --- | --- | --- | --- |
| Higgs Audio v2 | Triton delay/EOS/历史状态融合 | 基线已含音频路由优化；C32 吞吐额外 +1.64%，平均耗时 -2.01%，首包 P95 +2.43%。增益小且存在尾延迟回退，作为实验保留，尚未提 PR。 | [perf 分支](https://github.com/EddieTzy/vllm-omni/tree/perf/higgs-v2-triton-state) · [详情](optimizations/higgs-v2-triton-state/README.md) |

## 无稳定收益或暂不采用

| 模型 | 改动 | 已观测结果与决定 | 入口 |
| --- | --- | --- | --- |
| Higgs Audio v2 | 批量反馈 embedding lookup | 组件变快，但历史服务初筛 C32 吞吐约 -2.45%，C1 约 -1.15%，C16 无一致改善；停止单独 PR。代码和完整原始材料尚未发布为 perf 分支。 | [详情与证据限制](optimizations/higgs-v2-packed-feedback/README.md) |

## 详细记录

- [每项优化目录](optimizations/README.md)：动机、代码位置、原理、基线、profile、性能、正确性和取舍。
- [记录模板](optimizations/TEMPLATE.md) · [RTF、cost、首包与 profile 口径](records/performance-methodology.md)。
- [Higgs v2 模型记录](records/higgs-audio-v2.md) · [结构化索引](records/entries.json)。
- [路由 profiling 证据分支](https://github.com/EddieTzy/vllm-omni/tree/higgs-routing-profiling-evidence)：八份真实 CPU/CUDA trace、阶段统计和复现资料。它不替代原始无 profiler 性能对照。

## 维护规则

一项优化一个独立实现分支和详细目录，记录精确 base/head、叠加依赖、硬件、权重、软件、请求集、预热与重复顺序。没有数据的字段标记未测，不填零，不由吞吐倒算请求 RTF 或耗时。

性能分支、已提交 PR、已合并 PR 是不同状态；正确性通过不等于质量全面合格，组件加速也不等于服务收益。保留全部重复轮次、失败、回退和无收益记录；不同实验的收益不能直接相加。

公开记录仅包含已授权的优化摘要、公开源码引用及汇总证据，不上传权重、音频、凭据或私有项目源码。后续 GPU 工作遵守用户当次安排；记录缺项不代表自动启动补测。

结构参考 [sglang-omni 的同名记录分支](https://github.com/EddieTzy/sglang-omni/tree/docs/contribution-log)。这里的实测与 SGLang 记录分开维护。AI assistance：Codex 协助实现、实验和记录整理；不虚构人工审查或听测结论。
