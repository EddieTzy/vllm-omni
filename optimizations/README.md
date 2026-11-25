# 每项优化的详细记录

每项记录动机、代码位置与原理、精确基线、前后 profile、吞吐、RTF、cost、首包、正确性、质量及取舍。[总索引](../README.md) · [指标口径](../records/performance-methodology.md) · [模板](TEMPLATE.md)。

- [Higgs v2 / 纯音频路由](higgs-v2-audio-routing/README.md)：`merged`；PR #11068 已于北京时间 2026-11-25 合并。
- [Higgs v2 / Triton 状态更新融合](higgs-v2-triton-state/README.md)：`measured_tradeoff`；增量吞吐 +1.64%，首包 P95 回退，暂不提第二个 PR。
- [Higgs v2 / 批量反馈 embedding](higgs-v2-packed-feedback/README.md)：`not_adopted`；组件加速没有转化为稳定服务收益，原始资料尚未公开归档。

- [Higgs v2 / BOS/EOS logits mask 融合](higgs-v2-logits-mask/README.md)：`measured_tradeoff`；RTX PRO 6000，C32 吞吐 +19.44%、平均耗时 -14.10%，准备独立贡献，尚未创建 PR。
