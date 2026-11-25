# 每项优化的详细记录

每项记录动机、代码位置与原理、精确基线、前后 profile、吞吐、RTF、cost、首包、正确性、质量及取舍。[总索引](../README.md) · [指标口径](../records/performance-methodology.md) · [模板](TEMPLATE.md)。

- [Higgs v2 / 纯音频路由](higgs-v2-audio-routing/README.md)：`merged`；PR #11068 已于北京时间 2026-11-25 合并。
- [Higgs v2 / Triton 状态更新融合](higgs-v2-triton-state/README.md)：`measured_tradeoff`；增量吞吐 +1.64%，首包 P95 回退，暂不提第二个 PR。
- [Higgs v2 / 批量反馈 embedding](higgs-v2-packed-feedback/README.md)：`not_adopted`；组件加速没有转化为稳定服务收益，原始资料尚未公开归档。

- [Higgs v2 / BOS/EOS logits mask 融合](higgs-v2-logits-mask/README.md)：`pr_submitted`；PR #11119 开放；RTX PRO 6000，C32 吞吐 +19.44%、平均耗时 -14.10%、RTF 1.133→.996。
- [批量 RAS 重复检测](higgs-v2-ras-repeat/README.md) · 重复检测变快，C32 吞吐下降 6.45%。
- [LM 音频模式 mask](higgs-v2-audio-mode-mask/README.md) · 平均耗时下降 3.21%，吞吐下降 2.69%，两次候选启动的吞吐波动较大。
- [限制每个请求的音频历史](higgs-v2-bounded-history/README.md) · 内部历史在 200 帧时由 12,800 B 降至 448 B。基线有 40.6 秒长音频，吞吐差异无法归因。
- [保留 CPU 请求状态的 delay 更新融合](higgs-v2-delay-state/README.md) · 阶段耗时下降，C32 吞吐下降 11.01%，平均耗时增加 19.02%。
- [Higgs v3 / 流式 codec 校验](higgs-v3-codec-validation/README.md) · C32 校验同步由两次减为一次。吞吐 36.85 → 35.11 音频秒/秒，首包 P95 373 → 402 毫秒。保留实验，暂不提交。
