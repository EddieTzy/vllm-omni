# Higgs Audio v2：批量反馈 embedding lookup

状态：`not_adopted`。历史组件/服务初筛，停止单独贡献；没有对应已发布 perf 分支或上游 PR。

[总索引](../../README.md) · [指标口径](../../records/performance-methodology.md)

## 为什么优化、改了哪里

原 feedback helper 逐请求读取末帧并调用音频 embedding；尝试把已有帧打包为一次 lookup，减少小算子和 Python 循环。位置是 `higgs_audio_v2_talker.py` 的音频反馈嵌入 helper；该中间方案仍遍历 Python 请求状态并读取位置，未实现完整 GPU 状态 gather。

## 历史实测与证据范围

独立实验基于已经做过音频路由优化的 Higgs v2、同一 B200 及公开 pinned 权重。实际 BF16 embedding 权重、合成请求历史的 8 组控制 bitwise 相同；N32 all-audio 组件中位耗时约 1.3527→.2664ms。22 项单元/控制测试通过，不能据此推导服务收益。

随后 A/B/B2/A2 独立启动服务初筛覆盖历史 C1/C16/C32：C32 音频吞吐均值约 -2.45%，C1 约 -1.15%，native C16 未见超出波动的一致收益。此处只是已完成的历史记录，本次没有追加测试。

这些数字来源于当时本地实验汇总。完整 per-arm 性能 JSON、原始 trace 和公共源码快照尚未在此分支重新核实/发布，不能把这份摘要称为完整可复现实验包。逐请求 cost/RTF/首包、显存和质量明细保持未核实，不借用路由或状态融合的指标补齐。GPU 已关闭，不为补记录启动机器。

## 决定与后续

停止 packed-lookup 单独 PR；组件改善没有转化成稳定服务收益。保留作为请求 ID 对齐和后续 GPU feedback gather 的参考。先建立原始资料与源码的可核验归档，再决定是否建立单独 perf 分支；不能因目录存在就标成代码已发布。
