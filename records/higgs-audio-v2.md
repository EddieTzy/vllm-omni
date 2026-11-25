# Higgs Audio v2 优化记录

最后整理：2026-11-25（北京时间）。本页为既有实验索引，本次没有运行 GPU。

| 项目 | 增量基线 | 当前结论 | 详细记录 |
| --- | --- | --- | --- |
| 纯音频路由 | 原始 masked norm/MLP 路径 | C32 指定工作负载吞吐 +50.8%；PR #11068 已合并（2026-11-25 北京时间，`14a01235`） | [详情](../optimizations/higgs-v2-audio-routing/README.md) |
| 批量反馈 embedding | 已优化音频路由 | 组件加速但服务无稳定收益，停止单独 PR；未发布源码分支 | [详情](../optimizations/higgs-v2-packed-feedback/README.md) |
| Triton 状态更新融合 | 已优化音频路由 | C32 增量吞吐 +1.64%，首包 P95 +2.43%；保留实验 | [详情](../optimizations/higgs-v2-triton-state/README.md) |

路由 profiling 是路由优化的独立证据附件，不是第四项优化。真实 32 行 all-audio forward 的 route/index/sync 操作减少，支持路由机制解释；forward 外的 interval 不能标成纯 sampler。

三项实验各保留基线与适用条件；packed feedback 没有混入 state fusion，状态融合也不重复累计路由的收益。带 profiler 的数据与无 profiler 服务测量分开，动态合批和未配对随机输出的差异不能冒充算术质量回归。

后续代码方向：mask/停止逻辑对 GPU 状态的消费、RAS 小算子和 CPU 回传。尚未有独立实现和实测的方向不计入“进行中优化”的数量，也不自动启动额外并发测量。
