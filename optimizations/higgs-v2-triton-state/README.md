# Higgs Audio v2：Triton 状态更新融合

状态：`measured_tradeoff`。实现并验证，增量服务收益较小且首包 P95 回退，暂不提交第二个 PR。

[总索引](../../README.md) · [指标口径](../../records/performance-methodology.md) · [代码与原始说明](https://github.com/EddieTzy/vllm-omni/tree/perf/higgs-v2-triton-state)

## 为什么优化

在已完成纯音频路由优化后，采样后的 BOS/EOS、delay、末帧和历史拼接仍逐请求执行小算子。目标是减少这段派发、同步和历史拷贝，同时保持采样/RNG 不变。算子数量降低只是机制假设，是否改善服务须独立测量。

## 优化位置与原理

- `higgs_audio_v2_talker.py`：`_dispatch_audio_codes` 的 post-RAS 状态更新调用融合路径；原有当前帧发布保留，CPU/V1 仍走原路径；`on_requests_finished` 释放槽位。
- `integrated_state.py`：`RequestStatePool` 以 request ID 绑定持久 GPU 槽位；`_advance` 每批一次 Triton launch，每个 program 处理一个请求。

融合 leading BOS、trailing EOS、delay/remaining/terminate/frame counters 和全部 8 个 codebook 的 7 帧历史写入。复制环形窗口使最后 7 帧可作为有序 view，末列用于 feedback。独立输出帧不与复用的 ring alias，最终 ramp-down 帧仍输出一次。参考已有 GPU 状态池/整数融合思路，但保留 Higgs 的全部 codebook RAS 语义，不直接照搬 cb0-only 状态。

采样、mask 和 RAS 抽样顺序不变；当前 CPU mask/停止逻辑仍需一次批量 GPU→CPU scalar mirror。因此这版未移除所有同步，也不支持完整 CUDA Graph。实验池容量固定 256，生产版仍须设计 runner 容量和抢占生命周期。保存的是逐字节实测源码，不宣称已满足完整 pre-commit 或上游可合并要求。

## 精确基线与条件

代码在 [路由 perf 分支](https://github.com/EddieTzy/vllm-omni/tree/perf/higgs-v2-audio-routing) 上叠加这一项状态优化。base 已包含第一个 PR，不能把此前 +50.8% 重复计算。

| 项目 | 版本 |
| --- | --- |
| 共同原始 base | `4dafb464442956996be9633ed036cd24ebfef1d0` |
| 已启用路由的实测 source | `30dcfdc205d17072734328e74c5ca96a882ec548` |
| 路由归档 head | `130fbadf725639b467cb262a19e4aedaebffca92` |
| 状态融合归档 head | `06549a28d920f033de1bce3d0a89307151974d51` |

两份新增/修改的模型源码 SHA256 与 B200 实测文件相同，见 [manifest](metrics/manifest.json)。同一张 B200、driver 580.173.02、torch 2.14.0+cu130、vLLM .34.0、FlashInfer .7.3、transformers 5.22.0，模型、codec、数据集 pin 与[路由记录](../higgs-v2-audio-routing/README.md)相同。

相同 eager MRv2 BF16、TP=PP=1，两阶段 max_num_seqs=32、GPU 配额 .6/.25，async_chunk=false；temperature=1、top_k=50、top_p=.95、max_tokens=1024，不发送 request seed。A/B/B2/A2 四次独立启动，每次 32 warmup +64 measured，中英交错，两波各 32。仅 C32，没有追加其他并发档位。先前路由实验还先跑了 native C16，相应随机轨迹/流程不同，不能把两次绝对值拼成同一实验。

## 性能与取舍

| 指标：两轮均值 | 路由基线 | 路由 + 状态融合 | 变化 |
| --- | ---: | ---: | ---: |
| audio秒/秒 | 29.9803 | 30.4724 | +1.64% |
| 平均 cost（秒） | 3.2944 | 3.2282 | -2.01% |
| 平均请求 RTF | .7966 | .7545 | -5.28% |
| 首包 P50（秒） | 3.2708 | 3.2390 | -.97% |
| 首包 P95（秒） | 4.2358 | 4.3387 | +2.43%（回退） |

[全部轮次与汇总](metrics/serving_results.json) 保留 A/B/B2/A2；256 条 measured 请求全部成功，四轮 HTTP stream disconnect 后 WAV 恢复检查通过。恢复成功不等于服务端 abort acknowledgement 或无泄露证明。输出随机且长度变化；两轮范围仅为描述，不能当成正式置信区间。P99、纯 GPU 请求耗时、播放欠载和显存汇总尚未在此记录核实，明确留缺项。

## Base 与优化后 profile

这是**实际 dispatch + 合成 head/logits/codes** 的分阶段同步诊断，并非真实全模型服务 trace。16 步 warmup 后测 32 步，C32，保留同步成本；[phase_profile](metrics/phase_profile.json)。

| median ms | 路由基线 | 加状态融合 |
| --- | ---: | ---: |
| state update | 1.4754 | .1996 |
| mask | 4.9290 | 4.9878 |
| RAS | 1.7078 | 1.7600 |

组件明显变快，但端到端只改善较小。上述合成阶段耗时不等于真实模型各阶段占比；它支持继续调查 mask/RAS 和 host 边界，而非已证明它们是唯一服务瓶颈。完整的前后服务 CPU/CUDA timeline 尚未归档。早期 event launch loop 包含 host 派发间隙，不作纯 kernel 延迟。

## 正确性与质量

初始整数 kernel 998 组 CPU-reference parity。实际 `_dispatch_audio_codes` 在 CUDA 上对比原实现：150 步、4607 次请求状态检查、85 个 request ID，覆盖重排、部分 prefill、非音频、EOS、清理和槽位复用，最终所有槽位释放，历史不超过 7 帧。既有 MRv2 模块 16 passed /17 warnings。

真实模型、相同合批顺序和初始 audio RNG 的 C32 对照：32/32 完整 PCM byte-identical、32/32 codec histories 一致、179/179 规范化采样/RNG 事件一致，[配对结果](metrics/paired_results.json)。这部分诊断不参与性能统计。未做新的全量 ASR/SIM、听测、长时稳定、TP 或 CUDA Graph 资格验证，不宣称一般质量非劣。

## 结论与后续

保留实验，尚未创建第二个 PR。下一步候选是减少 CPU mask/停止逻辑对 scalar mirror 的依赖，并检查 RAS 小算子派发；先保持逐请求抽样顺序、请求生命周期和终帧语义。没有额外测量授权或 GPU 可用性时，只进行代码/证据整理，不自动补测。
