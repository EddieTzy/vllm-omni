# Higgs Audio v2：BOS/EOS logits mask Triton 融合

状态：`pr_submitted`。2026-11-25 在 RTX PRO 6000 上完成正确性和 C32 独立服务对照，有明确收益信号；[PR #11119](https://github.com/vllm-project/vllm-omni/pull/11119) 已提交，尚未合并。

[总索引](../../README.md) · [指标口径](../../records/performance-methodology.md) · [perf 分支](https://github.com/EddieTzy/vllm-omni/tree/perf/higgs-v2-logits-mask) · [完整 RESULTS](https://github.com/EddieTzy/vllm-omni/blob/075d70559532b27536786e1ebf54b2c96a1d19fb/benchmarks/higgs_delay_mask_experiment/RESULTS.txt)

## 为什么优化、改了哪里

原采样路径逐请求、逐 codebook 执行 leading BOS / trailing EOS logits mask，产生大量小算子。目标是在不改抽样和随机数顺序的前提下减少这段派发开销。

`higgs_audio_v2_talker.py::_dispatch_audio_codes` 的 CUDA 路径从已有 CPU 请求状态收集 delay/remaining；`delay_pattern_mask.py::mask_audio_logits_` 一次复制小型 int32 表并用单次 Triton launch 原位屏蔽非法 logit，每个 program 处理一个请求/codebook。保留 allowed logit 的原值，支持非重叠 stride 布局。CPU/非 CUDA tensor 继续走原路径，sampling、RAS、状态更新和当前帧交付不变。

没有接入旧的状态融合实验，也没有增加公共配置、依赖或修改权重。其他 GPU 平台、multi-GPU、graph 模式未资格验证。

## 精确版本和机器

- base：`649e7b3b9a5b93f9a33f753942df149ba939e523`，已含合并的路由 PR #11068。
- 实测实现提交：`273179aa656aba22a10ff760fbc927b389c857a4`。
- perf 归档 head：`075d70559532b27536786e1ebf54b2c96a1d19fb`。
- RTX PRO 6000 Blackwell Server Edition，97887MiB，SM120，driver 580.173.02。
- torch 2.14.0+cu130、CUDA build 13.0、vLLM .34.0、Triton 3.8.0、FlashInfer .7.3、transformers 5.22.0。Omni 活跃源码为上述 base+patch，公开源码归档缺少生成的版本文件，运行时报告 `dev`。

[manifest](metrics/manifest.json) 固定模型、codec、公开语料版本和源码哈希。[validation](metrics/validation.json) 对比 2398/2399 个 runtime Python 文件：仅 talker 被修改，并新增 mask 模块，没有其他 runtime 文件差异。

## 条件与复现

同一张新卡 A/B/B2/A2 四次独立启动，每次 32 warmup +64 measured，两波各 32；固定中英各 32 条公开目标，不发送 request seed。eager MRv2，BF16 engine，TP=PP=1，async_chunk=false，两阶段 max_num_seqs=32，GPU 配额 .6/.25；temperature=1、top_k=50、top_p=.95、max_tokens=1024。FlashAttention2 由运行时选择。

这是 C32 单点实测，没有补其他并发档位。全部模型与客户端启动参数、公开语料、配置和 benchmark client 见 [固定版复现说明](https://github.com/EddieTzy/vllm-omni/blob/075d70559532b27536786e1ebf54b2c96a1d19fb/benchmarks/higgs_delay_mask_experiment/README.md)。不混入 B200 结果，不累计原路由或状态融合百分比。

## 性能

无 profiler 的两轮 boot 均值；[原始各轮汇总](metrics/serving_results.json)。

| 指标 | 已含路由的 base | 加 mask kernel | 变化 |
| --- | ---: | ---: | ---: |
| audio秒/秒 | 19.6806 | 23.5063 | +19.44% |
| 平均 cost（秒） | 4.8095 | 4.1312 | -14.10% |
| 平均请求 RTF | 1.1333 | .9956 | -12.15% |
| 首包 P50（秒） | 4.7528 | 4.1907 | -11.83% |
| 首包 P95（秒） | 6.7390 | 5.3993 | -19.88% |

base 吞吐两轮为 19.8867 /19.4744，candidate 为 22.0689 /24.9436，完整保留 candidate 的波动。256 measured 请求全部成功。四次 HTTP disconnect 后跟随 WAV 恢复正常，不等于服务端 abort acknowledgement 或无泄露证明。

200ms 显存采样覆盖 warmup/测量和配置池：base 59596/59586MiB，candidate 59640/59594MiB，不宣称节省显存。E2E 含 HTTP/排队/talker/codec，TTFA 为首个非空 PCM 负载，在该部署中接近完成；请求 RTF 不等于吞吐倒数。P99 和播放欠载没有在此统一提取，不填零。

## Profile 与机制证据

已补充[真实服务 C32 CPU/CUDA profile](https://github.com/EddieTzy/vllm-omni/blob/77a2b9b/benchmarks/higgs_delay_mask_experiment/profile_rtx_20261125/README.md)：RTX PRO 6000、相同配置，A/B/B2/A2 四次启动，各 32 warmup +32 诊断请求；128 个 measured 请求全部成功。两轮 CPU wall 中位数取均值，delay-pattern mask **6.2305→0.1313ms（-97.89%）**，MRv2 sample_step **15.9686→9.6047ms（-39.85%）**。稳态基线 trace 有 480 次标量 H2D 拷贝及 480 次流同步；优化版为一次状态表 H2D、一次同步和一个 Triton kernel。全部五个 trace 均保留，基线不同 delay 状态的拷贝/同步次数也有差异。

归因位置为音频 head 之后、采样之前的 logits mask；RAS、状态更新代码不变。事件计时包含满足筛选条件的 warmup 和诊断流量，剔除 profiler 步骤及后两步，保留 A/B/B2/A2 各 97/96/101/107 个纯音频 C32 步骤。诊断周期 44.9565→36.7806ms，forward 等区域也存在运行波动，不能据此把正式端到端 +19.44% 精确拆分成 mask 单项贡献。原正式性能数据没有 profiler。[完整阶段摘要](metrics/profile_results.json)。

以下为此前独立组件证据：独立原型组件测试含 metadata H2D，但不含真实请求状态 lookup、head 和 sampling；[component_results](metrics/component_results.json)。N32 原循环 / PyTorch broadcast / Triton 的中位耗时为 5.681/.224/.040ms；N1 为 .183/.219/.035ms，因此没有仅按广播实现在大 batch 的表现作决定。

1060 组原型状态比较覆盖精度、计数边界、NaN/Inf、非连续布局和空输入，RNG 未变化。组件加速用于机制支持，不能当作真实服务阶段占比或整体加速倍数。

## 正确性、失败记录与取舍

实际集成 kernel 的 13 项 GPU 回归加原 MRv2 16 项：**29 passed /1 warning**。受控同批次和初始 audio RNG 的真实模型 C32：32/32 完整 PCM byte-identical、32/32 codec 历史和 200/200 采样/RNG 事件一致，[paired_results](metrics/paired_results.json)。完整 ASR/SIM、听测、其他硬件及全仓 CI 未执行，不宣称普遍质量非劣。

首轮 correctness baseline 在新卡 JIT/autotune 期间超过继承的 240s 健康检查预算，尚未发送音频请求；保留失败日志，以新 r2 标签和 900s 启动预算完成。没有丢弃计时性能轮次。模型和采样配置未变。

完整自检和最终 rebased 29 项回归测试通过（0.76s），PR #11119 已提交，head `0ee2017a6a59808544f7cacee49cd9d2dfce48e8`，base `a57d6d1a73c694476ae1cc1447bcd765dfd47dd7`；模型源码字节与实测一致，pre-commit、Python 3.11/3.12 构建、CodeQL、DCO 已通过，等待人工 review。原始日志、PCM 和完整逐步采样轨迹保留在隔离实验目录，公开归档只含摘要、hash 和复现材料。GPU 服务测完已停止、显存释放。后续可继续研究 RAS 重复检查，保持每请求重采样顺序，不能擅自改批量抽样语义。

## 评审修正与经验（2026-11-25）

针对 [Sy0307 的评审](https://github.com/vllm-project/vllm-omni/pull/11119#pullrequestreview-6055542476)，已推送 [`b19471884`](https://github.com/EddieTzy/vllm-omni/commit/b19471884f41310fc1429357a5eca95bdcbcb0db)：

- 配置初始化时一次性校验 BOS/EOS 在 `[0, codebook_size)`，非法值明确抛 `ValueError`；补充 6 个越界用例及合法边界、配置 round-trip 检查。
- 新 kernel 限定 NVIDIA CUDA、Triton 可用且 logits 在 CUDA 设备。ROCm/HIP 保留原路径，补充模拟 HIP 张量特征及无 Triton 的回退检查；未做 AMD 实机验证。
- 采纳 masked-store 简化，只写禁止的 logits，删除原有 logits `tl.load`，允许的值留在原处。

修正后的 RTX PRO 6000 定点测试 **38 passed, 1 warning in 1.63s**，本地 pre-commit 检查通过。此前的 29 项测试和完整 PCM 配对、端到端性能数字均属于前一版；本次没有重跑完整服务性能 A/B，也不宣称这次简化带来额外端到端收益。当前修正 head 为 `b19471884`；上节的 `0ee2017` 和当时 CI 状态属于历史记录。

本次遗漏与后续检查项见 [PR 评审经验](../../records/pr-lessons.md)。已在对应 review thread 回复修正内容，留待 reviewer 复查。
