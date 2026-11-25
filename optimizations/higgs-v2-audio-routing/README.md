# Higgs Audio v2：纯音频路由

状态：`merged`。已核实 [PR #11068](https://github.com/vllm-project/vllm-omni/pull/11068) 于北京时间 2026-11-25 01:58:56 合并进入上游 `main`，合并提交 [`14a01235186941ad7a26ce258a4ca27c45cb6ebf`](https://github.com/vllm-project/vllm-omni/commit/14a01235186941ad7a26ce258a4ca27c45cb6ebf)。

[总索引](../../README.md) · [指标口径](../../records/performance-methodology.md) · [实现与实测记录](https://github.com/EddieTzy/vllm-omni/tree/perf/higgs-v2-audio-routing)

## 为什么优化

Higgs v2 TTS 需要同时降低响应延迟和提高并发吞吐。原路径在纯音频批次中仍逐层判断 mask 并执行布尔索引；28 层的 norm/MLP 重复这项路由。该实现事实形成瓶颈假设，真实 profile 的支持范围见下文。

## 优化位置与原理

`vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_v2_talker.py` 的 talker `forward` 在进入层循环前判断一次全音频条件，复用 `_routed_norm` / `_routed_mlp` 的既有 `None` 路径。新增三行生产代码；text、BOS、混合 token 继续保留 mask。原实现已经跳过不存在的 text FFN 行，此次减少的是判断、索引和同步开销。仍有一次标量同步，不增加 CUDA Graph 支持。

参考 SonicMind 的音频专用路径思路，适配的是 vLLM-Omni 既有路由语义；不将其他项目中的速度提升归给此改动。

## 状态与版本

| 项目 | 固定版本 |
| --- | --- |
| 实测 base | `4dafb464442956996be9633ed036cd24ebfef1d0` |
| 实测路由提交 | `30dcfdc205d17072734328e74c5ca96a882ec548` |
| perf 归档 head | `130fbadf725639b467cb262a19e4aedaebffca92` |
| 提交时 PR head | `c70831d97aea6a1ad038706b2b58af6609cefb26`，rebase 后 talker 字节相同 |
| profile 证据 head | `30aa5bc7d9007cbddbb9ca0cfbd7680526970f49` |

源码 SHA256、公开模型/codec/数据集版本：[manifest](metrics/manifest.json)。GPU 为一张 B200（183359 MiB），driver 580.173.02，torch 2.14.0+cu130、vLLM 0.34.0、FlashInfer 0.7.3、transformers 5.22.0。

## 实验条件与复现

两阶段 eager MRv2、BF16、TP=PP=1、无量化，`async_chunk=false`，两阶段 `max_num_seqs=32`、显存配额 .6/.25，关闭 prefix cache 和 async scheduling。音频采样 temperature=1、top_k=50、top_p=.95、max_tokens=1024；客户端不发送 request seed，允许真实合批。

公开 SeedTTS 语料固定选择 EN/ZH 各 32 条。A/B/A2/B2 独立启动，保留两轮：native C16 每语言 16 warmup +32 measured，完成即补入请求；raw C32 为 32 warmup +64 measured，两波各 32 条。两者协议不同，不能视作同一扩展性曲线。具体设置及既有 CLI 见 [固定版 RESULTS](https://github.com/EddieTzy/vllm-omni/blob/130fbadf725639b467cb262a19e4aedaebffca92/benchmarks/higgs_audio_routing_experiment/RESULTS.txt) 和 PR 正文。仅整理历史数据，不新增并发测量。

## Base 与优化后 profile

另有四次独立诊断启动和八份真实 32 行纯音频 forward trace。筛选规则、样本数、trace 与非 trace 分离、复现资料见 [profile 说明](profiles/README.md)。平均两轮 forward host 中位数约 34.943 →17.227ms，逐步 decode cycle 中位数约 51.9 →34.4ms；outside-forward remainder 保持约 17ms，不能称为纯 sampler 耗时。

每个被采样的全音频 forward 中，`aten::any` 168→0、`aten::item` 168→1、`aten::index` 84→0，CUDA runtime stream synchronize 339→4。norm/MLP 调用数保持 56/28。该证据支持 talker 路由开销减少，不证明所有请求的精确耗时分解。CUDA event interval 含 CPU launch 间隙和 stream 等待，不等于 GPU kernel 时间总和。

## 性能：吞吐、RTF、cost 与首包

以下为无 profiler 的两轮均值；RTF 是逐请求比值的均值，cost 为客户端 HTTP send→完整音频结束，首包为首个非空音频负载。该部署首包接近请求完成。

| 负载 | audio秒/秒：base→优化 | 平均 RTF | 首包 P50（秒） | 平均 cost（秒） |
| --- | --- | --- | --- | --- |
| C16 EN | 11.918→18.214 | 1.095→.703 | 4.164→2.424 | 4.339→2.617 |
| C16 ZH | 12.168→19.026 | 1.064→.677 | 4.682→2.983 | 4.845→3.060 |
| C32 EN/ZH | 18.017→27.169（+50.80%） | 1.208→.789 | 5.008→3.367 | 5.049→3.333 |

[C16 各轮延迟/RTF](metrics/native_c16_metrics.json) 保留原始 native benchmark 的汇总字段与源文件指纹。C32 首包 P95：6.742→4.509 秒。两轮、成功数、显存及原始汇总：[serving](metrics/serving_results.json)；逐轮 C32 延迟/RTF：[latency](metrics/c32_latency_metrics.json)。采样显存峰值 base 110990/111012 MiB，优化 110970/110966 MiB，含配置池和 warmup，不宣称节省显存。P99、播放欠载和纯 GPU 推理耗时尚未统一归档，不填零或倒算。

## 正确性与质量

MRv2 回归模块 16 passed /17 warnings；改动文件适用 pre-commit 全部通过。真实 BF16 激活的 32424 次 norm/MLP 严格比较全部一致。受控 C32 A/B/A2 的 32/32 完整 PCM、32/32 codec 历史和 179/179 规范化采样/RNG 事件一致：[输出配对记录](metrics/controlled_output_equivalence.json)。诊断开销不计性能。

原始未配对随机样本 EN WER 为 12/712→11/712，ZH CER 为 7/1176→13/1176，全部保留，不当成配对质量回归结论。固定顺序重复 ASR 没有改变归一化分数；冻结 logits/RNG 的行重排试验显示随机抽样与请求行绑定。固定合批/RNG 后 PCM 完全一致，CER 三组均为 24/588，包含高错误率样本，不能解释为绝对质量提高。原始四轮缺少完整逐步 RNG 轨迹，无法逐条重建额外错误的随机历史。未建立一般听感、音色相似度或质量非劣保证。

## 结论与后续

指定 B200 eager MRv2 工作负载中有明显收益，窄范围 PR 已合并；full repository CI、其他硬件、TP/multi-GPU、CUDA Graph 和一般质量结论分别保留边界。后续状态融合应以此优化版作为增量基线，不能重复计算 +50.8%。

## RTX PRO 6000 补充实测（2026-11-25）

这是用户要求查看旧优化在新卡上的表现，单独归档，不重提旧 PR。已在 [PR #11068 描述末尾](https://github.com/vllm-project/vllm-omni/pull/11068) 增加补充结果，原 B200 数据保留。

一张 RTX PRO 6000 Blackwell Server Edition，97887MiB，SM120，driver 580.173.02；torch 2.14.0+cu130、vLLM .34.0、FlashInfer .7.3、transformers 5.22.0。双方都用 `649e7b3b9a5b93f9a33f753942df149ba939e523` runtime，A/A2 仅撤掉上次 3 行路由，B/B2 保留。核对双方各 2398 个 runtime Python 文件，只差 talker，没有 logits mask 或状态融合。

同配置 C32、A/B/B2/A2 独立启动，每轮 32 warmup +64 measured，两波各 32，公开 EN/ZH 各 32 条，不发送 speech request seed，无 profiler。eager MRv2、BF16 engine、TP=PP=1、两阶段 max_num_seqs32/GPU 配额 .6/.25、采样默认值保持一致。

| 两轮 boot 均值 | 原始路由 | 已合并路由 | 变化 |
| --- | ---: | ---: | ---: |
| audio秒/秒 | 10.5966 | 19.6801 | +85.72% |
| 平均 cost / E2E（秒） | 7.8102 | 4.8206 | -38.28% |
| **平均请求 RTF** | **1.8445** | **1.1502** | **-37.65%** |
| 首包 P50（秒） | 7.5530 | 4.8841 | -35.34% |
| 首包 P95（秒） | 11.4930 | 6.2978 | -45.20% |

RTF 先逐请求计算 elapsed/audio_seconds，再取平均，不是总吞吐的倒数。各轮 mean RTF：原始 1.7451/1.9440，优化 1.1526/1.1477；各轮吞吐：原始 11.1379/10.0553，优化 20.6074/18.7528，全部保留。256 measured 请求全部成功，四轮 HTTP disconnect 后 WAV 恢复正常，不声明完整 abort/leak 资格。

[全部四轮与均值](metrics/rtx_pro_6000_20261125/summary.json) · [源码身份](metrics/rtx_pro_6000_20261125/source-identity.json) · [文件比较/显存/恢复](metrics/rtx_pro_6000_20261125/validation.json) · [固定版独立报告](https://github.com/EddieTzy/vllm-omni/blob/655a515e4061ae7cd4893b048ba28ddf4ed74f4b/benchmarks/higgs_audio_routing_experiment/rtx_pro_6000_20261125/RESULTS.txt)。峰值显存原始 59598/59626MiB，优化 59568/59578MiB，含 warmup 和配置池，不宣称显存收益。

仅针对本机本轮 C32 条件，未额外补其他并发点，未新增质量评测。随机长度和波次长尾有波动，两轮不是正式置信区间。与 B200 历史实验的硬件、runtime、先跑 native C16 的预热/负载序列不同，不把 +85.72% 当成硬件的普遍优势；也不与新 mask PR #11119 的增量 +19.44% 直接相加或相乘作实测累计收益。
