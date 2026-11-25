# Fish S2-Pro 首次完整 prefill embedding

RTX PRO 6000，C32 吞吐 23.47 → 23.34 音频秒/秒（−0.56%），首包 P95 增加 3.50%。三组 trace 均消除了完整 prefill 的双向拷贝，保留实验。

| 指标 | A | B | B2 | A2 | 基线均值 | 候选均值 | 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 音频秒/秒 | 23.396 | 22.799 | 23.875 | 23.539 | 23.468 | 23.337 | -0.56% |
| 首 PCM P50（ms） | 1670.770 | 1691.554 | 1787.420 | 1691.870 | 1681.320 | 1739.487 | +3.46% |
| 首 PCM P95（ms） | 1750.973 | 1743.953 | 1915.841 | 1784.903 | 1767.938 | 1829.897 | +3.50% |
| 平均请求耗时（秒） | 3.770 | 3.783 | 3.824 | 3.778 | 3.774 | 3.804 | +0.79% |
| 平均请求 RTF | 1.100 | 1.106 | 1.112 | 1.109 | 1.104 | 1.109 | +0.48% |

两端基于 Fish SlowAR 的 MRv2 迁移提交 `95988e77`（上游 #9806），候选代码为 `b4336014`。首次 prefill 正好用完整个 prompt 时，直接复用 GPU embedding。分块路径继续使用原有的 pinned CPU 缓存。

每轮预热 32 个请求，再按 C32 分两批测 64 个流式克隆请求。独立诊断两端各命中 96/96 次，三组 trace 均显示每个请求少了一次约 1.3 MB 的 GPU→CPU 拷贝和同量回传。该段主机耗时中位数由 664.5 µs 降至 7.1 µs。

按 embedding 哈希、形状和长度比对，两批正式输入各有 32/32 个匹配，预热匹配 30/32。确定性 CUDA 对照通过 8/8，Fish 回归测试通过 16/16。尚无听测或 CER 结果。

[独立实验、正确性与限制](https://github.com/EddieTzy/vllm-omni/tree/057079b9654f37dedc90967cee470b76cea31c93/benchmarks/fish_s2_prefill_embeddings) · [Perfetto 时间线](https://raw.githubusercontent.com/EddieTzy/vllm-omni/057079b9654f37dedc90967cee470b76cea31c93/benchmarks/fish_s2_prefill_embeddings/evidence.tar.gz)
