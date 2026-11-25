# Higgs Audio v3 流式 codec 输入校验

局部同步开销减少，当前 C32 服务结果不足以支持提交性能 PR。

每个非空解码组先检查音频码范围，再选择 exact-shape CUDA Graph 或 eager 解码。基线分别取最大值和最小值并传回 CPU。候选用 `torch.aminmax` 算出两个边界，写入同一个两元素缓冲区，一次取回。错码处理仍按请求隔离。模型输入、采样和部署配置保持相同。

## 正式服务测量

RTX PRO 6000 Blackwell，单卡，C32。四次新进程启动，顺序 A/B/B2/A2。每轮先预热 32 条，随后测量 64 条固定中英公开文本，两波各 32。共 256 个测量请求，均收到多个非空 HTTP 块。客户端未传 seed，随服务正常采样。每个指标先按轮计算再取两次启动的算术均值。

| 指标 | A | B | B2 | A2 | 基线均值 | 候选均值 | 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 音频秒/墙钟秒 | 36.443 | 35.731 | 34.484 | 37.252 | 36.848 | 35.108 | -4.72% |
| 平均端到端耗时（秒） | 2.481 | 2.600 | 2.501 | 2.351 | 2.416 | 2.550 | +5.56% |
| 平均请求 RTF | 0.589 | 0.617 | 0.586 | 0.557 | 0.573 | 0.602 | +4.95% |
| 首个 PCM P50（毫秒） | 375.799 | 370.858 | 306.020 | 329.550 | 352.675 | 338.439 | -4.04% |
| 首个 PCM P95（毫秒） | 398.526 | 392.903 | 410.424 | 348.325 | 373.426 | 401.664 | +7.56% |

吞吐按生成音频总秒数除以测量墙钟时间。RTF 按每个请求的端到端时间除以该请求音频时长，再求平均。首包取收到首个非空 PCM 块的时间。两次启动的跨度保留在表中，动态合批与采样长度会影响结果。[各轮汇总](serving.json)和[逐请求指标与 PCM 哈希](request-metrics.json)可用于复核。

## 阶段证据

独立诊断各启一个基线和候选进程。客户端负载与正式轮次相同；服务启动后开始记录，含预热。采集第 3、24、80 个 codec forward 的 PyTorch Chrome trace。在 Perfetto 中搜索 `higgs3.codec.validate_codes`，看 CPU 线程里的 CUDA Runtime 调用和对应 GPU 拷贝。每份 trace 各有一次受标注的主机校验，PyTorch 另导出对应的 GPU annotation。

| 指标 | 基线 | 候选 |
| --- | ---: | ---: |
| 每个 trace 中每次校验的 D2H 拷贝 / 流同步 | 2 / 2 | 1 / 1 |
| 非 trace 解码组中的校验主机中位耗时 | 810 µs | 247 µs |
| 非 trace CUDA 解码组 | 160 | 155 |
| 其中命中 codec graph | 121 | 119 |
| 假设同帧数可以合批后的组数 | 160 | 155 |

主机校验时间包括等待同一 stream 上更早提交的工作。两侧的动态批次不同，不能用 563 µs 的差值预测首包改善。固定形状、相同输入的独立 GPU 对照，在五种形状上中位约 37 → 24 µs，见 [control.json](control.json)。第二个假设也核对过：这些批次中放宽左右裁剪量的分组条件没有减少组数，故本次未改合批逻辑。

[六份原始 Perfetto trace、事件和阶段摘要](profile/evidence.tar.gz) · [SHA-256](profile/SHA256SUMS) · [基线统计](profile/baseline.json) · [候选统计](profile/candidate.json)

## 正确性与复现

现有 codec batching 和 graph shapes 回归共 28 项通过（2 条警告）；候选生产文件的适用 pre-commit 检查通过。固定音频码的实际 codec 对照得到 7 条相同 PCM 和 1 个一致隔离的非法请求。

同输入校验覆盖非连续及连续张量、空帧、越界、形状错误，核对异常文本和原张量返回行为。实际权重 codec 的单请求、合批、命中 graph、eager、左右裁剪、异常同行结果见 [PCM 对照](pcm-control.json)。这些是固定码到 PCM 的检查；未做感知质量或多平台认证。

[manifest.json](manifest.json) 记录基线、候选、库版本、权重和源码哈希。服务使用上游原样的 `higgs_multimodal_qwen3_high_throughput.yaml`，开启 async_chunk、codec streaming，stage 0 最大 64、stage 1 最大 16。客户端 C32 表示同时发 32 个请求，实际 codec 组大小见诊断文件。

[复现脚本与固定公开文本](reproduction/) 使用已准备的双阶段 CUDA 环境，目录布局写在 trial.py 中。将基线和候选检出为 source 和 candidate，固定权重置于容器 /data/models/higgs-v3-tts-4b，语料置于 /data/higgs-routing/public-corpus.json。依次运行 trial.py base-a、trial.py cand-b --candidate、trial.py cand-b2 --candidate、trial.py base-a2；阶段复现另用 --profile 和新的 label，每次服务完成后退出。每个模型实验单独占用 GPU。

Codex assisted with implementation, experiments and documentation.
