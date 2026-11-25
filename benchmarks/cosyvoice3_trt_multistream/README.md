# CosyVoice3 TensorRT 跨请求并发实验

RTX PRO 6000 Blackwell Server Edition 96 GB，vLLM 0.34.0、PyTorch 2.14.0+cu130。通用 CosyVoice3 流式配置、Stage 0 FlashInfer、RAS。4 个英文文本共用一个参考音频。RTF 为逐请求端到端秒数除以生成音频秒数后取平均。C4 为客户端并发数。显式传 seed 时 stage 0 会隔离请求，不传 seed 才允许常规合批。数据仅覆盖本卡和这套语料。



该 ONNX 的六个输入均将 batch 固定为 2，代表同一请求的两条 CFG 分支，时间维动态。直接把两个请求塞进这个 engine 会需要 batch=4。微基准替代方案是每个请求独立 context 和 buffers，以 1/2/4 个 CUDA stream 运行同一个 engine。固定四个请求×10 次 enqueue，热身后交错测量，列出 12 次中位数。时间覆盖 enqueue 和 GPU 完成。输入为合成张量，仅反映 engine 执行，省略 Euler 的数值迭代、服务调度和声码器。

| mel 宽度 | 1 stream | 2 streams | 4 streams |
|---|---:|---:|---:|
| 450 / 450 / 450 / 450 | 125.4 ms | 92.9 ms | 79.0 ms |
| 450 / 510 / 630 / 750 | 154.4 ms | 113.4 ms | 101.0 ms |

[微基准代码](contexts.py) · [原始样本](contexts.json)

在独立服务副本里，为同一 stage-1 调用中已经就绪的请求分配最多 4 条 stream，4 个 TRT context 共享 engine，并保留逐请求 CFG 计算；组内缓存回传推迟至各个 stream 提交后。仅使用现有调度器交付的请求，未增加凑批等待。未传 seed 的四轮独立启动按基线、原型、原型、基线顺序；两组各 32 个请求，共 64 个。每次启动以四波音频总秒数除以四波墙钟总秒数计算吞吐，以下指标取两次启动均值。

| 指标 | 原始 TRT | 多 stream 原型 | 变化 |
|---|---:|---:|---:|
| 吞吐，音频秒/秒 | 18.876 | 19.066 | +1.01% |
| 平均请求 RTF | 0.2066 | 0.2054 | −0.59% |
| 首包 P95 | 0.554 s | 0.559 s | +0.85% |
| 平均 E2E | 1.656 s | 1.662 s | +0.40% |
| 平均收尾 | 0.759 s | 0.740 s | −2.48% |

原型第二次启动的预热期间，16 个多请求语音块用同一组输入及初始 RNG 先执行参考实现再执行原型，16/16 波形、缓存、GPU RNG 状态完全相同。正式输出长度随请求变化。抽样设备显存峰值在末次基线为 45,659 MiB，两次候选均为 46,435 MiB。首包 P95 每次仅 16 个请求，以上为描述性数据。微基准反映同一组请求的连续 enqueue；服务还需完成 talker、每请求的 10 步 Flow、HiFT、缓存和调度。本次无充分的用户端收益支撑上游性能 PR。

[四轮数据、显存和配对检查](service-comparison.json) · [原型的多 stream 调度代码](multi.py) · [数值验证](verify.py)


候选版没有单独抓取 trace。本项所依据的真实服务基线 trace 见 [TRT 路径归档](https://github.com/EddieTzy/vllm-omni/tree/perf/cosyvoice3-trt-profile/benchmarks/cosyvoice3_trt_profile)。

[复现步骤和负载客户端](REPRODUCE.md)

分支内的三处模型文件包含实验算法；归档时去掉逐项计时打印，并将多流调度限制在已加载 TRT estimator 的路径。实测调度函数（包含计时点）保存在 multi.py。独立服务测量使用正常合批、未传 seed。本分支仅用于实验。
