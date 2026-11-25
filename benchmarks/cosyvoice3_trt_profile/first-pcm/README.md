# CosyVoice3 首段 PCM 交付

RTX PRO 6000，TensorRT，无显式 seed，C4。使用已有两次服务计时和一次独立带 profiler 的诊断。脚本按客户端请求 ID 匹配 `CV3BEGIN` 的 offset=0，并在相邻调用之间查找唯一的 `CV3FLOW` 区间。正式轮次各有 16 个首块。首个 PCM 读的时刻用末次非空读、请求 E2E 和首包耗时近似还原，因此带有从末次读到 EOF 的误差。

| 基线 | 开始解码至首 PCM 均值 | 首块 Flow 主机调用均值 |
| --- | ---: | ---: |
| 第一次启动 | 230.9 ms | 36.3 ms |
| 第二次启动 | 220.9 ms | 35.1 ms |

在独立的四请求 Perfetto 里按时间顺序对齐了 19 次 `forward_streaming`、19 个 Flow 和 19 个 HiFT，检查线程时间线顺序与单调时钟相邻调用间距。首块 Flow、HiFT 的 CPU scope 均值分别是 36.8、28.8 ms，后续 15 块分别是 40.5、44.0 ms。CPU scope 包含子调用和等待，不代表 GPU 专属时间。

四个首块中，第一块单独交付给执行器，余下三块由同一个 `execute_context_3(75)` 处理。这三块各自结束 HiFT 到组执行结束还剩 136.9、66.3、0.1 ms；相应的组结束到客户端首 PCM 约 96、96、97 ms。单独执行的那个请求也在组结束约 212 ms 后才读到第一段 PCM。组内串行返回解释了部分等待。组结束后的间隔还包括下游处理、交付和客户端读；现有标记不能进一步归因。诊断轮 4 请求的分段时间独立于上表。

下一处值得打点的是 Stage 1 结果何时可交付、在下游何时被取走，以及 HTTP 何时写出首段。当前数据尚不足以估算提前交付或改声码器的服务收益。

[原始逐请求结果](first-pcm.json) · [分析脚本](analyze.py) · [原实验及 Perfetto](https://github.com/EddieTzy/vllm-omni/tree/b457e2975322a1b3acb8defb06e60c74c4b5c0fa/benchmarks/cosyvoice3_trt_profile)
