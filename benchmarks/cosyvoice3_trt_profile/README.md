# CosyVoice3 TensorRT Flow，RTX PRO 6000，流式 C4

2026-11-25，vLLM 0.34.0、torch 2.14.0+cu130、TensorRT 10.14.1.48.post1。复用原容器和模型权重，通过独立目录加载已有 TRT 依赖。通用流式配置、RAS、Stage 0 FlashInfer，DiT 编译关闭。TRT 同时用于 speaker embedding 和 Flow；日志确认实际走到了 flow-decoder estimator。使用本机模型捆绑的 FP32 ONNX，在本机新建 plan；该版本的 builder 对此模型优先设置可用的 FP16 flag，实际 trace 也有 FP16 GEMM。每次启动以单请求检查真实路径，然后预热，再发四波 C4。每波使用相同四条英文文本与参考音频。

## 首包和收尾在哪里

下表是每种流量第一次独立启动的基线。每组各 16 个正式请求，关闭 profiler，只保留少量时间戳。种子隔离由 Stage 0 调度器实现，客户端同时发出四个请求。去掉 seed 后输出长度也会变化，此比较用于说明两类用户流量的行为。

| 客户端流量 | 音频秒/秒 | 平均请求 RTF | 首包 P95 | 平均端到端 | 平均收尾 |
|---|---:|---:|---:|---:|---:|
| 显式 seed=42 | 10.18 | 0.265 | 2.326 s | 1.989 s | 0.179 s |
| 不传 seed | 18.84 | 0.206 | 0.553 s | 1.601 s | 0.746 s |

收尾是每请求最后的 token 块被标记就绪，到同机客户端最后一次读到非空 PCM 的时间。未传 seed 时，79 个块从 talker 交接层产出到进入逐块解码，平均 348 ms，P95 618 ms；其中 73 个开始解码时另有其他请求的块就绪。显式 seed 的 80 个块中此计数为 0。这个窗口包含交接、排队和模型调用前准备。按块的等待均值不能直接与按请求的端到端均值相加。

另起服务采集的无 seed C4 trace 有 19 次 Flow 调用、190 次 estimator 执行。依据 CUDA runtime/driver correlation ID，把实际设备事件关联回所在的 CPU scope。每步 estimator 的设备活动区间并集平均 4.92 ms；同一步首末设备事件跨度平均 5.57 ms。相邻 Euler 步的最后/首个相关设备事件间隔中位数 0.027 ms、P95 0.029 ms（171 个间隔）。这提供了优先排查请求间排队而非步间派发的依据。采集打开了 CPU/CUDA profiler 和模型标记，阶段计时只作归因。

[固定种子指标](perf1.json) · [无种子指标](unseed.json) · [无种子阶段 profile](unseedtrace.json) · [无种子 Perfetto trace](traces/unseeded-stage1.json.gz) · [模型 ONNX 输入维度](onnx-inputs.json) · [引擎指纹和 PCM 检查](validation.json)



在 [Perfetto](https://ui.perfetto.dev/) 点击 **Open trace file**，选择下载的 `.json.gz`。公开版保留全部事件、时间、轨道、算子名及数值关联 ID，裁去了宿主机名、输入值、自由文本参数等字段。文件指纹和逐类事件数量见 [trace-manifest.json](trace-manifest.json)。

[复现步骤和负载客户端](REPRODUCE.md)
