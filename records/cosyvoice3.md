# CosyVoice3 实验记录及 Perfetto 下载

RTX PRO 6000 Blackwell，流式 C4。点击下面的 trace 下载到本地，然后在 [Perfetto](https://ui.perfetto.dev/) 选择 **Open trace file**。链接固定到本轮归档提交。

## eager Flow 热点

带 profiler 的 C4 路径诊断，确认每次 RoPE 平均 24 个设备操作。 [独立分支归档](https://github.com/EddieTzy/vllm-omni/tree/0ab458950d9952f49bb803bbb8633102089a93cb/benchmarks/cosyvoice3_eager_profile)

- [stage1.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/0ab458950d9952f49bb803bbb8633102089a93cb/benchmarks/cosyvoice3_eager_profile/traces/stage1.json.gz)
- [stage0.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/0ab458950d9952f49bb803bbb8633102089a93cb/benchmarks/cosyvoice3_eager_profile/traces/stage0.json.gz)

## 已有 DiT 编译开关

显式 seed C4，吞吐 5.20→12.10；相同输入的音频块存在可测差异。 [独立分支归档](https://github.com/EddieTzy/vllm-omni/tree/d2816876f001e882ce8dcdab45ffb69437c55fcc/benchmarks/cosyvoice3_dit_compile)

- [1-eager.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/d2816876f001e882ce8dcdab45ffb69437c55fcc/benchmarks/cosyvoice3_dit_compile/traces/1-eager.json.gz)
- [2-eager.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/d2816876f001e882ce8dcdab45ffb69437c55fcc/benchmarks/cosyvoice3_dit_compile/traces/2-eager.json.gz)
- [1-compiled.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/d2816876f001e882ce8dcdab45ffb69437c55fcc/benchmarks/cosyvoice3_dit_compile/traces/1-compiled.json.gz)
- [2-compiled.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/d2816876f001e882ce8dcdab45ffb69437c55fcc/benchmarks/cosyvoice3_dit_compile/traces/2-compiled.json.gz)

## TRT 基线与排队

无 seed C4，首包 P95 0.553 秒，79 个语音块的平均等待为 348 ms。 [独立分支归档](https://github.com/EddieTzy/vllm-omni/tree/b457e2975322a1b3acb8defb06e60c74c4b5c0fa/benchmarks/cosyvoice3_trt_profile)

- [seeded-stage0.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/b457e2975322a1b3acb8defb06e60c74c4b5c0fa/benchmarks/cosyvoice3_trt_profile/traces/seeded-stage0.json.gz)
- [seeded-stage1.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/b457e2975322a1b3acb8defb06e60c74c4b5c0fa/benchmarks/cosyvoice3_trt_profile/traces/seeded-stage1.json.gz)
- [unseeded-stage1.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/b457e2975322a1b3acb8defb06e60c74c4b5c0fa/benchmarks/cosyvoice3_trt_profile/traces/unseeded-stage1.json.gz)
- [unseeded-stage0.json.gz](https://raw.githubusercontent.com/EddieTzy/vllm-omni/b457e2975322a1b3acb8defb06e60c74c4b5c0fa/benchmarks/cosyvoice3_trt_profile/traces/unseeded-stage0.json.gz)

在既有无 seed C4 时间线中，首块开始解码到客户端收到 PCM 的两次均值为 231 ms 和 221 ms。独立诊断里，同组三个首块完成 HiFT 后到该组执行结束的等待分别约为 137、66、0 ms。普通 C4 的四个首块均在组结束 0.36–0.51 ms 内完成设备回传，回传结束至客户端首读仍有约 95–211 ms。[离线拆分、原始计时和复现脚本](https://github.com/EddieTzy/vllm-omni/tree/0acba25a58e3c224b9577f657b672de552434aef/benchmarks/cosyvoice3_trt_profile/first-pcm)

后续两波普通采样 C4 采集了 8 个请求。Stage 1 提交首块拷贝后，先运行下一组音频计算，再取上一组结果。期间经过 78.6–195.2 ms，其中下一组模型执行占 92.9%–98.0%。8/8 取结果时拷贝均已就绪。从组织好主机音频到客户端首读为 5.9–9.2 ms。尚未运行调度改动的性能对照。[逐请求数据及复现](https://github.com/EddieTzy/vllm-omni/tree/51420dec3593988903084ee02663ffe8e033784b/benchmarks/cosyvoice3_trt_profile/first-pcm/request-delivery) · [下载交付时间线](https://raw.githubusercontent.com/EddieTzy/vllm-omni/51420dec3593988903084ee02663ffe8e033784b/benchmarks/cosyvoice3_trt_profile/first-pcm/request-delivery/delivery.perfetto.json.gz)

## TRT 多 stream 原型

无 seed C4，64 请求对照，吞吐 +1.01%，首包 P95 +0.85%，实验留档。 [独立分支归档](https://github.com/EddieTzy/vllm-omni/tree/bbc194bbcfff516fba3445f6e33a5668867919e6/benchmarks/cosyvoice3_trt_multistream)

候选服务以无 profiler 的四轮对照为准，保留了实现、16 个相同输入的状态/音频块检查及微基准。该项使用上一节的 TRT 基线诊断。

## Stage 1 调度与首包

普通流式 C4，96 个请求。Stage 1 关闭异步调度后，首包 P95 从 573.2 降至 497.9 ms，音频吞吐 18.58 → 18.59 秒/秒。第一块到第二块间隔 P95 从 344.0 升至 500.8 ms。 [逐启动指标、数据与复现](https://github.com/EddieTzy/vllm-omni/tree/e7e3b11ad2c4ab91215a5fb452e58ac166265f9c/benchmarks/cosyvoice3_stage1_scheduling)

[A1 时间线](https://raw.githubusercontent.com/EddieTzy/vllm-omni/e7e3b11ad2c4ab91215a5fb452e58ac166265f9c/benchmarks/cosyvoice3_stage1_scheduling/A1.perfetto.json.gz) · [B1 时间线](https://raw.githubusercontent.com/EddieTzy/vllm-omni/e7e3b11ad2c4ab91215a5fb452e58ac166265f9c/benchmarks/cosyvoice3_stage1_scheduling/B1.perfetto.json.gz) · [B2 时间线](https://raw.githubusercontent.com/EddieTzy/vllm-omni/e7e3b11ad2c4ab91215a5fb452e58ac166265f9c/benchmarks/cosyvoice3_stage1_scheduling/B2.perfetto.json.gz) · [A2 时间线](https://raw.githubusercontent.com/EddieTzy/vllm-omni/e7e3b11ad2c4ab91215a5fb452e58ac166265f9c/benchmarks/cosyvoice3_stage1_scheduling/A2.perfetto.json.gz)
