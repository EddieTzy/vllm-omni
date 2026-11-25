# Higgs v2：将内部音频历史限制为最近七帧

分支 `perf/higgs-v2-history-buffer`；代码提交 `e62a057054e7b2ad386bbf65325a418c2b1a8452`。以内部历史工作区有界为主要价值；服务吞吐是否改善按下表如实记录，不能把张量有效载荷下降说成整卡显存下降。

## 改动与基线

代码审计确认所有生产消费者只需要最近七帧（RAS）或最后一帧（反馈）。保留有独立存储的最近七帧，逐步交给 codec 的帧全部保留；请求清理沿用既有 hook。

四个新候选独立从 mask PR #11119 的 `0ee2017a6a59808544f7cacee49cd9d2dfce48e8` 出发，**不彼此叠加**。
基线已经有路由 PR #11068 和 audio delay-pattern logits mask。GPU运行源码
为649e加与#11119相同的模型文件；PR rebased基线a57的额外变更仅涉及WebSocket
ping/pong。source-validation.json记录完整runtime Python清单的比较结果及差异文件哈希。
最新上游95513b5的Higgs文件与a57一致；本表没有冒充在新runtime main上重测。

## 条件

单张RTX PRO 6000 Blackwell Server Edition，97887MiB，SM120，driver580.173.02。
Torch2.14.0+cu130、CUDA build13.0、vLLM0.34.0、Triton3.8.0、FlashInfer0.7.3、
transformers5.22.0，详见manifest.json的模型和codec版本。
BF16、eager MRv2、TP=PP=1、max_num_seqs=32、GPU配额.6/.25，
async_scheduling=false、async_chunk=false、无prefix caching。
同一公开语料，中英各32条，固定两波各32，HTTP请求不传seed；
temperature1、top_k50、top_p.95、max_tokens1024。
A/B/B2/A2四次独立启动，每次32warmup+64measured；256个measured全部成功。
仅C32；未补其他服务并发。四次disconnect/WAV恢复探针通过，不等于服务端
abort确认或无泄露证明。

## 无profiler的服务数据

以下是两次boot指标的算术均值。完整各轮数据见perf-result.json。
原脚本JSON中的`means.fused`字段泛指本项candidate，不意味着所有候选都有融合kernel。
200ms采样的各轮GPU显存峰值见source-validation.json；它覆盖预热和配置池，
不用于证明内部历史buffer之外的显存节省。

| 指标 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 吞吐（audio秒/秒） | 14.031235 | 22.684573 | +61.67% |
| 平均E2E（秒） | 4.284859 | 3.951069 | -7.79% |
| 平均请求RTF | 0.936235 | 0.935062 | -0.13% |
| 首个PCM P50（秒） | 3.940276 | 3.932419 | -0.20% |
| 首个PCM P95（秒） | 5.456633 | 5.518847 | +1.14% |

各轮吞吐与RTF：

- A: 10.441507 audio秒/秒，RTF 0.923247，平均E2E 4.524243秒。
- B: 22.487917 audio秒/秒，RTF 0.909953，平均E2E 3.950228秒。
- B2: 22.881228 audio秒/秒，RTF 0.960172，平均E2E 3.951910秒。
- A2: 17.620962 audio秒/秒，RTF 0.949222，平均E2E 4.045475秒。

吞吐=实际生成audio秒/完整测量wall time；RTF=每请求E2E/audio时长再平均，
不能由吞吐倒算。E2E包含HTTP、排队、talker和codec；首包为首个非空PCM，
在此async_chunk=false部署中接近完成。保留不同输出时长和批处理产生的波动，
两轮结果不是置信区间。不挑最好一轮、不删除长尾，不把这项百分比加到旧PR上。

## 正确性与阶段归因

32项回归，包含6/7/200帧窗口边界与清理验证。受控相同admission、行顺序与首个audio RNG的真实模型：
32/32完整PCM、32/32逐帧历史、全部200个normalized采样/RNG事件一致。
这不等于通用ASR、speaker similarity或感知质量非劣证明。
[阶段 profile、样本数与原始 trace](profile/README.md)。
事件计时包含launch gap；不同state/history/batching下未改动阶段也可能波动。

## 复现与限制

reproduction/c32.yaml、public-corpus.json及scripts/benchmark_client.py来自相同的
固定公开语料/客户端。需要同版本runtime和已下载的上述generation/codec权重。
把源码候选挂载到/opt/vllm-omni，runtime根挂载到/data，使用相同YAML启动
vllm serve /data/models/higgs-v2 --omni --deploy-config /data/reproduction/c32.yaml
--host 127.0.0.1 --port 18874 --trust-remote-code。
模型与codec路径需与YAML一致；固定服务port后运行：

    python scripts/benchmark_client.py --port 18874 --label independent-C32 \
      --concurrency 32 --count 64 --corpus reproduction/public-corpus.json \
      --omit-request-seed --request-timeout 360

每轮换新label并重启服务，顺序A/B/B2/A2。正确性诊断的canonical gate、初始RNG
控制和逐帧打印独立于正式benchmark，不能把其计时当作服务性能。
源码与窄回归测试保存在本perf分支；没有增加公共配置或依赖。
其他硬件、多GPU、CUDA graph、全仓硬件矩阵未测。
公开归档不包含凭据、生成音频或服务日志；原始日志/PCM保留在隔离实验目录。
AI assistance: Codex assisted with code, validation, measurements and writing.

## 内部历史工作区

仅audio_out_ids张量：原来每请求8×T个int64，随生成帧数T线性增长；
现在最多8×7个int64，即448字节有效载荷。200帧时由12800字节降至448字节；
这不包含allocator rounding、KV cache、配置池和完整codec输出。测试同时检查
底层storage有界，避免只裁切view却保留原大buffer。不是总GPU显存节省数字。

### 长尾对吞吐的影响

基线A的两个测量波次最长E2E为5.556/26.541秒，最长生成音频40.6秒；
基线A2最长音频14.88秒。候选B/B2最长音频8.64/7.04秒。
因此表中的+61.67%是这些未配对随机输出下的描述性结果，不能归因为有界历史
复制的确定性吞吐收益，也不能据此声称修复了质量问题。全部长尾请求保留。
精确的逐轮无文本诊断见source-validation.json的request_tail_diagnostics。

独立 main 基线分支 029ab59 的有界历史及 MRv2 回归共 19 项通过。
