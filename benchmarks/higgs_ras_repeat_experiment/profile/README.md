# 批量 RAS 重复检测的阶段计时

RTX PRO 6000，真实服务 C32。A 与 B 各独立启动一次，每次 32 条预热、32 条诊断请求。统计保留预热及诊断流量中符合条件的 32 行纯音频步，排除最初 16 个符合条件的步、采集 trace 的步及随后两步。

| 阶段 | 基线中位数 ms（样本数） | 候选中位数 ms（样本数） |
| --- | ---: | ---: |
| audio.ras | 2.615 (91) | 0.496 (89) |
| talker.sample_step | 9.018 (91) | 7.480 (89) |
| Decode cycle | 35.086 (88) | 34.221 (86) |

阶段计时取 CPU wall time，包含主机发射与等待。两轮的采样与合批工作量可能不同。decode cycle 为连续合格步的间隔。无 profiler 的端到端对照见上一级 README。

summary.json 含各阶段 CPU 与 CUDA 事件计时和 trace 操作计数。evidence.tar.gz 保存事件及 Chrome trace，bundle.json 记录 SHA256。manifest.json 记录源码与插桩哈希。run_profile.py 和 profile_support.py 是实验采集脚本，复现时需要匹配脚本中的实验目录和启动器。summarize.py 汇总已采集的事件。
