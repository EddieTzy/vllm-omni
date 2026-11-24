#!/usr/bin/env bash
set -euo pipefail
root=/home/tzy/zyb-fish-pr
name=${HIGGS_CONTAINER_NAME:-zyb-higgs-v2-baseline}
source_dir=${HIGGS_SOURCE_DIR:-$root/vllm-source}
port=${HIGGS_PORT:-18768}
configuration=${HIGGS_DEPLOY_CONFIG:-/data/higgs-routing/baseline.yaml}
stats_args=(--disable-log-stats)
if [[ ${HIGGS_LOG_STATS:-0} == 1 ]]; then
  stats_args=()
fi
sudo -n docker run -d --name "$name" --gpus all --network host --ipc private --shm-size 8g \
  --user "$(id -u):$(id -g)" -e HOME=/data \
  -e HF_HOME=/data/cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/vllm-omni -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e HIGGS_PROFILE_ARM="$HIGGS_PROFILE_ARM" -e OMP_NUM_THREADS=4 -e TRITON_CACHE_DIR=/data/cache/triton-vllm \
  -v "$root:/data" -v "$source_dir:/opt/vllm-omni:ro" \
  --entrypoint /data/venv-vllm/bin/vllm tts-runtime:20261117-pinned \
  serve /data/models/higgs-v2 --omni --deploy-config "$configuration" \
  --host 127.0.0.1 --port "$port" --trust-remote-code "${stats_args[@]}"
