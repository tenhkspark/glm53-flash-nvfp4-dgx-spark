#!/bin/bash
# Start nvidia/GLM-5.3-Flash-NVFP4 with the official vLLM image (arm64, CUDA 13) as TP=2 (head + worker on two nodes, Ray).
# Launch options follow the official command on the NVIDIA model card, adapted to TP=2.
set -euo pipefail
MODEL_DIR=${WEIGHTS_ROOT:-CHANGEME}/weights/GLM-5.3-Flash-NVFP4
IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130
HEAD_IP="${HEAD_IP:?set HEAD_IP}"          # this node's IP on the direct link
IFNAME="${IFNAME:?set IFNAME}"            # interface name of that link
PORT="${PORT:-8888}"
GMEM="${GPU_MEM:-0.85}"
MAXLEN="${MAX_MODEL_LEN:-131072}"
MAXSEQS="${MAX_NUM_SEQS:-32}"
test -d "$MODEL_DIR"
docker rm -f glm53-official-head >/dev/null 2>&1 || true
docker run -d --name glm53-official-head --network host --gpus all --shm-size=32g --ipc=host \
  -v "$MODEL_DIR":/checkpoint:ro -v ${WEIGHTS_ROOT:-CHANGEME}/vllm-overlay/flashinfer_mla_sparse_sm120.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:ro \
  --entrypoint bash \
  "$IMAGE" -lc "python3 -c 'import ray' 2>/dev/null || pip install -q ray; \
    export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HOST_IP=$HEAD_IP NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME && \
    ray start --head --node-ip-address=$HEAD_IP --port=6399 --num-gpus=1 --dashboard-host=127.0.0.1 && \
    exec vllm serve /checkpoint \
    --served-model-name GLM-5.3-Flash-NVFP4 \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 2 --data-parallel-size 1 --distributed-executor-backend ray \
    --enable-expert-parallel --enable-ep-weight-filter \
    --reasoning-parser glm45 --kernel-config '{\"enable_flashinfer_autotune\": false}' \
    --kv-cache-dtype fp8 \
    --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 32}' \
    --max-num-batched-tokens 8192 --enable-chunked-prefill \
    --max-num-seqs $MAXSEQS --max-model-len $MAXLEN \
    --gpu-memory-utilization $GMEM \
    --limit-mm-per-prompt '{\"image\":0,\"video\":0}'"
echo "head started (port $PORT)"
