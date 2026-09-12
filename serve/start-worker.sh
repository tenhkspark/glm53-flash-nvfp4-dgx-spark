#!/bin/bash
set -euo pipefail
IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130
MODEL_DIR=${WEIGHTS_ROOT:-CHANGEME}/weights/GLM-5.3-Flash-NVFP4
HEAD_IP="${HEAD_IP:?}"; MY_IP="${MY_IP:?}"; IFNAME="${IFNAME:?}"
test -d "$MODEL_DIR"
docker rm -f glm53-official-worker >/dev/null 2>&1 || true
docker run -d --name glm53-official-worker --network host --gpus all --shm-size=32g --ipc=host \
  -v "$MODEL_DIR":/checkpoint:ro -v ${WEIGHTS_ROOT:-CHANGEME}/vllm-overlay/flashinfer_mla_sparse_sm120.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:ro \
  --entrypoint bash \
  "$IMAGE" -lc "python3 -c 'import ray' 2>/dev/null || pip install -q ray; \
    export VLLM_HOST_IP=$MY_IP NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME && \
    ray start --address=$HEAD_IP:6399 --node-ip-address=$MY_IP --num-gpus=1 && sleep infinity"
echo "worker started"
