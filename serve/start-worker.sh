#!/bin/bash
set -euo pipefail
IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130
WEIGHTS_ROOT="${WEIGHTS_ROOT:?set WEIGHTS_ROOT to the directory holding weights/GLM-5.3-Flash-NVFP4 on this node}"
MODEL_DIR="$WEIGHTS_ROOT/weights/GLM-5.3-Flash-NVFP4"
OVERLAY_DIR="${REMOTE_OVERLAY:-recipe-factory/vllm-overlay}"   # $HOME-relative, same path glm-serve.sh syncs to
OVERLAY_FILE=flashinfer_mla_sparse_sm120.py
HEAD_IP="${HEAD_IP:?}"; MY_IP="${MY_IP:?}"; IFNAME="${IFNAME:?}"
test -d "$MODEL_DIR" || { echo "FAIL: no model dir $MODEL_DIR - copy the checkpoint there or fix WEIGHTS_ROOT" >&2; exit 1; }
test -f "$HOME/$OVERLAY_DIR/$OVERLAY_FILE" || { echo "FAIL: no overlay $HOME/$OVERLAY_DIR/$OVERLAY_FILE - copy it there or run serve/glm-serve.sh sync-overlay" >&2; exit 1; }
docker rm -f glm53-official-worker >/dev/null 2>&1 || true
docker run -d --name glm53-official-worker --network host --gpus all --shm-size=32g --ipc=host \
  -v "$MODEL_DIR":/checkpoint:ro \
  -v "$HOME/$OVERLAY_DIR/$OVERLAY_FILE":"/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/$OVERLAY_FILE":ro \
  --entrypoint bash \
  "$IMAGE" -lc "python3 -c 'import ray' 2>/dev/null || pip install -q ray; \
    export VLLM_HOST_IP=$MY_IP NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME && \
    ray start --address=$HEAD_IP:6399 --node-ip-address=$MY_IP --num-gpus=1 && sleep infinity"
echo "worker started"
