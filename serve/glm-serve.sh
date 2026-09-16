#!/bin/bash
# One command to bring up / inspect / stop / smoke-test GLM-5.3-Flash-NVFP4 on two DGX Spark nodes (Ray head + worker, TP=2).
# Usage: see README.md. Startup pitfalls: LOG.md.
set -euo pipefail
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"

# ==== Configuration (edit only this block) ====
IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130
MODEL_DIR=${WEIGHTS_ROOT:-CHANGEME}/weights/GLM-5.3-Flash-NVFP4
OVERLAY_FILE=flashinfer_mla_sparse_sm120.py
REMOTE_OVERLAY=recipe-factory/vllm-overlay      # path relative to $HOME on each node
PORT=8888
HEAD_HOST[1]=CHANGEME;  HEAD_IP[1]=CHANGEME;  HEAD_IF[1]=CHANGEME
WORK_HOST[1]=CHANGEME;  WORK_IP[1]=CHANGEME;  WORK_IF[1]=CHANGEME
HEAD_HOST[2]=CHANGEME;  HEAD_IP[2]=CHANGEME;  HEAD_IF[2]=CHANGEME
WORK_HOST[2]=CHANGEME;  WORK_IP[2]=CHANGEME;  WORK_IF[2]=CHANGEME
HEAD_NAME=glm53-official-head
WORK_NAME=glm53-official-worker
# ========================================
REMOTE_SCRIPTS=$(dirname "$REMOTE_OVERLAY")     # start-*.sh land next to the overlay dir

usage() {
  cat <<EOF
Usage: glm-serve.sh <subcommand> [pair]
  up <pair>      start pair (1|2): remove worker -> start head -> wait for Ray -> start worker -> wait READY (25 min max)
  down <pair>    stop the pair (other containers are left alone)
  status         print a table of both pairs
  smoke <pair>   send one question (thinking skipped) and print the answer
  sync-overlay   copy the overlay and start scripts to every configured node (checksum-verified)
Set GLM_AUTOTUNE=1 to enable FlashInfer autotune (off by default)
EOF
  exit 1
}

ssh_run() { ssh -o ConnectTimeout=10 -o BatchMode=yes "$@"; }

pair_configured() { # $1=pair. False for pairs left at CHANGEME or empty (two-node setups)
  local p=$1
  [ -n "${HEAD_HOST[$p]:-}" ] && [ "${HEAD_HOST[$p]}" != CHANGEME ] \
    && [ -n "${WORK_HOST[$p]:-}" ] && [ "${WORK_HOST[$p]}" != CHANGEME ]
}

require_pair() { # $1=pair. Stop with a readable message instead of ssh-ing to "CHANGEME"
  pair_configured "$1" && return 0
  echo "FAIL: pair $1 is not configured: HEAD_HOST[$1]/WORK_HOST[$1] are still CHANGEME or empty." >&2
  echo "      Edit the configuration block at the top of this script." >&2
  exit 1
}

# Digest helper. macOS ships md5/shasum, Linux ships md5sum/sha256sum, and this
# script may run from either kind of machine. Pick ONE algorithm and use the
# matching command on both ends - two different algorithms never compare equal.
if command -v md5sum >/dev/null 2>&1; then
  HASH_ALGO=md5;    hash_local() { md5sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  HASH_ALGO=sha256; hash_local() { shasum -a 256 "$1" | awk '{print $1}'; }
elif command -v md5 >/dev/null 2>&1; then
  HASH_ALGO=md5;    hash_local() { md5 -q "$1"; }
else
  echo "FAIL: need one of md5sum, shasum or md5 to verify the copied files" >&2; exit 1
fi

hash_remote_cmd() { # $1=remote path. Same algorithm as hash_local, whichever tool the node has
  if [ "$HASH_ALGO" = md5 ]; then
    echo "{ md5sum $1 2>/dev/null || md5 -q $1; } | awk '{print \$1}'"
  else
    echo "{ sha256sum $1 2>/dev/null || shasum -a 256 $1; } | awk '{print \$1}'"
  fi
}

head_died() { # $1=host $2=reason. Print the error lines from the head container log and exit 1
  echo "FAIL: $2" >&2
  ssh_run "$1" "docker logs $HEAD_NAME 2>&1 | grep -Ei 'error|died|RuntimeError' | tail -20" >&2 || true
  exit 1
}

# Launch options match start-head.sh / start-worker.sh in this directory.
# The --kernel-config JSON is passed via a variable so the quotes survive bash -lc (LOG.md, pitfall 12).
remote_head_script() {
  cat <<'REMOTE'
set -euo pipefail
HEAD_IP=$1; IFNAME=$2; PORT=$3; AUTOTUNE=$4; MODEL_DIR=$5; IMAGE=$6; OVERLAY_DIR=$7; OVERLAY_FILE=$8
if [ "$AUTOTUNE" = "1" ]; then
  KC='{"enable_flashinfer_autotune": true}'
else
  KC='{"enable_flashinfer_autotune": false}'
fi
test -d "$MODEL_DIR" || { echo "FAIL: no model dir $MODEL_DIR on $(hostname) - copy the checkpoint there or fix WEIGHTS_ROOT" >&2; exit 1; }
docker rm -f glm53-official-head >/dev/null 2>&1 || true
docker run -d --name glm53-official-head --network host --gpus all --shm-size=32g --ipc=host \
  -v "$MODEL_DIR":/checkpoint:ro \
  -v "$HOME/$OVERLAY_DIR/$OVERLAY_FILE":/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/$OVERLAY_FILE:ro \
  --entrypoint bash \
  "$IMAGE" -lc "python3 -c 'import ray' 2>/dev/null || pip install -q ray; \
    export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HOST_IP=$HEAD_IP NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME && \
    ray start --head --node-ip-address=$HEAD_IP --port=6399 --num-gpus=1 --dashboard-host=127.0.0.1 && \
    exec vllm serve /checkpoint \
    --served-model-name GLM-5.3-Flash-NVFP4 \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 2 --data-parallel-size 1 --distributed-executor-backend ray \
    --enable-expert-parallel --enable-ep-weight-filter \
    --reasoning-parser glm45 --kernel-config '$KC' \
    --kv-cache-dtype fp8 \
    --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 32}' \
    --max-num-batched-tokens 8192 --enable-chunked-prefill \
    --max-num-seqs 32 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{\"image\":0,\"video\":0}'"
REMOTE
}

remote_worker_script() {
  cat <<'REMOTE'
set -euo pipefail
HEAD_IP=$1; MY_IP=$2; IFNAME=$3; MODEL_DIR=$4; IMAGE=$5; OVERLAY_DIR=$6; OVERLAY_FILE=$7
test -d "$MODEL_DIR" || { echo "FAIL: no model dir $MODEL_DIR on $(hostname) - copy the checkpoint there or fix WEIGHTS_ROOT" >&2; exit 1; }
docker rm -f glm53-official-worker >/dev/null 2>&1 || true
docker run -d --name glm53-official-worker --network host --gpus all --shm-size=32g --ipc=host \
  -v "$MODEL_DIR":/checkpoint:ro \
  -v "$HOME/$OVERLAY_DIR/$OVERLAY_FILE":/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/$OVERLAY_FILE:ro \
  --entrypoint bash \
  "$IMAGE" -lc "python3 -c 'import ray' 2>/dev/null || pip install -q ray; \
    export VLLM_HOST_IP=$MY_IP NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME && \
    ray start --address=$HEAD_IP:6399 --node-ip-address=$MY_IP --num-gpus=1 && sleep infinity"
REMOTE
}

cmd_up() {
  local p=${1:?pair required (1|2)}
  require_pair "$p"
  local hh=${HEAD_HOST[$p]} hi=${HEAD_IP[$p]} hf=${HEAD_IF[$p]}
  local wh=${WORK_HOST[$p]} wi=${WORK_IP[$p]} wf=${WORK_IF[$p]}
  local t0=$(date +%s)

  echo "[pair $p] removing worker (${wh})"
  ssh_run "$wh" "docker rm -f $WORK_NAME >/dev/null 2>&1 || true"

  echo "[pair $p] starting head ($hh, $hi/${hf})"
  ssh_run "$hh" bash -s -- "$hi" "$hf" "$PORT" "${GLM_AUTOTUNE:-0}" "$MODEL_DIR" "$IMAGE" "$REMOTE_OVERLAY" "$OVERLAY_FILE" < <(remote_head_script)

  echo "[pair $p] waiting for Ray ($hh:6399, 300 s max)"
  local i=0
  until ssh_run "$hh" "timeout 2 bash -c '</dev/tcp/127.0.0.1/6399'" 2>/dev/null; do
    i=$((i+5)); [ $i -ge 300 ] && head_died "$hh" "Ray did not come up within 300 s"
    ssh_run "$hh" "docker inspect -f '{{.State.Running}}' $HEAD_NAME 2>/dev/null" | grep -q true || head_died "$hh" "head container exited"
    sleep 5
  done

  echo "[pair $p] starting worker (${wh}, $wi/${wf})"
  ssh_run "$wh" bash -s -- "$hi" "$wi" "$wf" "$MODEL_DIR" "$IMAGE" "$REMOTE_OVERLAY" "$OVERLAY_FILE" < <(remote_worker_script)

  echo "[pair $p] waiting for READY (25 min max)"
  i=0
  while :; do
    if ssh_run "$hh" "curl -s --max-time 5 http://127.0.0.1:$PORT/v1/models" | grep -q GLM; then
      echo "READY (pair $p, $(( $(date +%s) - t0 )) s)"
      return 0
    fi
    i=$((i+15)); [ $i -ge 1500 ] && head_died "$hh" "not READY within 25 min"
    ssh_run "$hh" "docker inspect -f '{{.State.Running}}' $HEAD_NAME 2>/dev/null" | grep -q true || head_died "$hh" "head container exited"
    sleep 15
  done
}

cmd_down() {
  local p=${1:?pair required (1|2)}
  require_pair "$p"
  echo "[pair $p] stopping worker (${WORK_HOST[$p]})"
  ssh_run "${WORK_HOST[$p]}" "docker rm -f $WORK_NAME"
  echo "[pair $p] stopping head (${HEAD_HOST[$p]})"
  ssh_run "${HEAD_HOST[$p]}" "docker rm -f $HEAD_NAME"
  echo "[pair $p] down done"
}

cmd_status() {
  printf '%-4s %-16s %-16s %-5s %s\n' pair head worker API 'last log line'
  local p
  for p in 1 2; do
    # skip pairs that are CHANGEME or empty (two-node setups)
    pair_configured "$p" || continue
    local hh=${HEAD_HOST[$p]} wh=${WORK_HOST[$p]}
    local hs ws api log
    hs="$(ssh_run "$hh" "docker inspect -f '{{.State.Status}}' $HEAD_NAME 2>/dev/null || echo missing")"
    ws="$(ssh_run "$wh" "docker inspect -f '{{.State.Status}}' $WORK_NAME 2>/dev/null || echo missing")"
    if ssh_run "$hh" "curl -s --max-time 5 http://127.0.0.1:$PORT/v1/models" | grep -q GLM; then api=OK; else api='---'; fi
    log="$(ssh_run "$hh" "docker logs --tail 1 $HEAD_NAME 2>&1" | tail -c 100 | tr '\n' ' ')" || true
    printf '%-4s %-16s %-16s %-5s %s\n' "$p" "$hh:$hs" "$wh:$ws" "$api" "$log"
  done
}

cmd_smoke() {
  local p=${1:?pair required (1|2)}
  [ -n "${HEAD_HOST[$p]:-}" ] || usage
  local hh=${HEAD_HOST[$p]}
  # Continue an empty assistant turn to skip the thinking block (LOG.md, pitfall 14). If content is empty, take reasoning.
  local req='{"model":"GLM-5.3-Flash-NVFP4","messages":[{"role":"user","content":"What is DGX Spark? Answer in one sentence."},{"role":"assistant","content":""}],"continue_final_message":true,"add_generation_prompt":false,"max_tokens":256,"temperature":0}'
  local t0=$(date +%s)
  local resp
  resp="$(ssh_run "$hh" "curl -s --max-time 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @-" <<<"$req")"
  local sec=$(( $(date +%s) - t0 ))
  local text ct
  text="$(jq -r '.choices[0].message.content // empty' <<<"$resp")"
  [ -n "$text" ] || text="$(jq -r '.choices[0].message.reasoning // empty' <<<"$resp")"
  ct="$(jq -r '.usage.completion_tokens // 0' <<<"$resp")"
  if [ -z "$text" ]; then
    echo "FAIL: empty answer. Response:" >&2; jq -c . <<<"$resp" | head -c 2000 >&2; echo >&2; exit 1
  fi
  if [ "${#text}" -lt 20 ]; then
    echo "FAIL: answer too short:" >&2; printf '%s
' "$text" | head -c 500 >&2; exit 1
  fi
  echo "smoke(pair $p): OK  ${sec} s  completion_tokens=$ct"
  echo "answer: $text"
}

cmd_sync_overlay() {
  local src="$LLM_DIR/../overlay"
  local overlay_files=("$src"/*)
  local top_files=("$LLM_DIR/start-head.sh" "$LLM_DIR/start-worker.sh")
  local h ok=1
  local hosts=() p
  # skip pairs that are CHANGEME or empty (two-node setups)
  for p in 1 2; do
    if pair_configured "$p"; then
      hosts+=("${HEAD_HOST[$p]}" "${WORK_HOST[$p]}")
    fi
  done
  for h in "${hosts[@]}"; do
    echo "[sync] $h"
    ssh_run "$h" "mkdir -p ~/$REMOTE_OVERLAY ~/$REMOTE_SCRIPTS"
    rsync -az --delete "${overlay_files[@]}" "$h:$REMOTE_OVERLAY/"
    rsync -az "${top_files[@]}" "$h:$REMOTE_SCRIPTS/"
    local lf f rf lmd rmd
    for lf in "${overlay_files[@]}" "${top_files[@]}"; do
      f=$(basename "$lf")
      case "$f" in
        start-*) rf="$REMOTE_SCRIPTS/$f" ;;
        *) rf="$REMOTE_OVERLAY/$f" ;;
      esac
      lmd=$(hash_local "$lf")
      rmd=$(ssh_run "$h" "$(hash_remote_cmd "~/$rf")")
      if [ "$lmd" != "$rmd" ]; then echo "FAIL: $HASH_ALGO mismatch for $h:$rf" >&2; ok=0; fi
    done
  done
  [ "$ok" = 1 ] && echo "sync-overlay: all ${#hosts[@]} nodes match"
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  down) shift; cmd_down "$@" ;;
  status) cmd_status ;;
  smoke) shift; cmd_smoke "$@" ;;
  sync-overlay) cmd_sync_overlay ;;
  *) usage ;;
esac
