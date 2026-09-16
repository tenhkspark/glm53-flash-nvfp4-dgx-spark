# RUNBOOK — GLM-5.3-Flash NVFP4 on 2× DGX Spark

A copy-paste runbook for an AI agent (or a human) to run straight through. Go top to bottom: fill in the values, run the five commands in order, check the expected output, and hit the failure list if something doesn't match. The background for every flag and every failure is in [LOG.md](LOG.md).

## Prerequisites

- Two DGX Spark nodes (DGX Spark) connected back-to-back. I used the QSFP ports, but any directly reachable pair of IPs works — the scripts pin Ray and NCCL traffic to these point-to-point addresses.
- Both nodes need Docker, the official `vllm/vllm-openai:glm53-flash-arm64-cu130` image (built and tested with vLLM v0.28.1rc1.dev580+g385dce36b and flashinfer 0.6.18), and the 191 GB NVIDIA GLM-5.3-Flash-NVFP4 checkpoint.
- Wherever you run `serve/glm-serve.sh` from (either node or a laptop), you need passwordless SSH access to both machines, plus `rsync`, `jq`, and a checksum tool on both ends (`md5sum`, `shasum`, or `md5` — the script picks whichever it finds and uses the matching command on the nodes). No need to install Ray on the hosts; the container brings it up when it starts.
- Run the commands from the repository root.

## Values to fill in

Edit the configuration block at the top of `serve/glm-serve.sh`. Everything ships as `CHANGEME`:

| Value | Meaning |
| --- | --- |
| `HEAD_HOST[1]` | SSH host of node A (runs the head) |
| `WORK_HOST[1]` | SSH host of node B (runs the worker) |
| `HEAD_IP[1]` / `WORK_IP[1]` | the two endpoints of the direct link (point-to-point IPs) |
| `HEAD_IF[1]` / `WORK_IF[1]` | interface name carrying each of those IPs |
| `WEIGHTS_ROOT` | not in the script — exported in the shell; weights live at `$WEIGHTS_ROOT/weights/GLM-5.3-Flash-NVFP4` on both nodes (absolute path) |
| `IMAGE` | image tag; default `vllm/vllm-openai:glm53-flash-arm64-cu130` |

If you're running a two-node setup, you can just leave pair 2 (`HEAD_HOST[2]` … `WORK_IF[2]`) as `CHANGEME`; those pairs are skipped automatically. `PORT` defaults to 8888, and `REMOTE_OVERLAY` defaults to `recipe-factory/vllm-overlay` (relative to `$HOME` on each node).

## Commands, in order

```bash
export WEIGHTS_ROOT=/absolute/path/on/the/nodes
./serve/glm-serve.sh sync-overlay
./serve/glm-serve.sh down 1
./serve/glm-serve.sh up 1
./serve/glm-serve.sh status
./serve/glm-serve.sh smoke 1
```

`up` blocks for ~14 min (measured 13 min 47 s and 14 min 16 s to READY on the two pairs); just let it run, don't re-run it in parallel. Here is the expected output per step, with the times I measured on 2026-09-12 at 21:57 using a fresh agent on a second node pair (node A / node B) and only the published files:

1. **`sync-overlay`** — 4.7 s. Copies the overlay and start scripts to every configured node and verifies the checksums:
```
[sync] <node A ssh host>
[sync] <node B ssh host>
sync-overlay: all 2 nodes match
```
2. **`down 1`** — 0.8 s. Removes worker first, then head:
```
[pair 1] stopping worker (<node B ssh host>)
[pair 1] stopping head (<node A ssh host>)
down done
```
3. **`up 1`** — 856 s (weights load ~12 min + engine init; READY at 14 min 16 s):
```
[pair 1] removing worker (<node B ssh host>)
[pair 1] starting head (<node A ssh host>, <ip>/<interface>)
[pair 1] waiting for Ray (<node A ssh host>:6399, 300 s max)
[pair 1] starting worker (<node B ssh host>, <ip>/<interface>)
[pair 1] waiting for READY (25 min max)
READY (pair 1, 856 s)
```
4. **`status`** — 1.3 s. Head and worker should both be `running` and API `OK`:
```
pair  head              worker              API  last log line
1     <node A>:running  <node B>:running    OK   ...
```
5. **`smoke 1`** — 5.2 s. One question with the thinking block skipped:
```
smoke(pair 1): OK  5.2 s  completion_tokens=<number>
answer: DGX Spark is NVIDIA's compact, palm-sized AI supercomputer (formerly known as Project DIGITS) that delivers petaflop-scale GPU computing performance for running large AI models locally on a developer's desk.
```

## Five common failures

1. **`concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla`** during KV-cache profiling right after weights load. Either the overlay isn't mounted, or the KV cache dtype got set to `auto` (which picks up the checkpoint's `kv_cache_quant_algo: FP8` and hits the same error). Run `sync-overlay`, then `down 1` and `up 1` again — the overlay bind-mounts when the container starts — and leave the KV cache dtype alone. (LOG.md #7, #10)
2. **`No valid attention backend found ... head_size=512, use_mla=True, use_sparse=True`**. Someone forced the KV cache to bfloat16. On sm_121, BF16 leaves no sparse-MLA backend at all; leave `--kv-cache-dtype fp8` as-is. The fix on this hardware is the overlay, not the dtype. (LOG.md #8)
3. **`ActorHandle objects are not valid across Ray sessions`** right after the weights load. A leftover worker from an earlier run attached to a fresh head. Run `down` before `up` every time (`up` already cleans up the worker first), and don't restart the head while an old worker is still hanging around. (LOG.md #2, #12a)
4. **Rank 0 dies after graph capture** during FlashInfer SM120 sparse decode autotuning (the mixed-batch dummy run). Autotune is off by default (`--kernel-config '{"enable_flashinfer_autotune": false}'`); do not set `GLM_AUTOTUNE=1` on this hardware. (LOG.md #12b)
5. **`up` reports FAIL, or not READY within 25 min.** Startup takes ~14 min end to end (9–12 min just loading the 191 GB of weights, plus engine init), so the scripts set `VLLM_ENGINE_READY_TIMEOUT_S=3600` (the default 600 s will time out). Check the `FAIL:` line and the head container log it prints. Don't retype the `--kernel-config` JSON by hand: inside the double-quoted `bash -lc` string the inner quotes need `\"`, and a single quoting mistake kills the container instantly (cost me 28 minutes once just watching a dead container) — `glm-serve.sh` passes it through a variable so this cannot happen. (LOG.md #12b)
