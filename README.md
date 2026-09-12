# GLM-5.3-Flash NVFP4 on DGX Spark — a one-file vLLM overlay for no-rope sparse MLA

As shipped, the official image does not start on DGX Spark (sm_121). This one-file overlay makes it run, verified on four nodes (two pairs).

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh.md) · [한국어](README.ko.md)

## Why this is public

An official open model should actually run for anyone who downloads it, not just on whatever hardware it was validated against. So I'm putting the raw breakdown here: what broke, where I hit a wall, and how I fixed each step, keeping timestamps and exact error messages intact. Hopefully this saves someone an all-nighter on the next release.

NVIDIA ships an official NVFP4 quantization of GLM-5.3-Flash alongside an official vLLM image (`vllm/vllm-openai:glm53-flash-arm64-cu130`, arm64 + CUDA 13 build — the model card's validation hardware is GB200). The image crashes on DGX Spark (compute capability sm_121) because the SM120 sparse-MLA attention backend doesn't have a code path for no-rope MLA, which is the exact attention shape GLM-5.3-Flash uses. I put together a single-file Python overlay in this repo that patches in the missing path. There are no C++/CUDA changes and no need to rebuild the image—I just bind-mount the file over the one it replaces.

## What didn't work

I spun up the official image across two DGX Spark nodes (Ray, tensor parallel = 2) using the official flags, and it dies right after loading while profiling the KV cache:

```
concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla
```

When I forced `--kv-cache-dtype bfloat16`, the failure just moved somewhere else:

```
No valid attention backend found ... head_size=512, use_mla=True, use_sparse=True
```

Basically, FP8 KV crashes the cache writer, and BF16 leaves me with no backend at all. None of the flag combinations work. I logged all 14 issues I hit while trying to get it up in [LOG.md](LOG.md).

## Cause
- GLM-5.3-Flash uses sparse MLA with an NSA-style token-selection indexer: attention only pulls the top-k selected tokens per query, and the attention itself runs no-rope — `qk_rope_head_dim=0`, `qk_nope_head_dim=256`, and `kv_lora_rank=512`.
- On compute capability 12.x, `vllm/platforms/cuda.py` narrows the MLA backend choices down to `TRITON_MLA` and `FLASHINFER_MLA_SPARSE_SM120`. Since the model is sparse, it defaults to picking `FlashInferMLASparseSM120Backend`.
- The SM100 backend (`flashinfer_mla_sparse.py`) actually handles this no-rope shape, but the SM120 backend (`flashinfer_mla_sparse_sm120.py`) doesn't: its KV write path calls `concat_and_cache_mla`, where the `fp8_ds_mla` branch expects a 64-wide rope tensor (`pe_dim must be 64`) — but without rope, that tensor simply isn't there.
- The SM120 sparse kernel itself (the v32 / GLM-NSA decode path) strictly requires a d_qk=576 layout (512 latent + 64 rope). I can't just copy the SM100 route over either, because the SM100 no-rope kernel (TRTLLM-GEN) rejects sm_121 completely (`Unsupported architecture` in `fmhaRunner.cuh`).
- Bottom line: the image was clearly tested on GB200-class hardware (sm_100) where the no-rope path exists. DGX Spark (sm_121) just gets caught right in the crack between both backends.

## The fix: zero-fill the rope dimensions

The kernel forces a 576-wide query/key layout, but those 64 rope dimensions don't actually need rope data. So I did this:

- **KV side**: I kept the SM120 default `fp8_ds_mla` cache layout (656-byte packed blob per token) and zeroed out the 64 rope slots. My cache writer override swaps the zero-width `k_pe` tensor for a 64-wide zero tensor right before `concat_and_cache_mla`.
- **Query side**: I padded the 512-wide latent query with 64 zeros and passed `qk_rope_head_dim=64` to the kernel.

Since zero times anything is zero, those extra dimensions add 0 to the dot products. The 576-wide kernel ends up computing the exact same attention as a native no-rope run—it's mathematically identical, not an approximation. I left the fp8 KV scaling (`kv_scale_format=arbitrary_fp32` for GLM models) as is.

Before touching the server, I tested this kernel path on sm_121 with standalone unit tests. Outputs match the BF16 reference with a max absolute error of 0.0024 (just fp8 quantization noise), and invalid (`-1`) top-k indices mask out fine (0.0031).

## Repository layout

```
overlay/flashinfer_mla_sparse_sm120.py   the overlay (drop-in replacement)
overlay.diff                            the same change as a diff
serve/start-head.sh                     Ray head + vLLM server (run on node A)
serve/start-worker.sh                   Ray worker (run on node B)
serve/glm-serve.sh                      up / down / status / smoke / sync wrapper
RUNBOOK.md                              copy-paste runbook: values, commands, expected output, common failures
LOG.md                                  startup log: 15 stumbling blocks, timestamps
```

## Quick start

[RUNBOOK.md](RUNBOOK.md) is a copy-paste runbook for an AI agent (or a human) to run: the values you need to plug in, the five commands in order, expected outputs with measured times for each step, and how to get past the five most common failures.

## Usage

Two DGX Spark nodes hooked together over a direct link (I use the QSFP ports; any directly reachable pair of IPs works—the scripts pin Ray and NCCL traffic to these point-to-point addresses).

1. Weights and image on both nodes. Built and tested against the official image (vLLM v0.28.1rc1.dev580+g385dce36b, flashinfer 0.6.18):
```bash
docker pull vllm/vllm-openai:glm53-flash-arm64-cu130
# NVIDIA GLM-5.3-Flash-NVFP4 checkpoint, e.g. under ~/models/GLM-5.3-Flash-NVFP4
```

2. Drop `overlay/flashinfer_mla_sparse_sm120.py` in the same directory on both nodes, like `~/vllm-overlay/` (or run `serve/glm-serve.sh sync-overlay`).

3. Bring up the head, then the worker (fill in the `CHANGEME` values or export them beforehand):
```bash
# node A (head)
HEAD_IP=CHANGEME IFNAME=CHANGEME ./serve/start-head.sh
# node B (worker)
HEAD_IP=CHANGEME MY_IP=CHANGEME IFNAME=CHANGEME ./serve/start-worker.sh
```
`HEAD_IP` and `MY_IP` are the endpoints of the point-to-point link; `IFNAME` is the interface carrying that IP. Cold start takes ~13 minutes (191 GB of weights). The scripts bump `VLLM_ENGINE_READY_TIMEOUT_S` to 3600 s because the default 600 s times out.

4. Order matters here: if an old worker is still hanging around from a previous run, kill it before restarting the head, and only then bring the worker back up. A leftover worker trying to join a new Ray cluster crashes the head with a session-mismatch error. `serve/glm-serve.sh up` handles the sequence automatically (`down` / `up` / `status` / `smoke`).

5. FlashInfer autotune is explicitly turned off (`--kernel-config '{"enable_flashinfer_autotune": false}'`). Leaving it on crashed rank 0 during the autotune dummy pass on this setup.

### Skipping the thinking block

The GLM-5.3 chat template forces an open `<think>` tag (`add_generation_prompt` injects `<|assistant|><think>`) and doesn't provide an `enable_thinking` toggle. If you set a small `max_tokens`, the model blows the whole budget inside the thought block, leaving you with an empty response.

The workaround I use is appending an empty assistant turn and requesting a continuation:
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": ""}
  ],
  "continue_final_message": true,
  "add_generation_prompt": false
}
```

This makes generation start right from `<|assistant|><think></think>`, closing the thinking block immediately so the output goes straight into the actual response (tested a 256-token request and got 183 tokens of straight answer text). 

Catch: because the `glm45` reasoning parser never encounters a closing `</think>`, it might tag the entire body as `reasoning` and leave `content` blank. Just read `content`, and if it's empty, grab `reasoning`. `serve/glm-serve.sh smoke` already handles this fallback.

## Measured (2× DGX Spark, TP=2 over Ray)
- Engine init took 105 s; KV cache came out to 7.69 GiB per node (`--max-model-len 131072`, fp8_ds_mla layout).
- Ran 48 requests at 16 concurrent: 0 failures, 61.5 tok/s aggregate, case sat around 64–66 °C.
- Bumped it to 32 concurrent for the follow-up production run without changing anything.

## Time spent

From my first attempt to bring it up in the middle of the night on 2026-09-12 to the actual production run later that day (all times local):

| Time | Milestone |
| --- | --- |
| small hours | Copied weights over direct links; tried starting my own build |
| 06:50 | Called it on my own build after 6 failed startups — hit the wall detailed in "What didn't work" |
| 07:10 | Spotted the official image on the model card |
| 07:40 | Confirmed the official image ran into the exact same wall (TP=2) |
| 07:40–11:30 | Dug out root cause, put together the overlay, wrote and ran five standalone unit tests on sm_121 |
| 11:30–11:50 | Overlay startup attempts 1 and 2 failed |
| 12:32 | Overlay attempt 3: server finally came up (engine init took 105 s) |
| 12:35–12:47 | Ran validation: 48 requests, concurrency 16, pushed 61.5 tok/s aggregate |
| ~12:50 | Kicked off the production run for the training corpus at 32 concurrent |

Total startup attempts: 6 on my own build, 3 on the official image, and 3 on the overlay — overlay #3 was the one that finally worked. I tracked wait times and where each failure died in [LOG.md](LOG.md).

## How this was made
- **Planning and verification** — GLM-5.3 (Z.ai): helped plan the steps, checked results, and kept the work on track.
- **Implementation** — GLM-5.3-Flash in the Z.ai app: I used reasoning effort Low for design through the overlay, and bumped it to High for the `serve/glm-serve.sh` wrapper.
- **Research and triage** — GLM-5.3 (Z.ai): dug through the official docs and narrowed down the root causes.
- **Hardware** — four DGX Spark nodes (DGX Spark), split into two pairs with a direct QSFP link between each pair.
- **Software** — official vLLM image (`vllm/vllm-openai:glm53-flash-arm64-cu130`) plus my custom `serve/glm-serve.sh` script.
- I made the decisions and moved output between the tools; the models handled the implementation and testing.
- I didn't use any third-party recipes, patches, or forks here. Everything came straight from official sources: the NVIDIA model card and image, the vLLM source code already inside that image, and Z.ai's docs.

## Log

I kept the full startup log in [LOG.md](LOG.md). It lists all 15 issues in the order I ran into them, along with timestamps and the raw error messages. It's basically my running record of why each flag in the serve scripts is set the way it is.

## Acknowledgments

Thank you to Z.ai for releasing GLM-5.3-Flash as an open model, and to NVIDIA for publishing the official NVFP4 checkpoint and the official vLLM image this runs on.
