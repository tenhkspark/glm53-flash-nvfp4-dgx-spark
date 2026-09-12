# Startup log — GLM-5.3-Flash NVFP4 on 2× DGX Spark (2026-09-12)

Timeline of everything that went wrong while bringing up nvidia/GLM-5.3-Flash-NVFP4 on two DGX Spark nodes (Ray, TP=2), ending in a working server at 12:32. Error messages are verbatim. Times are local.

Materials: the official NVFP4 checkpoint, the official vLLM image `vllm/vllm-openai:glm53-flash-arm64-cu130`, and my serve scripts running vLLM with Ray TP=2 over a direct QSFP link between the two nodes.

## Time spent

It took me 12 attempts to get one working server: 6 on my own build, 3 on the official image, and 3 on the overlay (the third attempt finally came up). Any run that made it past initial config checks cost ~13 minutes just loading weights (191 GB). Wait times below are rough estimates.

| Route | # | Wait before it fell | Stage where it fell |
| --- | --- | --- | --- |
| own build | 1 | seconds | argument parsing — `--tool-call-parser glm47_moe` not in this build (#1) |
| own build | 2 | ~13 min | Ray session mismatch right after loading hit 100% (#2) |
| own build | 3 | ~13 min | same error again after a full rebuild (misdiagnosis, #2) |
| own build | 4 | minutes | Ray worker init — `Every node should have a unique IP address ... 3 unique IP addresses` (#6) |
| own build | 5 | ~13 min | KV-cache profiling — `pe_dim must be 64 for fp8_ds_mla` with `--kv-cache-dtype auto` (#7) |
| own build | 6 | engine init | backend selection — `No valid attention backend found` with BF16 (#8); gave up 06:50 |
| official image | 1–3 | ~13 min each | after loading: FP8 KV → the same `pe_dim must be 64` (07:40, #10); BF16 → no valid backend. No flag combination through |
| overlay | 1 | ~1 min | worker join — Ray session mismatch again, stale unpatched worker (#12a) |
| overlay | 2 | ~14 min | after graph capture — rank 0 died in the FlashInfer autotune dummy run (#12b) |
| overlay | 3 | 28 min lost, then up | `--kernel-config` quoting slipped → instant `vllm serve: error`; fixed → **12:32 up** |

1. **Missing parser name.** `--tool-call-parser glm47_moe` is not in this build (only `glm45` / `glm47` are). Since I only need generation here, I dropped the tool parser and kept `--reasoning-parser glm45`.

2. **Ray session mismatch.** Right after model loading hit 100%:
```
ActorHandle objects are not valid across Ray sessions (job 01000000 → 02000000)
```
vLLM v1 spawns EngineCore in a separate process, which runs its own `ray.init()`—meaning a whole new session. Setting `VLLM_ENABLE_V1_MULTIPROCESSING=0` didn't help (EngineCore stays out-of-process on the Ray executor path). The issue is Ray executor v2 (`ray_executor_v2.py`); setting `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0` forces the old executor instead. Build: vLLM v0.28.1rc1.dev559+g28a2ccee7. I initially misdiagnosed this as a stale worker still hanging onto a dead head, rebuilt everything from scratch, and ran right back into it.

3. **Nodes cannot resolve each other's names.** Node names only exist in my laptop's `/etc/hosts`, so running `rsync nodeB:` from node A fails. I fixed this by using the point-to-point IP addresses of the direct QSFP links directly. SSH keys are forwarded from only one of the nodes.

4. **`pkill -f` killed my own session.** Running `ssh nodeA 'pkill -f "rsync -a --partial"; ...'` matched its own command line (the pattern string was sitting right there in the ssh call) and killed my own session.

5. **Copy timings.** Moving 191 GB of weights over the QSFP link: 350 MB/s, ~10 min per node. 31 GB image moved via `docker save | ssh docker load`.

6. **Three IPs.** With the old executor, Ray threw:
```
Every node should have a unique IP address ... 3 unique IP addresses
```
My head script was exporting `VLLM_HOST_IP` *after* `ray start`, so only the head's local Ray workers announced the laptop-network IP. Moving that export before `ray start` fixed it. The worker script already had the right order.

7. **KV cache dtype.** Once past distributed init and loading, profiling crashed with `concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla`. This image was built for DeepSeek and defaults the KV cache to fp8_ds_mla, which does not match GLM's MLA shape. Trying `--kv-cache-dtype auto` (5th attempt) blew up the exact same way: the checkpoint's `hf_quant_config.json` specifies `kv_cache_quant_algo: FP8`, and `auto` just follows it into FP8 KV. I had to explicitly pass `--kv-cache-dtype bfloat16` next (6th attempt).

8. **The wall.** Running bfloat16:
```
No valid attention backend found ... head_size=512, use_mla=True, use_sparse=True
```
GLM-5.3-Flash uses sparse MLA (DSA-style); this build's sparse MLA kernels assume FP8 KV and DeepSeek's pe_dim=64. FP8 dies on pe_dim, BF16 has no backend—no flag combination makes it through. GLM-5's sparse attention needs a vLLM that actually supports it. I called it for the night here (06:50).

9. **Should have read the model card first (07:10).** The nvidia/GLM-5.3-Flash-NVFP4 model card explicitly specifies the official image `vllm/vllm-openai:glm53-flash-arm64-cu130` (arm64, CUDA 13—aimed at DGX Spark) and the official launch command (`--kv-cache-dtype fp8 --reasoning-parser glm45 --enable-expert-parallel --enable-ep-weight-filter --max-num-batched-tokens 8192 --enable-chunked-prefill`, TP=4). I only caught this after six failed startups on my own DeepSeek-oriented build. Lesson: official weights ship paired with an official runtime.

10. **The official image hits the same wall (07:40).** Running TP=2 on `vllm/vllm-openai:glm53-flash-arm64-cu130` (vLLM v0.28.1rc1.dev580+g385dce36b) throws that exact same `pe_dim must be 64 for fp8_ds_mla`. Root cause: `vllm/platforms/cuda.py` caps the MLA candidates on capability 12.x (DGX Spark = 12.1) to `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`, and because the model is sparse, it picks `FlashInferMLASparseSM120Backend`. The SM100 backend (`flashinfer_mla_sparse.py`) explicitly supports native no-rope MLA (`qk_rope_head_dim=0`, nope 256, kv_lora_rank 512); the SM120 one completely lacks that branch, so its KV write falls into `concat_and_cache_mla` (fp8_ds_mla, hardcoded pe_dim=64). The image works on the hardware it was validated on (GB200, sm_100); DGX Spark (sm_121) falls through the cracks. The only way through is adding a no-rope path to the SM120 backend.

11. **Breakthrough.** I wrote the no-rope branch for `flashinfer_mla_sparse_sm120.py` (zero-filling the 64 rope slots on the K side, zero-padding query from 512 to 576). Ran five standalone tests on sm_121: 0.0024 max error compared against a BF16 reference. Just one Python file, zero C++/CUDA edits. I mounted it into the official image as a bind-mount overlay and served it using the scripts in `serve/`.

12. **Overlay startup (11:30–11:50).**
- (a) I restarted the head while an unpatched worker was still running → hit the Ray session mismatch again (same issue as #2). The working order: kill the worker, then the head, then start worker.
- (b) Model loading, profiling, KV sizing, and graph capture went through fine, but rank 0 died during FlashInfer SM120 sparse MLA decode autotuning (on the mixed-batch dummy run). I disabled autotuning with `--kernel-config '{"enable_flashinfer_autotune": false}'` and bumped `VLLM_ENGINE_READY_TIMEOUT_S=3600` (default is 600 s; this 13-minute startup needs way more room). **12:32: up.** Engine init took 105 s, 7.69 GiB KV cache per node. Sent a prompt about DGX Spark in Japanese and got the right thinking back (128 GB, Project DIGITS heritage)—the zero-filled sparse MLA path works in actual production, not just unit tests.
- Also: that `--kernel-config` JSON has to use `\"` inside the double-quoted `bash -lc` string. My quoting broke once, `vllm serve: error` immediately dropped the container, and my watcher missed the exit, and 28 minutes went by while it polled an idle state.

13. **Numbers (12:35–12:47, node A / node B, 16 concurrent).** 48 requests, 0 failures, 61.5 tok/s aggregate, temps at 64–66 °C. It solved a written-out math problem without errors, self-check included. Scaled it up to 32 concurrent for the production run right after.

14. **Thinking ate the body (13:10).** Ran 179 requests and every single one got cut off at `max_tokens=1024`, spitting back nothing except the (English) thinking trace as the actual body. GLM-5.3's `chat_template.jinja` automatically injects `<think>` whenever `add_generation_prompt` is on; there's no `enable_thinking` toggle. My workaround: terminate the message list with an empty assistant message and set `continue_final_message: true, add_generation_prompt: false`—generation kicks off right from `<|assistant|><think></think>` and bypasses thinking entirely (gives 183 tokens of Japanese body text only). Side effect: the reasoning parser (glm45) never gets to see a closing `</think>`, so it ends up stuffing the body into `reasoning` → pull `content`, and fall back to `reasoning` if it comes back empty.

## Verified as published

15. **Verified as published (2026-09-12, 21:35 and 21:57).** Before tagging this release, I had fresh agents bring up both node pairs (node A = head, node B = worker in each pair) using nothing but the files published in this repo. I hit three bugs in the published scripts right away, all caught by the same test run: (a) under `set -u`, a bare `$var` followed by a full-width `）` in a comment/echo got parsed as part of the variable name and crashed as unbound — the scripts had Japanese text in them, so I stripped it and made everything ASCII; (b) `sync-overlay` choked on pairs still set to `CHANGEME` (a two-node setup only has one pair) — it skips those now; (c) the md5 check branch still looked for the old `serve-*` file name pattern after the scripts were renamed. After those fixes, pair 1 came up cleanly at 21:35: `sync-overlay` 4.9 s, `down` 5.2 s (removed the old containers), `up` 827 s (weights loaded in 9 min 17 s, READY at 13 min 47 s), `status` 2.8 s, `smoke` 4.9 s. Pair 2 followed at 21:57: `sync-overlay` 4.7 s, `down` 0.8 s, `up` 856 s (weights load ~12 min + engine init, READY at 14 min 16 s), `status` 1.3 s, `smoke` 5.2 s — same smoke answer, first line: "DGX Spark is NVIDIA's compact, palm-sized AI supercomputer (formerly known as Project DIGITS) that delivers petaflop-scale GPU computing performance for running large AI models locally on a developer's desk." Both pairs, all four nodes, now run from the published set.

## Lessons from the release test

Running that verification taught me as much as doing the startup itself.

1. **The published scripts failed three times before they passed.** (a) A bare `$var` right before a full-width `）` in a comment got picked up as the variable name under `set -u`; I fixed it by rewriting the text. (b) `sync-overlay` bailed on pairs left as `CHANGEME` — which is what any two-node setup has; I fixed it by having it skip those pairs. (c) The md5 check branch was still looking for the old `serve-*` pattern after I renamed the scripts; I fixed it by updating the pattern. Just three small, ordinary bugs, and I wouldn't have caught any of them if I hadn't replayed the published files from scratch.
2. **Machine-facing text is plain ASCII English now.** I had written Japanese comments and messages in the scripts originally, but bash blew up under `set -u` because of a single full-width parenthesis inside a comment. The shell just treated the byte as part of `$var` right next to it and crashed with an unbound-variable error, which was completely misleading. So I switched everything the shell or another tool parses over to ASCII English. Human-facing reports, including this log, stay in Japanese.
3. **Every rerun started from a clean state.** I dropped all containers, reset the config back to `CHANGEME`, and ran the test again using only the published files. That keeps the passing run honest: it tests the release itself, not leftovers from my local box.
4. **Why go through all this?** If a set of files has only ever run on the author's machines, it is not a release yet — it is a demo. The three failures were mine, not the tools', and blowing away the environment between runs is what turned "works on my machine" into "works". That is also why I kept the timestamps and verbatim error messages in here: so the next person can see exactly where I tripped up, avoid it, and get a running server in one night instead of a week.

## What went wrong

Not just the obstacles — the detours, and what they actually cost me:

1. **Didn't read the official model card first (found it 07:10).** The card lists the official image and the launch command. Instead, I just jumped in with my own DeepSeek-oriented build and only checked the card after six failed boots.
2. **Six startups on a build that could never work.** Parser name broke right away, two Ray session mismatches (~13 min of model loading each before crashing), a three-IP complaint, FP8 KV throwing `pe_dim must be 64`, and BF16 complaining about no backend. Hit wall after wall trying to run GLM on an image baked for DeepSeek's attention shape. Finally gave up at 06:50.
3. **Misread the Ray session mismatch.** On attempt 2, `ActorHandle objects are not valid across Ray sessions` made me think a stale worker was still attached to an old head node, so I tore down and rebuilt both nodes from scratch. Hit the exact same error on attempt 3. The actual issue was Ray executor v2 calling `ray.init()` again inside the EngineCore process (#2).
4. **28 minutes lost to a quoting slip (around 11:50).** The `--kernel-config` JSON needs `\"` inside the double-quoted `bash -lc` string. I messed up the quoting, the container immediately crashed with `vllm serve: error`, and my watcher script didn't detect the exit. The watcher polled an idle state for 28 minutes before I checked the container logs (#12).
5. **179 generations truncated before I noticed the thinking block (13:10).** Every single request in the batch truncated at `max_tokens=1024` with only the English thinking trace in the response body. The chat template injects `<think>` by default and lacks an `enable_thinking` toggle. Only figured out the fix after the batch finished (#14).

## Open question
Which upstream vLLM version runs GLM-5's sparse MLA (BF16, or GLM-shaped FP8) natively on sm_121? Once I know that, I'll know when I can drop this overlay.
