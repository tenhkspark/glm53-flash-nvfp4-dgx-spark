# 启动日志 — 2× DGX Spark 上部署 GLM-5.3-Flash NVFP4 (2026-09-12)

记录在两台 DGX Spark 节点上拉起 nvidia/GLM-5.3-Flash-NVFP4（Ray, TP=2）过程中踩的所有坑，最终在 12:32 成功跑通服务。错误信息均为原文摘录。时间为本地时间。

物料：官方 NVFP4 checkpoint、官方 vLLM 镜像 `vllm/vllm-openai:glm53-flash-arm64-cu130`，以及我写的服务启动脚本（通过两台节点之间的直连 QSFP 链路运行带 Ray TP=2 的 vLLM）。

## 耗时统计

我一共试了 12 次才把一台可用服务器拉起来：自己 build 的版本试了 6 次，官方镜像试了 3 次，overlay 方案试了 3 次（最后第 3 次终于成了）。只要配置校验通过，光是加载权重（191 GB）每次就要耗掉约 13 分钟。下面的等待时间均为粗略估计。

| 方案 | 次数 | 挂掉前的等待时间 | 挂掉的阶段 |
| --- | --- | --- | --- |
| 自己 build | 1 | 几秒 | 参数解析阶段 — 当前 build 不支持 `--tool-call-parser glm47_moe` (#1) |
| 自己 build | 2 | ~13 分钟 | 权重刚加载到 100% 就报 Ray session mismatch (#2) |
| 自己 build | 3 | ~13 分钟 | 重新全量 build 后依然报同样的错（属于误诊，#2） |
| 自己 build | 4 | 几分钟 | Ray worker 初始化 — `Every node should have a unique IP address ... 3 unique IP addresses` (#6) |
| 自己 build | 5 | ~13 分钟 | KV-cache profiling 阶段 — 带 `--kv-cache-dtype auto` 时报 `pe_dim must be 64 for fp8_ds_mla` (#7) |
| 自己 build | 6 | 引擎初始化 | backend 选择阶段 — 用 BF16 时报 `No valid attention backend found` (#8)；06:50 放弃 |
| 官方镜像 | 1–3 | 每次 ~13 分钟 | 加载完毕后：FP8 KV → 依然是 `pe_dim must be 64` (07:40, #10)；BF16 → 无可用 backend。试了各种参数组合都没跑通 |
| overlay | 1 | ~1 分钟 | worker 加入阶段 — 又是 Ray session mismatch，用了未打补丁的残留 worker (#12a) |
| overlay | 2 | ~14 分钟 | graph capture 之后 — rank 0 在 FlashInfer autotune dummy run 中挂掉 (#12b) |
| overlay | 3 | 28 分钟耗在空轮询上，随后跑通 | `--kernel-config` 引号转义漏了 → 秒报 `vllm serve: error`；修好后 → **12:32 成功启动** |

1. **Missing parser name.** 这个 build 里没有 `--tool-call-parser glm47_moe`（只有 `glm45` / `glm47`）。因为我这里只需要做生成，所以直接去掉了 tool parser，只保留 `--reasoning-parser glm45`。

2. **Ray session mismatch.** 模型加载刚跑到 100% 就报错：
```
ActorHandle objects are not valid across Ray sessions (job 01000000 → 02000000)
```
vLLM v1 会在单独的进程中拉起 EngineCore，而它会执行自己的 `ray.init()`——这就产生了一个全新的 session。设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 没用（在 Ray executor 路径下 EngineCore 依然是跨进程的）。问题出在 Ray executor v2 (`ray_executor_v2.py`)；配置 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0` 可以强制走老版本 executor。Build 版本：vLLM v0.28.1rc1.dev559+g28a2ccee7。我一开始误判是有残留的 worker 连着死掉的 head，从头重建了一遍环境，结果立马踩回同一个坑。

3. **Nodes cannot resolve each other's names.** 节点名称只写在我笔记本的 `/etc/hosts` 里，所以在 node A 上跑 `rsync nodeB:` 会失败。我直接改用直连 QSFP 链路的点对点 IP 地址搞定。SSH key 只从其中一个节点做了转发。

4. **`pkill -f` 终止了我自己的 session。** 执行 `ssh nodeA 'pkill -f "rsync -a --partial"; ...'` 时，这个字符串也出现在 ssh 调用自身的命令行里，pattern 匹配到了自己，当前 session 随之被终止。

5. **Copy timings.** 通过 QSFP 链路传输 191 GB 权重：速度 350 MB/s，每个节点约 10 分钟。31 GB 的镜像通过 `docker save | ssh docker load` 同步。

6. **Three IPs.** 切回老版本 executor 后，Ray 报错：
```
Every node should have a unique IP address ... 3 unique IP addresses
```
我的 head 脚本在 `ray start` *之后* 才 export `VLLM_HOST_IP`，导致只有 head 本地的 Ray worker 上报了笔记本所在网段的 IP。把 export 移到 `ray start` 之前就解决了。worker 脚本里的顺序本来就是对的。

7. **KV cache dtype.** 过了分布式初始化和加载阶段后，profiling 阶段崩溃并报错 `concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla`。这个镜像是为 DeepSeek 打的，默认将 KV cache 设为 fp8_ds_mla，与 GLM 的 MLA 维度不匹配。尝试 `--kv-cache-dtype auto`（第 5 次尝试）炸得一模一样：checkpoint 的 `hf_quant_config.json` 里指定了 `kv_cache_quant_algo: FP8`，`auto` 就顺着直接走了 FP8 KV。接着我只能显式传入 `--kv-cache-dtype bfloat16`（第 6 次尝试）。

8. **The wall.** 跑 bfloat16 时报错：
```
No valid attention backend found ... head_size=512, use_mla=True, use_sparse=True
```
GLM-5.3-Flash 使用 sparse MLA（DSA 风格）；当前 build 的 sparse MLA kernel 默认基于 FP8 KV 以及 DeepSeek 的 pe_dim=64。FP8 卡在 pe_dim，BF16 没有可用 backend——试遍所有参数组合都跑不通。GLM-5 的 sparse attention 必须依赖真正支持它的 vLLM 版本。折腾到这里我决定先睡觉（06:50）。

9. **Should have read the model card first (07:10).** nvidia/GLM-5.3-Flash-NVFP4 的 model card 上明确指定了官方镜像 `vllm/vllm-openai:glm53-flash-arm64-cu130`（arm64，CUDA 13——针对 DGX Spark）以及官方启动命令（`--kv-cache-dtype fp8 --reasoning-parser glm45 --enable-expert-parallel --enable-ep-weight-filter --max-num-batched-tokens 8192 --enable-chunked-prefill`，TP=4）。我自己拿着适配 DeepSeek 的 build 踩了 6 次启动失败才注意到这点。教训：官方权重必须搭配官方 runtime 一起用。

10. **The official image hits the same wall (07:40).** 在 `vllm/vllm-openai:glm53-flash-arm64-cu130`（vLLM v0.28.1rc1.dev580+g385dce36b）上跑 TP=2，报出了完全相同的 `pe_dim must be 64 for fp8_ds_mla`。根本原因：`vllm/platforms/cuda.py` 把算力 12.x（DGX Spark = 12.1）的 MLA 候选限制在 `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`，又因为该模型是 sparse 架构，它最终选中了 `FlashInferMLASparseSM120Backend`。SM100 的 backend (`flashinfer_mla_sparse.py`) 明确支持原生 no-rope MLA（`qk_rope_head_dim=0`，nope 256，kv_lora_rank 512）；而 SM120 的实现完全缺失了这个分支，所以其 KV 写入直接掉进了 `concat_and_cache_mla`（fp8_ds_mla，硬编码了 pe_dim=64）。这个镜像在它验证过的硬件（GB200，sm_100）上能跑；DGX Spark（sm_121）正好成了漏网之鱼。唯一的解决办法是给 SM120 backend 补上 no-rope 分支。

11. **Breakthrough.** 我为 `flashinfer_mla_sparse_sm120.py` 补写了 no-rope 分支（在 K 侧将 64 个 rope 槽位补零，将 query 从 512 零填充到 576）。在 sm_121 上跑了 5 组独立测试：对比 BF16 参考实现，最大误差仅 0.0024。只改动了一个 Python 文件，没有任何 C++/CUDA 修改。我把它通过 bind-mount overlay 挂进官方镜像，用 `serve/` 下的脚本启动服务。

12. **Overlay startup (11:30–11:50).**
- (a) 在未打补丁的 worker 仍在运行的情况下我重启了 head → 再次踩中 Ray session mismatch（和第 2 点同样的问题）。正确的操作顺序：先 kill worker，再 kill head，最后启动 worker。
- (b) 模型加载、profiling、KV 分配和 graph capture 都顺利通过，但 rank 0 在 FlashInfer SM120 sparse MLA decode autotuning（mixed-batch dummy run 阶段）挂掉了。我通过 `--kernel-config '{"enable_flashinfer_autotune": false}'` 禁用了 autotuning，并把 `VLLM_ENGINE_READY_TIMEOUT_S=3600` 调大（默认 600 s；这个 13 分钟的启动过程需要宽裕得多的超时窗口）。**12:32: 服务就绪。** 引擎初始化耗时 105 s，每个节点占用 7.69 GiB KV cache。发了一个关于 DGX Spark 的日文 prompt，正常拿到了正确的 thinking 回复（提到 DGX Spark、128 GB、Project DIGITS 渊源）——零填充的 sparse MLA 分支在实际生产链路中验证可用，不仅仅是单测能跑。
- 另外：`--kernel-config` 里的 JSON 在双引号包裹的 `bash -lc` 字符串内必须使用 `\"` 转义。我转义写错了一次，`vllm serve: error` 导致容器瞬间退出，而监控脚本没捕捉到退出状态——对着空转的容器轮询了 28 分钟。

13. **Numbers (12:35–12:47, node A / node B, 16 concurrent).** 48 个请求，0 失败，总吞吐 61.5 tok/s，温度保持在 64–66 °C。完整解出了一道文字表述的数学题，包含自我校验步骤，毫无差错。随后直接将并发切到 32 跑正式生产。

14. **Thinking ate the body (13:10).** 跑了 179 个请求，全部在 `max_tokens=1024` 处被截断，返回的实际正文全被截断成了（英文）thinking trace。GLM-5.3 的 `chat_template.jinja` 只要开启 `add_generation_prompt` 就会自动注入 `<think>`；并没有配置 `enable_thinking` 开关。我的临时方案：在 message 列表末尾追加一条空的 assistant 消息，并设置 `continue_final_message: true, add_generation_prompt: false`——这样生成直接从 `<|assistant|><think></think>` 后面开始，彻底跳过了思考过程（纯输出 183 个 token 的日文正文）。副作用：reasoning parser (glm45) 始终匹配不到闭合的 `</think>`，导致它把正文塞进了 `reasoning` 字段 → 解析时优先读取 `content`，如果为空再 fallback 到 `reasoning`。

## Verified as published

15. **Verified as published (2026-09-12, 21:35 and 21:57).** 在给这个版本打 tag 之前，我让全新的 agent 仅使用本仓库发布的这套文件去拉起两对节点（每对都是 node A = head、node B = worker）。刚跑起来我就在发布的脚本里踩了三个 bug，都是在同一轮测试中抓出来的：(a) 在 `set -u` 下，注释或 echo 里的裸 `$var` 紧跟着全角 `）`，结果被解析成了变量名的一部分，直接报 unbound 崩溃——脚本里之前混了日文，我全删了改成了纯 ASCII；(b) `sync-overlay` 在处理仍然设为 `CHANGEME` 的配对时挂了（两节点配置只有一对）——现在已经加了跳过逻辑；(c) 脚本重命名之后，md5 check 分支里还在找旧的 `serve-*` 文件名模式。修完这些之后，第 1 对在 21:35 干净利落地跑了起来：`sync-overlay` 4.9 s、`down` 5.2 s（清掉了旧容器）、`up` 827 s（权重加载 9 min 17 s，13 min 47 s 达到 READY）、`status` 2.8 s、`smoke` 4.9 s。第 2 对在 21:57 跟上：`sync-overlay` 4.7 s、`down` 0.8 s、`up` 856 s（权重加载约 12 min + 引擎初始化，14 min 16 s 达到 READY）、`status` 1.3 s、`smoke` 5.2 s。smoke 的回答相同，第一行是："DGX Spark is NVIDIA's compact, palm-sized AI supercomputer (formerly known as Project DIGITS) that delivers petaflop-scale GPU computing performance for running large AI models locally on a developer's desk." 现在两对节点——全部四台——都跑在发布的一套文件上。

## Lessons from the release test

跑这遍验证让我学到的东西，不亚于做启动本身。

1. **发布出去的脚本挂了三次才跑通。** (a) 注释里一个紧挨着全角 `）` 的裸 `$var`，在 `set -u` 下被误当成了变量名；我重写了那段说明把它修掉了。(b) `sync-overlay` 遇到保持默认 `CHANGEME` 的配置对直接退出了——而任何双节点环境都是这么配的；我改成让它直接跳过这类配置对。(c) 重命名脚本后，md5 检查分支还在找旧的 `serve-*` 匹配模式；我更新了匹配规则把它修掉了。不过是三个很不起眼的普通 bug，但要不是我把发布的文件从零完整重放了一遍，根本一个都抓不到。

2. **面向机器的文本现已全部采用纯 ASCII 英文。** 我原本在脚本里写了日文注释和提示信息，但在启用 `set -u` 后，bash 却因为注释里的一个全角括号直接报错崩溃。shell 把紧挨着它的字节当成了 `$var` 的一部分，报出完全具有误导性的 unbound-variable 错误并退出了。因此，我把所有需要交给 shell 或其他工具解析的内容都换成了 ASCII 英文。面向人员看的报告（包括这份记录）则保留日文。

3. **每次重跑都从干净的状态开始。** 我删除了所有容器，把配置重置回 `CHANGEME`，并且只用已发布的文件重新跑测试。这样能保证跑通的结果是真实可信的：测的是发布版本本身，而不是我本地机器上的残留环境。

4. **为什么要折腾这些？** 如果一套文件只在作者自己的机器上跑过，那它还算不上发布版本——充其量是个 demo。那三次失败是我自己的问题，跟工具无关；而每次重跑前把环境彻底推倒重来，才让"在我机器上能跑"变成了真正的"能跑"。这也是为什么我把时间戳和一字不差的错误信息都留在了这里：这样下一个人就能清楚看到我究竟在哪踩了坑，绕开它，用一晚而不是一周的时间把服务器跑起来。

## What went wrong

不光是遇到的阻碍，更多是走的弯路，以及这些弯路实际让我付出的代价：

1. **一开始没先看官方 model card（07:10 才发现）。** 卡片里其实明确写了官方镜像和启动命令。但我直接拿自己之前跑 DeepSeek 的镜像去启动，直到连挂了 6 次启动，才去看了一眼 card。
2. **在一个跑不通的镜像上试了 6 次启动。** 一上来就报 parser name 挂掉；接着是两次 Ray session mismatch（每次光加载模型就要等约 13 分钟，加载完立马崩溃）；报过三 IP 冲突问题；开 FP8 KV 时报 `pe_dim must be 64`；切到 BF16 又报没有可用 backend。拿针对 DeepSeek attention shape 构建的镜像去跑 GLM，结果四处碰壁。06:50 放弃。
3. **把 Ray session mismatch 的原因误判了。** 第 2 次尝试时，报了 `ActorHandle objects are not valid across Ray sessions`，我以为是之前的残留 worker 还挂在旧的 head node 上，于是把两台节点的进程彻底清空重拉了一遍。第 3 次启动时原样报错。其实根本原因是在 EngineCore 进程内部，Ray executor v2 又去调了一次 `ray.init()`（#2）。
4. **引号转义写错损失了 28 分钟（大概在 11:50）。** `--kernel-config` 传的 JSON 字符串在最外层双引号的 `bash -lc` 里需要加 `\"` 转义。我引号写挂了，容器启动直接报 `vllm serve: error` 秒退，而我的监控脚本又没抓到容器退出状态。监控脚本空转轮询了 28 分钟，我才去翻容器日志（#12）。
5. **跑了 179 次生成才发现全卡在思考块里了（13:10）。** 整个 batch 里的每一次请求全都在 `max_tokens=1024` 被截断，返回的响应内容里全都是英文的思考过程（thinking trace）。因为该模型的 chat template 默认注入了 `<think>`，而且没有提供 `enable_thinking` 开关。直到整个 batch 全跑完，我才摸索出怎么解决（#14）。

## Open question

哪个上游 vLLM 版本能在 sm_121 上原生运行 GLM-5 的稀疏 MLA（BF16 或 GLM 规格的 FP8）？一旦搞清楚这点，我就知道什么时候可以去掉这个 overlay 了。
