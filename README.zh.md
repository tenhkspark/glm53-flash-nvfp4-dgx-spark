# GLM-5.3-Flash NVFP4 on DGX Spark — a one-file vLLM overlay for no-rope sparse MLA

官方镜像按原样部署时无法在 DGX Spark�(��sm_121）上启动。这个单文件 overlay 让它得以运行，已在四台节点（两对）上完成验证。

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh.md) · [한국어](README.ko.md)

## 为什么公开

一个官方开源模型，应该让任何下载它的人都能真正跑起来，而不是只能在官方验证过的那几种硬件上运行。所以直接把最原始的排查过程记录在这里：哪里挂了、卡在什么地方，以及每一步是怎么解决的，时间戳和具体的错误信息都原封不动保留。希望能帮大家在下次发版时省下一个通宵。

NVIDIA 官方发布了 GLM-5.3-Flash 的 NVFP4 量化版本，同时提供了一个官方 vLLM 镜像（`vllm/vllm-openai:glm53-flash-arm64-cu130`，arm64 + CUDA 13 构建（模型卡的验证硬件是 GB200））。但这个镜像在 DGX Spark�(��计算能力 sm_121）上会直接崩溃，原因是 SM120 sparse-MLA attention 后端缺少支持 no-rope MLA 的代码路径，而 GLM-5.3-Flash 用的恰恰就是这种 attention 结构。

我在这个 repo 里写了一个单文件的 Python overlay，把缺失的代码路径补上了。不需要改动任何 C++/CUDA 代码，也无需重新构建镜像——直接通过 bind-mount 把这个文件挂载覆盖掉原来的文件就行。

## 哪些尝试行不通

我用官方推荐的参数在两台 DGX Spark 节点上拉起了官方镜像（Ray，tensor parallel = 2），刚加载完模型、在做 KV cache profiling 的时候就直接挂了：

```
concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla
```

当我强制加上 `--kv-cache-dtype bfloat16` 之后，报错只是换了个地方出现：

```
No valid attention backend found ... head_size=512, use_mla=True, use_sparse=True
```

简单来说就是：开 FP8 KV 会把 cache writer 搞崩，换成 BF16 又完全找不到能用的 attention backend。试了各种参数组合全都不行。全部 15 个问题都记在 [LOG.md](LOG.md) 里：#1–#14 是我跑通这套环境时踩到的，#15 是后来用发布出来的这套文件做的复现测试。

## 原因
- GLM-5.3-Flash 采用了带 NSA 风格 token 选择索引器的 sparse MLA：attention 只针对每个 query 提取 top-k 个选中的 token，而且 attention 本身是不带 RoPE 运行的——即 `qk_rope_head_dim=0`、`qk_nope_head_dim=256` 以及 `kv_lora_rank=512`。
- 在 compute capability 12.x 上，`vllm/platforms/cuda.py` 会把 MLA backend 的候选范围缩小到 `TRITON_MLA` 和 `FLASHINFER_MLA_SPARSE_SM120`。由于模型本身是 sparse 结构，系统默认选中了 `FlashInferMLASparseSM120Backend`。
- SM100 的 backend（`flashinfer_mla_sparse.py`）实际上能处理这种 no-rope 的 shape，但 SM120 的 backend（`flashinfer_mla_sparse_sm120.py`）却不行：它的 KV 写入路径调用了 `concat_and_cache_mla`，其中的 `fp8_ds_mla` 分支要求必须存在一个 64 维的 rope tensor（`pe_dim must be 64`）——但在没有 RoPE 的情况下，这个 tensor 根本不存在。
- SM120 sparse kernel 本身（即 v32 / GLM-NSA decode 路径）硬性要求 d_qk=576 的 layout（512 latent + 64 rope）。我也没办法直接把 SM100 的方案硬搬过来，因为 SM100 的 no-rope kernel（TRTLLM-GEN）对 sm_121 完全不兼容（`fmhaRunner.cuh` 直接报错 `Unsupported architecture`）。
- 归根结底：这个镜像显然只在具备 no-rope 路径的 GB200 系列硬件（sm_100）上测过。DGX Spark（sm_121）正好卡在两个 backend 之间的盲区里了。

## 修复方法：RoPE 维度补零

这个 kernel 强制要求 576 宽度的 query/key layout，但实际上那 64 个 rope 维度并不需要真正的 rope 数据。所以我做了以下处理：

- **KV side**：我保留了 SM120 默认的 `fp8_ds_mla` cache layout（每个 token 对应 656-byte packed blob），并将其中 64 个 rope 槽位全部填零。在执行 `concat_and_cache_mla` 之前，我的 cache writer override 会把原本宽度为 0 的 `k_pe` tensor 替换成一个宽度为 64 的全零 tensor。
- **Query side**：我把 512 宽度的 latent query 用 64 个零进行了 pad，并给 kernel 传入了 `qk_rope_head_dim=64`。

因为 0 乘以任何数都等于 0，多出来的这些维度在点积运算中贡献全为 0。576 宽度的 kernel 最终算出的 attention 结果跟原生不带 rope 的跑法完全一致——这在数学上是严格等价的，并不是近似计算。至于 fp8 KV scaling（针对 GLM 模型的 `kv_scale_format=arbitrary_fp32`），我保持原样没有改动。

在上服务器之前，我在 sm_121 上用独立的 unit tests 对这条 kernel 路径跑了测试。输出结果与 BF16 reference 对齐，最大绝对误差为 0.0024（纯粹是 fp8 量化噪声），并且无效（`-1`）的 top-k indices 也能正常 mask 掉（误差 0.0031）。

## 仓库结构

```
overlay/flashinfer_mla_sparse_sm120.py   overlay 文件（直接替换即可）
overlay.diff                            同一处改动的 diff 文件
serve/start-head.sh                     Ray head + vLLM 服务端（在节点 A 上跑）
serve/start-worker.sh                   Ray worker（在节点 B 上跑）
serve/glm-serve.sh                      up / down / status / smoke / sync 封装脚本
RUNBOOK.md                              可直接复制粘贴的操作手册：参数值、命令、预期输出、常见故障
LOG.md                                  启动日志：15 个踩坑记录、时间戳
```

## 快速开始

[RUNBOOK.md](RUNBOOK.md) 是一份可以直接复制粘贴的操作手册，供 AI agent（或人工）执行：里面包含你需要填入的参数、按顺序执行的 5 条命令、每一步的预期输出及实测耗时，以及最常见的 5 种故障该如何排查解决。

## 用法

两台 DGX Spark 节点通过直连线连在一起（我走的是 QSFP 端口；只要是两端能直连互通的 IP 都行——脚本里已经把 Ray 和 NCCL 的流量绑定死在这对点对点地址上了）。

1. 两个节点上都要备好权重和镜像。测试基于官方镜像构建验证（vLLM v0.28.1rc1.dev580+g385dce36b，flashinfer 0.6.18）：
   ```bash
   docker pull vllm/vllm-openai:glm53-flash-arm64-cu130
   # NVIDIA GLM-5.3-Flash-NVFP4 权重，例如放在 ~/models/GLM-5.3-Flash-NVFP4
   ```
2. 把 `overlay/flashinfer_mla_sparse_sm120.py` 放到两台机器的相同目录下：`~/recipe-factory/vllm-overlay/`，也就是 `serve/glm-serve.sh` 里 `REMOTE_OVERLAY` 的默认值（或者直接跑 `serve/glm-serve.sh sync-overlay`，它会拷到这个目录）。
3. 先拉起 head 节点，再拉起 worker（填好 `CHANGEME` 的具体参数，或者提前 export 环境变量）：
   ```bash
   # 节点 A (head)
   HEAD_IP=CHANGEME IFNAME=CHANGEME ./serve/start-head.sh
   # 节点 B (worker)
   HEAD_IP=CHANGEME MY_IP=CHANGEME IFNAME=CHANGEME ./serve/start-worker.sh
   ```
   `HEAD_IP` 和 `MY_IP` 是直连链路两端的端点；`IFNAME` 是承载该 IP 的网卡接口。冷启动大概需要 14 分钟（权重有 191 GB；两对节点实测到 READY 分别是 13 min 47 s 和 14 min 16 s）。脚本里把 `VLLM_ENGINE_READY_TIMEOUT_S` 调到了 3600 s，因为默认的 600 s 肯定会超时。
4. 这里的启动顺序不能乱：如果上次运行还残留着旧的 worker 进程，重启 head 前务必先停止它，然后再重新拉起 worker。残留的 worker 尝试加入新 Ray 集群时会直接报 session-mismatch 导致 head 挂掉。`serve/glm-serve.sh up` 已经把这套顺序处理好了（支持 `down` / `up` / `status` / `smoke`）。
5. FlashInfer autotune 已经被显式关掉了（`--kernel-config '{"enable_flashinfer_autotune": false}'`）。在这个环境下保持开启的话，在 autotune dummy pass 阶段 rank 0 会直接崩掉。

### 跳过思考块

GLM-5.3 的 chat template 会强制塞入一个未闭合的 `<think>` 标签（`add_generation_prompt` 会注入 `<|assistant|><think>`），而且没提供 `enable_thinking` 开关。如果把 `max_tokens` 设得比较小，模型会把整个 token 预算全耗在思考块里，导致最终吐出来的正文回复为空。

我采用的绕过方案是追加一个空的 assistant 轮次并开启续写：
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
这样生成会直接紧接着 `<|assistant|><think></think>` 开始，当场闭合 thinking 块，让输出直接切入真正的正文回答（实测发一个 256-token 的请求，能拿到 183 token 纯正文内容）。

需要注意的坑点：由于 `glm45` reasoning parser 一直没碰上闭合标签 `</think>`，它可能会把全部返回内容打上 `reasoning` 标记，导致 `content` 为空。读取逻辑直接优先看 `content`，如果为空就去兜底读 `reasoning` 字段。`serve/glm-serve.sh smoke` 里已经把这个兜底逻辑写进去了。

## 实测（2× DGX Spark，基于 Ray 的 TP=2）

- 引擎初始化耗时 105 s；单节点 KV cache 占用 7.69 GiB（`--max-model-len 131072`，fp8_ds_mla layout）。
- 服务刚起来时压了 48 个请求，并发开到 16：0 失败，机箱温度基本稳定在 64–66 °C。这一轮我没有记录生成条件（请求长度、max_tokens、temperature），所以这里不给出它的 tok/s —— 并发 16 下可比的数字是下面固定标尺里的 69.85 tok/s。
- 随后跑生产环境没做任何改动，直接把并发拉到了 32。

### 原始基线（单流）

2026-09-13 用本仓库的 serve 脚本原样（所有 flag 不变，`--max-model-len 131072`）测得。标尺：同一组 64 条固定提示词（冻结的日文散文种子），每条生成 512 个 completion token，temperature 0，thinking 用上面的空 assistant 续写技巧跳过，流式响应，聚合 tok/s = completion token 总数 ÷ 墙钟时间，各档位 0 次失败。

- 单条依次请求（8 条顺序提示词）：10.75 tok/s，TPOT 中位数 95.3 ms（平均 92.5 ms），TTFT 中位数 0.361 s。
- 聚合吞吐随并发上升后在约 95 tok/s 处走平：并发 16 为 69.85 tok/s，32 为 94.66，64 为 95.08（服务器把同时序列数限制在 32，所以 64 里有一半在排队）。

### 仅改配置的提速（2026-09-13）

把 `--max-model-len` 从模型上限 131072 降到 16384（同日、同一标尺），单流 TPOT 降了约 10%：中位数 95.3 ms → 86.0 ms（复测 85.2 ms），聚合吞吐不变（并发 32 为 94.92 对 94.66 tok/s，64 为 95.63 对 95.08），而且上限调小后 KV 池反而更大。`serve/glm-serve.sh` 和 `serve/start-head.sh` 现在出厂即为 16384：真实提示词远远到不了 131072 个 token，模型上限只是压缩了实际请求共享的 KV 池。

除上限外试过的其他方案（各一句话）：
- MTP 投机解码，原生自动草稿（不带草稿目录）：引擎在把 BF16 草稿 expert 权重装进 NVFP4 打包的 expert 缓冲区时挂掉；显式 BF16 草稿目录同样失败（此镜像的草稿 expert 加载器不做 TP 分片）。
- `--load-format fastsafetensors`：启动 OOM —— 它急切地整块映射 shard，在 128 GB 统一内存的节点上触顶内存监控。
- n-gram 投机 K=4：同样启动 OOM。
- 第一次 16384 尝试在 warmup 崩溃，但那是我的测试启动器漏挂了 worker 侧 overlay，不是这个 flag 的问题 —— 挂载补上后重跑干净启动。

注意：只是一个下午、一种工作负载（日文散文、512 token 输出）的结果。

## 耗时

从 2026-09-12 后半夜第一次尝试拉起服务，到当天晚些时候正式跑上线（以下均为本地时间）：

| Time | Milestone |
| --- | --- |
| small hours | 走直连把权重拷了过来；尝试拉起我自己的构建版本 |
| 06:50 | 连续 6 次启动失败后，我放弃了自己构建的版本——正好卡在「哪些尝试行不通」一节详述的问题上 |
| 07:10 | 在 model card 上看到了官方镜像 |
| 07:40 | 确认官方镜像碰到了完全一样的问题 (TP=2) |
| 07:40–11:30 | 排查出 root cause，拼出 overlay，在 sm_121 上写了 5 个独立的单测并跑通 |
| 11:30–11:50 | overlay 第 1、2 次启动尝试均告失败 |
| 12:32 | overlay 第 3 次尝试：服务终于拉起来了（engine 初始化耗时 105 s） |
| 12:35–12:47 | 跑验证：48 个请求，并发 16，0 失败 |
| ~12:50 | 挂上训练语料，以 32 并发正式起跑生产任务 |

总共尝试启动了这么多次：我自己的构建版本 6 次，官方镜像 3 次，overlay 3 次——最终靠 overlay 第 3 次跑通。等待耗时以及每次挂掉的具体位置，我都记录在 [LOG.md](LOG.md) 里面了。

## 制作方式
- **Planning and verification** — GLM-5.3 (Z.ai)：协助规划步骤、检查执行结果。
- **Implementation** — Z.ai 应用里的 GLM-5.3-Flash：做 overlay 设计时我把 reasoning effort 设为 Low，写 `serve/glm-serve.sh` 包装脚本时调成了 High。
- **Research and triage** — GLM-5.3 (Z.ai)：翻官方文档排查，帮忙锁定了几个根本原因。
- **Hardware** — 4 台 DGX Spark 节点�(��，分成两组，每组节点之间走 QSFP 直连。
- **Software** — 官方 vLLM 镜像（`vllm/vllm-openai:glm53-flash-arm64-cu130`），外加我自己写的 `serve/glm-serve.sh` 脚本。
- 策略决策和工具之间的内容搬运由我来做，实现和测试交给各模型完成。
- 我这里没有使用任何第三方配方、补丁或分支。所有内容均直接来自官方渠道：NVIDIA 模型卡和镜像、该镜像中已自带的 vLLM 源代码，以及 Z.ai 的文档。

## 日志

完整的启动日志我保存在 [LOG.md](LOG.md) 里了。里面按我踩坑的先后顺序记录了全部 15 个问题，附带了时间戳以及原始错误信息。这基本上就是我记录各个 serve 脚本中每个 flag 为何如此配置的排查过程。

## 致谢

感谢 Z.ai 将 GLM-5.3-Flash 作为开源模型发布，也感谢 NVIDIA 发布了官方 NVFP4 权重以及运行本方案所用的官方 vLLM 镜像。
