# DGX SparkにおけるGLM-5.3-Flash NVFP4 — no-rope sparse MLA向け単一ファイルvLLMオーバーレイ

公開状態の公式イメージは、DGX Spark�(��sm_121）ではそのまま起動しません。この単一ファイルのオーバーレイで動くようになり、4台のノード（2ペア）で検証済みです。

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh.md) · [한국어](README.ko.md)

## なぜ公開するのか

公式に公開されているオープンモデルは、検証環境となった特定のハードウェアだけでなく、ダウンロードした誰もが実際に動かせなければなりません。そのため、何が壊れ、どこで壁に当たり、各ステップをどのように修正したのかについて、タイムスタンプや正確なエラーメッセージをそのまま残した生の記録をここに残しておきます。これが次のリリースで誰かの徹夜を防ぐのに役立てば幸いです。

NVIDIAは、公式のvLLMイメージ（`vllm/vllm-openai:glm53-flash-arm64-cu130`、arm64 + CUDA 13 ビルド（モデルカードの検証機は GB200））とともに、GLM-5.3-Flashの公式NVFP4量子化を提供しています。しかし、このイメージはDGX Spark�(��コンピュート・ケイパビリティ sm_121）上でクラッシュします。原因は、SM120 sparse-MLAアテンションバックエンドにno-rope MLA用のコードパスが存在せず、これがまさにGLM-5.3-Flashで使用されているアテンション形状であるためです。

そこで私は、不足しているパスをパッチ適用する単一ファイルのPythonオーバーレイを本リポジトリにまとめました。C++/CUDAの変更は一切なく、イメージをリビルドする必要もありません。対象のファイルに本ファイルをバインドマウントして置き換えるだけで動作します。

## 動かなかったもの

公式フラグを使用して2台のDGX Sparkノード間で公式イメージを起動（Ray、tensor parallel = 2）したところ、ロード直後のKVキャッシュのプロファイリング中に停止しました。

```
concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla
```

`--kv-cache-dtype bfloat16` を強制指定したところ、エラーの発生箇所が変わりました。

```
No valid attention backend found ... head_size=512, use_mla=True, use_sparse=True
```

要するに、FP8 KVではキャッシュライターがクラッシュし、BF16では対応するバックエンドが全く存在しない状態になります。フラグをどのように組み合わせても動作しませんでした。起動を試みる中で直面した全14件の事象については、[LOG.md](LOG.md) に記録しました。

## 原因
- GLM-5.3-Flash は NSA スタイルのトークン選択インデクサーを備えた sparse MLA を採用しています。アテンションはクエリごとに上位 k 個の選択されたトークンのみを取得し、アテンション処理自体は no-rope（`qk_rope_head_dim=0`、`qk_nope_head_dim=256`、`kv_lora_rank=512`）で動作します。
- コンピュート機能 12.x では、`vllm/platforms/cuda.py` によって MLA バックエンドの選択肢が `TRITON_MLA` と `FLASHINFER_MLA_SPARSE_SM120` に絞り込まれます。モデルが sparse であるため、デフォルトで `FlashInferMLASparseSM120Backend` が選択されます。
- SM100 バックエンド（`flashinfer_mla_sparse.py`）はこの no-rope の形状に対応していますが、SM120 バックエンド（`flashinfer_mla_sparse_sm120.py`）は対応していません。その KV 書き込みパスが `concat_and_cache_mla` を呼び出し、そこで `fp8_ds_mla` ブランチが 64 次元の rope テンソル（`pe_dim must be 64`）を想定していますが、rope なし構成ではそのテンソルが存在しません。
- SM120 sparse カーネル自体（v32 / GLM-NSA デコードパス）は、d_qk=576（512 latent + 64 rope）のレイアウトを厳密に要求します。また、SM100 の no-rope カーネル（TRTLLM-GEN）は sm_121 を完全に拒否するため（`fmhaRunner.cuh` で `Unsupported architecture`）、SM100 のルートをそのまま移植することも私にはできません。
- 結論として、このイメージは明らかに no-rope パスが存在する GB200 クラスのハードウェア（sm_100）でテストされたものと考えられます。DGX Spark（sm_121）は両バックエンドのちょうど狭間に取り残されてしまっています。

## 修正内容: RoPE次元のゼロ埋め

カーネルにより576幅のquery/keyレイアウトが強制されますが、この64次元のRoPE領域には実際にはRoPEデータを載せる必要がありません。そのため、私は以下の対応を取りました。

- **KV側**: SM120デフォルトの `fp8_ds_mla` キャッシュレイアウト（トークンあたり656バイトのパックされたblob）を維持し、64個のRoPEスロットをゼロ埋めしました。自作のキャッシュ書き込みオーバーライド処理により、`concat_and_cache_mla` の直前で幅0の `k_pe` テンソルを幅64のゼロテンソルに差し替えています。
- **Query側**: 512幅のlatent queryに64個のゼロをパディングし、カーネルに `qk_rope_head_dim=64` を渡しました。

ゼロに何を掛けてもゼロになるため、これらの余分な次元が内積に与える影響は0です。結果として、576幅のカーネルはネイティブなRoPEなしの実行と完全に同一のアテンションを計算します。これは近似ではなく、数学的に等価です。

fp8 KVスケーリング（GLMモデル向けの `kv_scale_format=arbitrary_fp32`）はそのままにしています。

サーバー実機に反映する前に、単体テストを実行してsm_121上でこのカーネルパスを検証しました。出力はBF16のリファレンスと一致しており、最大絶対誤差は0.0024（単なるfp8量子化ノイズ）でした。また、無効な（`-1`）top-kインデックスも正常にマスク処理されています（0.0031）。

## リポジトリの構成

```
overlay/flashinfer_mla_sparse_sm120.py  オーバーレイ（そのまま置き換えて使用）
overlay.diff                            同等の変更内容をまとめたdiff
serve/start-head.sh                     Ray head + vLLM サーバー（ノードAで実行）
serve/start-worker.sh                   Ray worker（ノードBで実行）
serve/glm-serve.sh                      up / down / status / smoke / sync のラッパースクリプト
RUNBOOK.md                              コピペ用ランブック：設定値、コマンド、想定される出力、よくある障害
LOG.md                                  起動ログ：15件のつまずきポイント、タイムスタンプ
```

## クイックスタート

[RUNBOOK.md](RUNBOOK.md) は、AIエージェント（または人間）が実行するためのコピー＆ペースト可能なランブックです。具体的には、設定が必要な値、順番に実行する5つのコマンド、各ステップの実測所要時間を含む想定出力、そして最もよくある5つの失敗への対処方法をまとめています。

## 使い方

2台のDGX Sparkノードを直結リンクで接続しています（私はQSFPポートを使用していますが、直接疎通可能なIPのペアであれば何でも動作します。スクリプト側でRayとNCCLのトラフィックをこれらのpoint-to-pointアドレスに固定しています）。

1. 両ノードに重みとイメージを用意します。公式イメージ（vLLM v0.28.1rc1.dev580+g385dce36b, flashinfer 0.6.18）をベースにビルド・検証しました。
```bash
docker pull vllm/vllm-openai:glm53-flash-arm64-cu130
# NVIDIA GLM-5.3-Flash-NVFP4 checkpoint, e.g. under ~/models/GLM-5.3-Flash-NVFP4
```

2. `overlay/flashinfer_mla_sparse_sm120.py` を両ノードの同一ディレクトリ（例: `~/vllm-overlay/`）に配置します（あるいは `serve/glm-serve.sh sync-overlay` を実行します）。

3. headを起動した後にworkerを起動します（`CHANGEME` の値を埋めるか、事前にexportしておきます）。
```bash
# node A (head)
HEAD_IP=CHANGEME IFNAME=CHANGEME ./serve/start-head.sh
# node B (worker)
HEAD_IP=CHANGEME MY_IP=CHANGEME IFNAME=CHANGEME ./serve/start-worker.sh
```
`HEAD_IP` と `MY_IP` はpoint-to-pointリンクのエンドポイントです。`IFNAME` はそのIPが割り当てられているインターフェースです。コールドスタートには約13分かかります（重み191 GB）。デフォルトの600秒ではタイムアウトするため、スクリプト側で `VLLM_ENGINE_READY_TIMEOUT_S` を3600秒に引き上げています。

4. 起動順序に注意が必要です。前回の実行時の古いworkerが残っている場合は、headを再起動する前に必ず停止させ、その後にworkerを再起動します。残存したworkerが新しいRayクラスタに参加しようとすると、session-mismatchエラーでheadがクラッシュします。`serve/glm-serve.sh up` を使えば、この一連の手順（`down` / `up` / `status` / `smoke`）が自動で処理されます。

5. FlashInferのautotuneは明示的に無効化しています（`--kernel-config '{"enable_flashinfer_autotune": false}'`）。この構成では、autotuneを有効のままにするとダミーパスの実行中にrank 0がクラッシュしました。

### 思考ブロックのスキップ

GLM-5.3のチャットテンプレートは `<think>` タグを強制的に開始し（`add_generation_prompt` が `<|assistant|><think>` を挿入します）、`enable_thinking` のトグルも用意されていません。`max_tokens` を小さく設定していると、モデルが思考ブロック内ですべてのトークン予算を使い切ってしまい、レスポンスが空になってしまいます。

私が採用している回避策は、空のassistantターンを追加して続きの生成（continuation）を要求する方法です。
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
これにより、生成が `<|assistant|><think></think>` の直後から開始される形となり、思考ブロックが即座に閉じられて実際の回答テキストが直接出力されます（256トークンのリクエストでテストしたところ、183トークンの回答テキストがそのまま得られました）。

注意点: `glm45` のreasoningパーサーが終了タグ `</think>` を検出しないため、本文全体が `reasoning` として扱われ、`content` が空のままになることがあります。その場合は `content` を参照し、空であれば `reasoning` を取得するようにします。`serve/glm-serve.sh smoke` ではこのフォールバック処理を既に対応済みです。

## 実測（2× DGX Spark、Ray 経由の TP=2）
- Engine init に 105 s かかりました。KV cache はノードあたり 7.69 GiB となりました（`--max-model-len 131072`、fp8_ds_mla layout）。
- 並列数 16 で 48 リクエストを実行しました。失敗は 0 件、全体で 61.5 tok/s となり、筐体温度はおよそ 64–66 °C で推移しました。
- その後の本番実行に向けて、設定は一切変更せずに並列数を 32 に引き上げました。

## かかった時間

2026-09-12 の深夜に最初の起動を試みてから、同日午後の本番実行に至るまでの記録です（時刻はすべて現地時間）。

| Time | Milestone |
| --- | --- |
| small hours | ダイレクトリンク経由で重みをコピーし、独自ビルドの起動を試行 |
| 06:50 | 6回起動に失敗したところで独自ビルドを断念（「動かなかったもの」に記載した問題に直面） |
| 07:10 | モデルカード上で公式イメージを発見 |
| 07:40 | 公式イメージでも全く同じ問題（TP=2）が発生することを確認 |
| 07:40–11:30 | 根本原因を特定し、オーバーレイを作成。sm_121 上で5つの単体テストを作成して実行 |
| 11:30–11:50 | オーバーレイの起動試行1回目と2回目が失敗 |
| 12:32 | オーバーレイ試行3回目：サーバーがようやく起動（engine init に 105 s） |
| 12:35–12:47 | 検証を実行：48 requests、concurrency 16、全体で 61.5 tok/s を記録 |
| ~12:50 | 32 concurrent で学習コーパスの本番実行を開始 |

起動試行回数の合計は、独自ビルドで6回、公式イメージで3回、オーバーレイで3回でした。最終的にオーバーレイの3回目で動作しました。待ち時間と各失敗の停止箇所については [LOG.md](LOG.md) に記録してあります。

## どのように作ったか
- **Planning and verification** — GLM-5.3 (Z.ai): 手順の計画を補助し、結果を確認しました。
- **Implementation** — Z.ai アプリ内の GLM-5.3-Flash: オーバーレイを介した設計には reasoning effort を Low で使用し、`serve/glm-serve.sh` ラッパーの作成時には High に引き上げました。
- **Research and triage** — GLM-5.3 (Z.ai): 公式ドキュメントを調査し、根本原因を絞り込みました。
- **Hardware** — 4台の DGX Spark ノード�(��。2ペアに分割し、各ペア間を直接 QSFP リンクで接続しました。
- **Software** — 公式 vLLM イメージ（`vllm/vllm-openai:glm53-flash-arm64-cu130`）に加え、自作の `serve/glm-serve.sh` スクリプトを使用しました。
- 私自身は方針の判断とツール間での受け渡しを行い、実装とテストは各モデルに任せました。
- 私はここではサードパーティ製のレシピやパッチ、フォークなどは一切使用していません。すべて公式ソース（NVIDIAのモデルカードとイメージ、そのイメージ内にあらかじめ含まれていたvLLMのソースコード、Z.aiのドキュメント）から直接利用しています。

## ログ

起動ログ全体は [LOG.md](LOG.md) に残してあります。私が遭遇した順に全15件の問題が、タイムスタンプおよび生のエラーメッセージとともに記載されています。実質的には、serve スクリプト内の各フラグがなぜそのように設定されているのかを示す、私の作業記録となっています。

## 謝辞

GLM-5.3-Flash をオープンモデルとしてリリースしてくれた Z.ai、ならびに公式 NVFP4 チェックポイントおよび動作対象の公式 vLLM イメージを公開してくれた NVIDIA に感謝します。
