# 起動ログ — 2× DGX Spark 上の GLM-5.3-Flash NVFP4 (2026-09-12)

2台の DGX Spark ノード（Ray、TP=2）上で nvidia/GLM-5.3-Flash-NVFP4 を立ち上げる際に発生したトラブルのタイムラインです。12:32 にサーバーが正常稼働するまでの全記録となります。エラーメッセージは原文ママです。時刻は現地時間です。

使用資材: 公式の NVFP4 チェックポイント、公式の vLLM イメージ `vllm/vllm-openai:glm53-flash-arm64-cu130`、および2ノード間の直結 QSFP リンク経由で Ray TP=2 を用いて vLLM を実行する私の serve スクリプト。

## Time spent

サーバーが1台正常に起動するまでに、計12回の試行を要しました。内訳は、独自ビルドで6回、公式イメージで3回、オーバーレイで3回です（オーバーレイの3回目でようやく起動しました）。
初期設定のチェックを通過した実行はいずれも、重み（191 GB）のロードだけで約13分を消費しました。以下の待ち時間は概算です。

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

1. **Missing parser name.** このビルドには `--tool-call-parser glm47_moe` が含まれていませんでした（`glm45` と `glm47` のみ存在）。今回は生成処理のみが必要なため、ツールパーサーは外し、`--reasoning-parser glm45` のみを維持しました。

2. **Ray session mismatch.** モデルのロードが100%に達した直後に以下のエラーが発生しました。
```
ActorHandle objects are not valid across Ray sessions (job 01000000 → 02000000)
```
vLLM v1 は EngineCore を別プロセスで起動し、その中で独自の `ray.init()` を実行するため、完全に新しいセッションになってしまいます。`VLLM_ENABLE_V1_MULTIPROCESSING=0` を設定しても解決しませんでした（Ray executor のパス上では EngineCore はプロセス外のままです）。原因は Ray executor v2 (`ray_executor_v2.py`) にあり、`VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0` を設定して旧 executor を強制的に使用するようにしました。ビルドは vLLM v0.28.1rc1.dev559+g28a2ccee7 です。最初、停止した head に残った古い worker がぶら下がっているのが原因だと誤診し、すべて最初からリビルドしたため、同じ問題に再度直面することになりました。

3. **Nodes cannot resolve each other's names.** ノード名は私のノート PC の `/etc/hosts` にしか存在しないため、node A から `rsync nodeB:` を実行すると失敗します。直接接続している QSFP リンクのポイントツーポイント IP アドレスを直接指定することで解決しました。SSH 鍵は一方のノードからのみフォワードしています。

4. **`pkill -f` で自分のセッションが停止。** `ssh nodeA 'pkill -f "rsync -a --partial"; ...'` を実行したところ、この文字列が ssh 呼び出し自身のコマンドラインにも含まれていたためパターンが自分自身にマッチし、自分のセッションが停止しました。

5. **Copy timings.** QSFP リンク経由での 191 GB の重み転送は、350 MB/s でノードあたり約10分でした。31 GB のイメージは `docker save | ssh docker load` 経由で転送しました。

6. **Three IPs.** 旧 executor を使用したところ、Ray から以下のエラーが出力されました。
```
Every node should have a unique IP address ... 3 unique IP addresses
```
head 側スクリプトで `VLLM_HOST_IP` の export を `ray start` の*後*に行っていたため、head のローカル Ray worker のみがノート PC 向けネットワークの IP を通知していました。この export を `ray start` の前に移動することで修正しました。worker 側のスクリプトは最初から正しい順序になっていました。

7. **KV cache dtype.** 分散初期化とロードを通過した後、プロファイリング時に `concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla` でクラッシュしました。このイメージは DeepSeek 向けにビルドされており、KV cache のデフォルトが fp8_ds_mla になっているため、GLM の MLA 形状と一致しません。`--kv-cache-dtype auto` を試しましたが（5回目の試行）、チェックポイントの `hf_quant_config.json` に `kv_cache_quant_algo: FP8` が指定されており、`auto` ではそのまま FP8 KV が選択されるため、まったく同じエラーで落ちました。次に、明示的に `--kv-cache-dtype bfloat16` を渡す必要がありました（6回目の試行）。

8. **The wall.** bfloat16 で実行したところ、以下のエラーが出ました。
```
No valid attention backend found ... head_size=512, use_mla=True, use_sparse=True
```
GLM-5.3-Flash はスパース MLA（DSA スタイル）を使用していますが、このビルドのスパース MLA カーネルは FP8 KV かつ DeepSeek の pe_dim=64 を前提としています。FP8 は pe_dim で落ち、BF16 にはバックエンドが存在しないため、どのフラグの組み合わせでも通りません。GLM-5 のスパースアテンションには、それを実際にサポートしている vLLM が必要です。ここでその日の作業を切り上げました（06:50）。

9. **Should have read the model card first (07:10).** nvidia/GLM-5.3-Flash-NVFP4 のモデルカードには、公式イメージ `vllm/vllm-openai:glm53-flash-arm64-cu130`（arm64、CUDA 13、DGX Spark 向け）と公式の起動コマンド（`--kv-cache-dtype fp8 --reasoning-parser glm45 --enable-expert-parallel --enable-ep-weight-filter --max-num-batched-tokens 8192 --enable-chunked-prefill`、TP=4）が明記されていました。自分で用意した DeepSeek 向けビルドでの6回の起動失敗を経て、ようやく気づきました。公式の重みには公式のランタイムが同梱されているという教訓です。

10. **The official image hits the same wall (07:40).** `vllm/vllm-openai:glm53-flash-arm64-cu130`（vLLM v0.28.1rc1.dev580+g385dce36b）上で TP=2 で実行したところ、まったく同一の `pe_dim must be 64 for fp8_ds_mla` が発生しました。根本原因は、`vllm/platforms/cuda.py` が compute capability 12.x（DGX Spark = 12.1）での MLA 候補を `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]` に制限しており、モデルがスパースであるため `FlashInferMLASparseSM120Backend` が選択される点にあります。SM100 バックエンド (`flashinfer_mla_sparse.py`) はネイティブの no-rope MLA（`qk_rope_head_dim=0`、nope 256、kv_lora_rank 512）を明示的にサポートしていますが、SM120 側にはその分岐が完全に欠落しており、KV の書き込みが `concat_and_cache_mla`（fp8_ds_mla、pe_dim=64 がハードコードされている）にフォールバックしてしまいます。このイメージは検証済みハードウェア（GB200, sm_100）では動作しますが、DGX Spark（sm_121）は見落とされています。解決策は、SM120 バックエンドに no-rope のパスを追加することだけです。

11. **Breakthrough.** `flashinfer_mla_sparse_sm120.py` 向けに no-rope 分岐を作成しました（K 側の 64 個の rope スロットをゼロ埋めし、query を 512 から 576 へゼロパディング）。sm_121 上で5つの単体テストを実行したところ、BF16 リファレンスとの比較で最大誤差は 0.0024 でした。Python ファイル1つのみの修正で、C++/CUDA の変更は一切不要でした。これをバインドマウントのオーバーレイとして公式イメージにマウントし、`serve/` 内のスクリプトを使って起動しました。

12. **Overlay startup (11:30–11:50).**
- (a) パッチ未適用の worker が実行中のまま head を再起動したため、Ray session mismatch が再発しました（#2 と同じ問題）。正常に動作する順序は、worker を停止 → head を停止 → worker を起動、です。
- (b) モデルのロード、プロファイリング、KV サイズ決定、グラフキャプチャは正常に完了しましたが、FlashInfer SM120 スパース MLA decode の autotuning 中（mixed-batch のダミー実行時）に rank 0 が停止しました。`--kernel-config '{"enable_flashinfer_autotune": false}'` で autotuning を無効化し、`VLLM_ENGINE_READY_TIMEOUT_S=3600` に引き上げました（デフォルトは 600 s ですが、起動に13分かかるため大幅な余裕が必要です）。
**12:32: 起動完了。** エンジンの初期化には 105 秒かかり、ノードあたりの KV cache は 7.69 GiB となりました。DGX Spark に関するプロンプトを日本語で送信したところ、正しい思考プロセス�(��128 GB、Project DIGITS の系譜）が返ってきました。ゼロ埋めしたスパース MLA パスが単体テストだけでなく実稼働環境でも動作することを確認しました。
- 補足: `--kernel-config` の JSON は、ダブルクォートで囲まれた `bash -lc` 文字列の内部で `\"` を使用してエスケープする必要があります。クォートの指定を誤った際、`vllm serve: error` でコンテナが即座に終了したものの、ウォッチャーがその終了を検知できず、アイドル状態を28分間ポーリングし続けました。

13. **Numbers (12:35–12:47, node A / node B, 16 concurrent).** 48 リクエスト処理、失敗 0 件、合計 61.5 tok/s、温度は 64–66 °C でした。文章題の数学の問題を自己検証も含めてエラーなく解くことができました。直後に本番実行のため並列度を 32 に引き上げました。

14. **Thinking ate the body (13:10).** 179 リクエストを実行したところ、全リクエストが `max_tokens=1024` で途切れ、本文として（英語の）思考トレースのみが返ってくる状態になりました。GLM-5.3 の `chat_template.jinja` は `add_generation_prompt` が有効な場合、自動的に `<think>` を注入し、`enable_thinking` のような切り替えオプションはありません。メッセージリストの末尾を空の assistant メッセージで終端させ、`continue_final_message: true, add_generation_prompt: false` を設定することで回避しました。これにより生成が `<|assistant|><think></think>` の直後から開始され、思考プロセスを完全にバイパスして日本語の本文のみ（183 トークン）を取得できるようになりました。副作用として、推論パーサー（glm45）が閉じタグの `</think>` を検出できなくなるため、本文を `reasoning` に格納してしまいます。そのため、`content` を取得し、それが空の場合は `reasoning` にフォールバックする処理を入れる必要があります。

## Verified as published

15. **Verified as published (2026-09-12, 21:35 and 21:57).** このリリースにタグを打つ前に、私は新規エージェントに、本リポジトリで公開されているファイルのみを使って両ノードペア（それぞれ node A = head、node B = worker）を起動させました。その際、同一テストの実行で公開スクリプト内の3件のバグを直ちに検知しました。(a) `set -u` の環境下で、コメントやecho内のむき出しの `$var` の直後に全角の `）` があると変数名の一部としてパースされてしまい、unboundとしてクラッシュしました。スクリプト内に日本語の記述が含まれていたため、私はそれらを除去してすべてASCIIに統一しました。(b) `sync-overlay` がまだ `CHANGEME` に設定されているペアで停止してしまいました（2ノード構成には1ペアしかありません）。現在はこれらをスキップするように修正しています。(c) スクリプトの名前を変更した後も、md5チェックの分岐が古い `serve-*` というファイル名パターンを参照し続けていました。修正後、ペア1は 21:35 にクリーンに起動しました。`sync-overlay` 4.9 s、`down` 5.2 s（旧コンテナを削除）、`up` 827 s（重みのロード 9 分 17 秒、13 分 47 秒で READY）、`status` 2.8 s、`smoke` 4.9 s。ペア2は 21:57 に続きました。`sync-overlay` 4.7 s、`down` 0.8 s、`up` 856 s（重みのロード約12分 + エンジン初期化、14分16秒で READY）、`status` 1.3 s、`smoke` 5.2 s。smoke の応答は同じで、1行目は次のとおりです："DGX Spark is NVIDIA's compact, palm-sized AI supercomputer (formerly known as Project DIGITS) that delivers petaflop-scale GPU computing performance for running large AI models locally on a developer's desk." これで両ペア、4台すべてのノードが公開一式で動いています。

## Lessons from the release test

その検証を実行したことで、起動作業そのものを実施したのと同じくらい多くの学びが得られました。

1. **公開されたスクリプトは、成功するまでに3回失敗しました。** (a) コメント内の全角の `）` の直前にある裸の `$var` が、`set -u` 環境下で変数名の一部として認識されてしまいました。テキストを書き直すことで修正しました。(b) `sync-overlay` が `CHANGEME` のまま残されているペア（どのような2ノード構成にも存在するものです）で異常終了しました。それらのペアをスキップするように処理を変更して修正しました。(c) スクリプト名を変更した後も、md5チェックの分岐が古い `serve-*` パターンを探し続けていました。パターンを更新することで修正しました。どれも些細でよくあるバグに過ぎませんが、公開されたファイルを最初から再生して確認していなければ、いずれも気付けなかったはずです。

2. **Machine-facing text is plain ASCII English now.** 当初はスクリプト内のコメントやメッセージを日本語で書いていましたが、コメント内の全角括弧が原因で `set -u` 配下の bash が異常終了しました。シェルがそのバイト列を直前の `$var` の一部として解釈し、unbound-variable エラーでクラッシュしてしまい、完全に誤認を招く挙動でした。そのため、シェルや他のツールがパースするテキストはすべて ASCII の英語に切り替えました。この記録を含む人間向けのレポートについては、引き続き日本語のままとしています。

3. **Every rerun started from a clean state.** すべてのコンテナを破棄し、設定を `CHANGEME` にリセットした上で、公開されたファイルのみを使って再度テストを実行しました。これにより、成功した実行結果の信頼性を担保しています。つまり、ローカル環境の残骸ではなく、リリースそのものをテストしている状態を保てます。

4. **Why go through all this?** 一連のファイルが作者のマシン上でしか動作したことがないなら、それはまだリリースではなく、単なるデモです。3回の失敗はツールの問題ではなく私のミスであり、実行ごとに環境を完全に破棄したからこそ、「私のマシンでは動く」を「どこでも動く」に変えることができました。タイムスタンプと一言一句違わぬエラーメッセージをここにそのまま残しているのもそれが理由です。次に作業する人が、私がどこでつまずいたのかを正確に把握して回避し、1週間ではなく一晩で稼働するサーバーを立ち上げられるようにするためです。

## What went wrong

障害だけでなく、回り道とそこで要した時間の記録です。

1. **最初に公式の model card を読んでいなかった（07:10 に確認）。**
   model card には公式イメージと起動コマンドが記載されていました。それにもかかわらず、私は DeepSeek 向けに自作したビルドで作業を始めてしまい、起動に6回失敗した段階で、ようやく card を確認しました。

2. **どうやっても動作しないビルドで6回起動を試行した。**
   parser 名の解決エラーが即座に発生し、Ray セッションの不一致が2回（モデルロードに毎回約13分かかった後にクラッシュ）、IPが3つあることによる警告、FP8 KV による `pe_dim must be 64` の送出、さらに BF16 でバックエンドが存在しないというエラーが出ました。DeepSeek の attention shape 向けに作成されたイメージで GLM を動かそうとしたため、次々と問題に直面しました。最終的に 06:50 に断念しました。

3. **Ray セッション不一致の原因を取り違えた。**
   2回目の試行時、`ActorHandle objects are not valid across Ray sessions` というエラーから、古い head ノードに古い worker がまだ接続されたままになっていると判断し、両方のノードをゼロから破棄・再構築しました。しかし、3回目の試行でも全く同じエラーが発生しました。実際の問題は、Ray executor v2 が EngineCore プロセス内で再度 `ray.init()` を呼び出していたことでした（#2）。

4. **クォートのエスケープミスで28分の損失（11:50頃）。**
   ダブルクォートで囲んだ `bash -lc` 文字列の内部では、`--kernel-config` の JSON に `\"` を使用する必要があります。このクォート処理を誤ったため、コンテナは `vllm serve: error` で即座にクラッシュしましたが、私の監視スクリプトがその終了を検知できませんでした。コンテナログを確認するまで、何もしないまま28分間ポーリングを続けてしまいました（#12）。

5. **thinking ブロックに気づくまで179件の生成が思考だけで終わった（13:10）。**
   バッチ内のすべてのリクエストが `max_tokens=1024` で切り捨てられ、レスポンスボディには英語の thinking トレースしか残っていませんでした。チャットテンプレートがデフォルトで `<think>` を挿入する仕様になっており、`enable_thinking` の切り替えトグルも存在しませんでした。修正方法が判明したのは、バッチ処理が完了した後でした（#14）。

## 2026-09-13 — 素のベースライン速度と設定だけの高速化

- 出荷時のままの serve スクリプトで素のベースラインを測定しました: 単一ストリームで 10.75 tok/s（TPOT 中央値 95.3 ms、TTFT 中央値 0.361 s）、同時 32〜64 で集計約 95 tok/s に頭打ち。計測条件: 固定プロンプト 64 本、各 512 completion トークン、temperature 0、ストリーミング、失敗 0 件。
- 設定だけの改善: `--max-model-len` を 131072 → 16384 に下げると単一ストリームの TPOT が約 10 % 改善（中央値 95.3 ms → 86.0 ms、再測定 85.2 ms）し、集計スループットは変わらなかったため、`serve/glm-serve.sh` は 16384 を出荷値にしています。
- それ以外に試したものはすべて起動またはロードに失敗しました: MTP 推測デコーディング（ネイティブ自動ドラフトと明示 BF16 ドラフト dir）、fastsafetensors ローダー、n-gram K=4。数値と各アームの一言は [README.ja.md](README.ja.md) を参照。

## Open question

どのアップストリームの vLLM バージョンであれば、sm_121 上で GLM-5 の sparse MLA（BF16、または GLM 形状の FP8）をネイティブに実行できるでしょうか。それが分かれば、私がこのオーバーレイをいつ外せるかも判断できます。
