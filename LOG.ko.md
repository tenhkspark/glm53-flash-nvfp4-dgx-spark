# Startup log — GLM-5.3-Flash NVFP4 on 2× DGX Spark (2026-09-12)

DGX Spark 2개 노드(Ray, TP=2) 환경에서 nvidia/GLM-5.3-Flash-NVFP4 구동 중 발생한 문제들과 12:32에 정상 서빙에 성공하기까지의 타임라인을 정리합니다. 오류 메시지는 원문 그대로 기록하며, 시간은 로컬 기준입니다.

사용 환경: 공식 NVFP4 체크포인트, 공식 vLLM 이미지 `vllm/vllm-openai:glm53-flash-arm64-cu130`, 두 노드 간 직결 QSFP 링크를 통해 Ray TP=2로 vLLM을 띄우는 나의 serve 스크립트.

## Time spent

정상 구동되는 서버 1대를 띄우기까지 12번의 시도가 필요했습니다. 자체 빌드에서 6번, 공식 이미지에서 3번, 오버레이에서 3번 시도 끝에 세 번째 시도에서 마침내 올라왔습니다. 초기 설정 검사를 통과한 시도들은 가중치(191 GB)를 로드하는 데만 매번 약 13분을 소모했습니다. 아래 대기 시간은 대략적인 추정치입니다.

| Route | # | Wait before it fell | Stage where it fell |
| --- | --- | --- | --- |
| own build | 1 | seconds | 인자 파싱 — 이 빌드에 `--tool-call-parser glm47_moe` 없음 (#1) |
| own build | 2 | ~13 min | 로딩이 100%에 도달한 직후 Ray 세션 불일치 발생 (#2) |
| own build | 3 | ~13 min | 전체 재빌드 후 동일한 오류 재발 (원인 오진, #2) |
| own build | 4 | minutes | Ray 워커 초기화 — `Every node should have a unique IP address ... 3 unique IP addresses` (#6) |
| own build | 5 | ~13 min | KV 캐시 프로파일링 — `--kv-cache-dtype auto` 설정 시 `pe_dim must be 64 for fp8_ds_mla` (#7) |
| own build | 6 | engine init | 백엔드 선택 — BF16 설정 시 `No valid attention backend found` (#8); 06:50에 중단 |
| official image | 1–3 | ~13 min each | 로딩 후: FP8 KV → 동일한 `pe_dim must be 64` 발생 (07:40, #10); BF16 → 유효한 백엔드 없음. 어떤 플래그 조합도 통과하지 못함 |
| overlay | 1 | ~1 min | 워커 조인 — 패치되지 않은 이전 워커 잔재로 인한 Ray 세션 불일치 재발 (#12a) |
| overlay | 2 | ~14 min | 그래프 캡처 후 — FlashInfer autotune 더미 실행 중 rank 0 종료 (#12b) |
| overlay | 3 | 28 min lost, then up | `--kernel-config` 따옴표 누락 → 즉시 `vllm serve: error` 발생; 수정 완료 → **12:32 up** |

1. **Missing parser name.** `--tool-call-parser glm47_moe` 옵션이 이 빌드에는 없습니다(`glm45` / `glm47`만 존재). 이번 작업에서는 텍스트 생성만 필요하므로 tool parser는 제거하고 `--reasoning-parser glm45`만 유지했습니다.

2. **Ray session mismatch.** 모델 로딩이 100% 완료된 직후 아래 에러가 발생했습니다.
```
ActorHandle objects are not valid across Ray sessions (job 01000000 → 02000000)
```
vLLM v1은 별도 프로세스로 EngineCore를 띄우는데, 여기서 자체적으로 `ray.init()`을 다시 호출하면서 완전히 새로운 세션이 생성됩니다. `VLLM_ENABLE_V1_MULTIPROCESSING=0`을 설정해도 해결되지 않았습니다(Ray executor 경로에서는 EngineCore가 계속 별도 프로세스로 유지됨). 원인은 Ray executor v2(`ray_executor_v2.py`)에 있었으며, `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0`을 설정해 구형 executor를 강제로 사용하도록 변경했습니다. 빌드 버전은 vLLM v0.28.1rc1.dev559+g28a2ccee7입니다. 처음에는 죽은 head에 stale worker가 물려 있는 문제로 잘못 진단하여 처음부터 전부 재빌드했으나 동일한 문제에 다시 부딪혔습니다.

3. **Nodes cannot resolve each other's names.** 노드 이름이 내 노트북의 `/etc/hosts`에만 등록되어 있어 node A에서 `rsync nodeB:`를 실행하면 실패합니다. 다이렉트 QSFP 링크의 point-to-point IP 주소를 직접 지정하여 해결했습니다. SSH 키는 노드 중 한 곳에서만 포워딩됩니다.

4. **`pkill -f`로 내 세션이 종료됨.** `ssh nodeA 'pkill -f "rsync -a --partial"; ...'` 명령을 실행했더니, ssh 호출 인자에 패턴 문자열이 그대로 포함되어 있어 자기 자신의 커맨드 라인과 매칭되면서 내 세션 자체가 종료되었습니다.

5. **Copy timings.** QSFP 링크를 통해 191 GB 가중치 전송 시: 350 MB/s, 노드당 약 10분 소요되었습니다. 31 GB 이미지는 `docker save | ssh docker load`로 전송했습니다.

6. **Three IPs.** 구형 executor를 사용하자 Ray에서 다음 에러를 뱉었습니다.
```
Every node should have a unique IP address ... 3 unique IP addresses
```
내 head 스크립트에서 `ray start` *이후*에 `VLLM_HOST_IP`를 export하고 있었기 때문에, head의 로컬 Ray 워커들만 노트북 네트워크 IP를 공표하고 있었습니다. 해당 export를 `ray start` 이전으로 옮겨 해결했습니다. worker 스크립트는 이미 순서가 올바르게 되어 있었습니다.

7. **KV cache dtype.** 분산 초기화와 로딩을 통과한 후, 프로파일링 단계에서 `concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla` 에러와 함께 크래시가 발생했습니다. 이 이미지는 DeepSeek용으로 빌드되어 KV 캐시 기본값이 fp8_ds_mla로 설정되어 있었고, 이는 GLM의 MLA shape와 맞지 않습니다. `--kv-cache-dtype auto`를 적용해 보았으나(5번째 시도) 완전히 동일하게 실패했습니다. 체크포인트의 `hf_quant_config.json`에 `kv_cache_quant_algo: FP8`이 지정되어 있어 `auto` 설정 시 그대로 FP8 KV를 따라가기 때문입니다. 다음에는 `--kv-cache-dtype bfloat16`을 명시적으로 전달해야 했습니다(6번째 시도).

8. **The wall.** bfloat16으로 실행 시 다음 에러가 발생했습니다.
```
No valid attention backend found ... head_size=512, use_mla=True, use_sparse=True
```
GLM-5.3-Flash는 sparse MLA(DSA 스타일)를 사용합니다. 이 빌드의 sparse MLA 커널은 FP8 KV와 DeepSeek의 pe_dim=64를 전제로 작성되어 있습니다. FP8은 pe_dim에서 터지고, BF16은 지원하는 백엔드가 없어 어떤 플래그 조합으로도 통과할 수 없습니다. GLM-5의 sparse attention을 온전히 지원하는 vLLM이 필요한 상황입니다. 여기서 당일 작업을 마감했습니다(06:50).

9. **Should have read the model card first (07:10).** nvidia/GLM-5.3-Flash-NVFP4 모델 카드에 공식 이미지 `vllm/vllm-openai:glm53-flash-arm64-cu130`(arm64, CUDA 13—DGX Spark 타깃)과 공식 실행 명령(`--kv-cache-dtype fp8 --reasoning-parser glm45 --enable-expert-parallel --enable-ep-weight-filter --max-num-batched-tokens 8192 --enable-chunked-prefill`, TP=4)이 명시되어 있었습니다. DeepSeek 타깃 빌드로 6번이나 기동 실패를 겪고 나서야 이를 확인했습니다. 공식 가중치에는 그에 맞는 공식 런타임이 함께 제공된다는 점을 재확인했습니다.

10. **The official image hits the same wall (07:40).** `vllm/vllm-openai:glm53-flash-arm64-cu130`(vLLM v0.28.1rc1.dev580+g385dce36b) 환경에서 TP=2로 실행하자 동일한 `pe_dim must be 64 for fp8_ds_mla` 에러가 발생했습니다. 원인: `vllm/platforms/cuda.py`가 capability 12.x(DGX Spark = 12.1) 환경에서 MLA 후보를 `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`으로 제한하고, 모델이 sparse 구조이기 때문에 `FlashInferMLASparseSM120Backend`를 선택합니다. SM100 백엔드(`flashinfer_mla_sparse.py`)는 native no-rope MLA(`qk_rope_head_dim=0`, nope 256, kv_lora_rank 512)를 명시적으로 지원하지만, SM120 백엔드에는 해당 브랜치가 전혀 없어 KV 쓰기 작업이 `concat_and_cache_mla`(fp8_ds_mla, pe_dim=64 하드코딩)로 빠집니다. 즉 검증 대상 하드웨어(GB200, sm_100)에서는 정상 동작하지만, DGX Spark(sm_121)는 지원 누락 상태입니다. SM120 백엔드에 no-rope 경로를 추가하는 것 외에는 해결 방법이 없습니다.

11. **Breakthrough.** `flashinfer_mla_sparse_sm120.py`에 no-rope 브랜치를 직접 작성했습니다(K 측의 64 rope 슬롯을 0으로 채우고, 쿼리를 512에서 576으로 제로 패딩). sm_121 환경에서 5개의 단독 테스트를 수행한 결과 BF16 레퍼런스 대비 최대 오차는 0.0024였습니다. C++/CUDA 수정 없이 파이썬 파일 단 하나만 작업했습니다. 이를 공식 이미지에 bind-mount overlay 방식으로 마운트한 뒤 `serve/`에 있는 스크립트로 띄웠습니다.

12. **Overlay startup (11:30–11:50).**
- (a) 패치되지 않은 worker가 실행 중인 상태에서 head를 재시작하자 Ray session mismatch가 다시 발생했습니다(#2와 동일한 문제). 정상적인 재시작 순서: worker 종료 → head 종료 → worker 시작 순입니다.
- (b) 모델 로딩, 프로파일링, KV 사이징, 그래프 캡처는 정상 통과했으나, rank 0이 FlashInfer SM120 sparse MLA decode autotuning 단계(mixed-batch dummy run 중)에서 종료되었습니다. `--kernel-config '{"enable_flashinfer_autotune": false}'`로 autotuning을 비활성화하고, `VLLM_ENGINE_READY_TIMEOUT_S=3600`으로 늘려주었습니다(기본값 600초이나 기동에 13분이 걸려 여유가 필요함).
**12:32: 기동 완료.** 엔진 초기화에 105초 소요되었고, 노드당 7.69 GiB의 KV cache를 확보했습니다. DGX Spark 관련 질의를 일본어로 전송해 정상적인 사고 과정 응답(128 GB, Project DIGITS 계보)을 수신했습니다. 제로 패딩 처리한 sparse MLA 경로가 단순 단위 테스트를 넘어 실제 환경에서 정상 동작함을 확인했습니다.
- 추가 참고: `--kernel-config` JSON은 double-quoted `bash -lc` 문자열 내부에서 `\"`로 이스케이프해야 합니다. 따옴표 처리가 한 번 꼬이면서 `vllm serve: error`와 함께 컨테이너가 즉시 내려갔는데, 감시 스크립트가 컨테이너 종료를 감지하지 못하고 유휴 상태를 28분 동안 폴링했습니다.

13. **Numbers (12:35–12:47, node A / node B, 16 concurrent).** 48개 요청 처리, 실패 0건, aggregate 61.5 tok/s, 온도는 64–66 °C를 유지했습니다. 서술형 수학 문제를 자체 검증 단계까지 포함해 오류 없이 풀어냈습니다. 직후 프로덕션 런을 위해 동시 요청 수를 32로 상향 조정했습니다.

14. **Thinking ate the body (13:10).** 179개 요청을 돌렸으나 전량 `max_tokens=1024`에서 잘렸으며, 본문 대신 (영문) thinking trace만 출력되었습니다. GLM-5.3의 `chat_template.jinja`는 `add_generation_prompt`가 켜져 있으면 무조건 `<think>`를 삽입하며, 별도의 `enable_thinking` 토글이 존재하지 않습니다. 우회책으로 메시지 리스트 끝에 빈 assistant 메시지를 넣고 `continue_final_message: true, add_generation_prompt: false`로 설정했습니다. 이렇게 하면 `<|assistant|><think></think>` 바로 뒤부터 생성이 시작되어 사고 과정을 완전히 건너뜁니다(일본어 본문 텍스트만 183 토큰 정상 출력). 부작용으로 reasoning parser(glm45)가 닫는 태그인 `</think>`를 전달받지 못해 본문 내용을 `reasoning` 필드에 전부 밀어 넣습니다. 따라서 `content`를 먼저 읽고, 비어 있을 경우 `reasoning`을 폴백으로 읽도록 처리했습니다.

## Verified as published

15. **Verified as published (2026-09-12, 21:35 and 21:57).** 이번 릴리스에 태그를 달기 전에, 나는 새로운 에이전트들이 이 리포지토리에 공개된 파일들만을 사용해 두 노드 쌍(각 쌍에서 node A = head, node B = worker)을 띄우도록 했습니다. 배포된 스크립트에서 곧바로 세 가지 버그를 발견했으며, 모두 동일한 테스트 실행 중에 잡혔습니다: (a) `set -u` 환경에서 주석/echo 내의 단독 `$var` 바로 뒤에 붙은 전각 `）` 문자가 변수명의 일부로 파싱되어 unbound 에러로 비정상 종료되었습니다 — 스크립트에 일본어 텍스트가 포함되어 있었기 때문에, 이를 제거하고 모두 ASCII로 변경했습니다; (b) `sync-overlay`가 여전히 `CHANGEME`로 설정되어 있는 쌍(2노드 환경에서는 쌍이 하나만 존재함)에서 중단되었습니다 — 이제는 이를 건너뜁니다; (c) md5 확인 브랜치에서 스크립트 이름이 변경된 이후에도 여전히 이전 `serve-*` 파일명 패턴을 찾고 있었습니다. 수정 작업을 마친 후, 첫 번째 쌍은 21:35에 문제없이 기동했습니다: `sync-overlay` 4.9 s, `down` 5.2 s (기존 컨테이너 제거), `up` 827 s (가중치 로드 9 min 17 s, 13 min 47 s에 READY), `status` 2.8 s, `smoke` 4.9 s. 두 번째 쌍은 21:57에 이어졌습니다: `sync-overlay` 4.7 s, `down` 0.8 s, `up` 856 s (가중치 로드 약 12 min + 엔진 초기화, 14 min 16 s에 READY), `status` 1.3 s, `smoke` 5.2 s. smoke 응답은 동일했으며, 첫 줄: "DGX Spark is NVIDIA's compact, palm-sized AI supercomputer (formerly known as Project DIGITS) that delivers petaflop-scale GPU computing performance for running large AI models locally on a developer's desk." 이제 두 쌍, 네 대의 노드 전부가 공개된 파일 세트로 구동됩니다.

## Lessons from the release test

이번 검증을 직접 돌려본 것이, 시작 작업 자체만큼이나 많은 것을 가르쳐 주었습니다.

1. **공개된 스크립트는 통과하기까지 세 번 실패했습니다.** (a) 주석 내 전각 `）` 바로 앞에 있던 단순 `$var`가 `set -u` 환경에서 변수명으로 인식되었습니다. 해당 텍스트를 수정하여 해결했습니다. (b) 2노드 구성이라면 기본적으로 존재하는 `CHANGEME` 상태의 쌍에서 `sync-overlay`가 중단되었습니다. 해당 쌍들을 건너뛰도록 처리하여 해결했습니다. (c) 스크립트 이름을 변경한 이후에도 md5 확인 브랜치가 여전히 기존의 `serve-*` 패턴을 찾고 있었습니다. 패턴을 업데이트하여 수정했습니다. 작고 평범한 버그 3개에 불과했지만, 공개된 파일을 처음부터 다시 재실행해 보지 않았다면 하나도 잡지 못했을 것입니다.

2. **Machine-facing text is plain ASCII English now.** 원래는 스크립트 내 주석과 메시지를 일본어로 작성했으나, 주석 안의 전각 괄호 하나 때문에 `set -u` 환경에서 bash 오류가 발생했습니다. 쉘이 해당 바이트를 바로 옆에 있던 `$var`의 일부로 인식하면서 unbound-variable 오류를 내며 비정상 종료되었고, 이는 원인을 파악하기에 매우 혼란스러웠습니다. 이에 따라 쉘이나 다른 도구가 파싱하는 모든 텍스트를 ASCII 영어로 전환했습니다. 이 로그를 포함하여 사람이 직접 확인하는 리포트는 계속 일본어로 유지합니다.

3. **Every rerun started from a clean state.** 나는 모든 컨테이너를 내리고, 설정을 다시 `CHANGEME`로 초기화한 뒤, 배포된 파일만을 사용해 테스트를 재실행했습니다. 이렇게 해야 통과 결과에 거짓이 없습니다. 로컬 머신의 잔재가 아니라 릴리스 자체를 검증하는 것이기 때문입니다.

4. **Why go through all this?** 특정 파일 세트가 작성자의 머신에서만 돌아간 적이 있다면, 그것은 아직 릴리스가 아니라 데모일 뿐입니다. 3번의 실패는 도구가 아니라 내 탓이었고, 실행 사이에 환경을 완전히 날려버리는 과정을 거쳤기에 "내 머신에서는 된다"가 진짜 "된다"로 바뀔 수 있었습니다. 타임스탬프와 원문 그대로의 오류 메시지를 여기에 남겨둔 이유도 바로 이 때문입니다. 다음 작업자가 내가 정확히 어디서 걸려 넘어졌는지 확인하고, 이를 피해 일주일이 아니라 하룻밤 만에 정상 작동하는 서버를 띄울 수 있도록 하기 위함입니다.

## What went wrong

장애뿐 아니라 어떤 우회로를 탔고, 각각 시간이 얼마나 들었는지 정리합니다.

1. **공식 모델 카드를 먼저 확인하지 않음 (07:10에 발견).** 모델 카드에 공식 이미지와 실행 커맨드가 이미 나와 있었습니다. 하지만 나는 내가 쓰던 DeepSeek 전용 빌드로 작업을 이어갔고, 6번의 부팅 실패 끝에야 카드를 확인했습니다.
2. **원천적으로 동작 불가능한 빌드로 6번이나 부팅 시도.** 파서 이름부터 즉시 터졌고, Ray 세션 불일치 2회(매번 모델 로딩에 약 13분씩 소요된 후 크래시), 3개 IP 관련 에러, FP8 KV의 `pe_dim must be 64` 에러, BF16의 백엔드 부재 에러가 연달아 발생했습니다. DeepSeek의 어텐션 구조에 맞춰 빌드된 이미지로 GLM을 띄우려다 온갖 에러에 부딪혔고, 결국 06:50에 포기했습니다.
3. **Ray 세션 불일치 에러 오판.** 2번째 시도에서 `ActorHandle objects are not valid across Ray sessions`가 발생했을 때, 기존 헤드 노드에 잔여 워커가 여전히 붙어 있다고 생각해 양쪽 노드를 바닥부터 다시 밀고 재구축했습니다. 3번째 시도에서도 똑같은 에러가 났습니다. 실제 원인은 Ray executor v2가 EngineCore 프로세스 내부에서 `ray.init()`을 다시 호출한 것이었습니다 (#2).
4. **따옴표 이스케이프 실수로 잃은 28분 (11:50경).** `--kernel-config` JSON은 큰따옴표로 둘러싸인 `bash -lc` 문자열 내부에서 `\"` 처리가 필요합니다. 이 따옴표 처리를 잘못해서 컨테이너가 `vllm serve: error`를 뱉으며 즉시 크래시되었으나, 와처 스크립트가 컨테이너 종료를 감지하지 못했습니다. 컨테이너 로그를 확인하기 전까지 28분 동안 유휴 상태를 폴링했습니다 (#12).
5. **생각 블록(thinking block)을 뒤늦게 발견해 179개 생성 작업이 모두 잘림 (13:10).** 배치 작업의 모든 요청이 `max_tokens=1024`에서 잘렸고, 응답 본문에는 영어로 된 생각 과정(thinking trace)만 남아 있었습니다. 챗 템플릿이 기본값으로 `<think>`를 주입하는데 `enable_thinking` 토글이 없었던 탓입니다. 배치가 완전히 끝난 뒤에야 해결책을 파악했습니다 (#14).

## 2026-09-13 — 기본 베이스라인 속도와 설정만으로의 속도 향상

- 출고 상태 그대로의 serve 스크립트로 기본 베이스라인을 측정했습니다: 단일 스트림 10.75 tok/s(TPOT 중앙값 95.3 ms, TTFT 중앙값 0.361 s), 동시 32~64에서 집계 약 95 tok/s로 정체. 측정 조건: 고정 프롬프트 64개, 각 512 completion 토큰, temperature 0, 스트리밍, 실패 0건.
- 설정만으로 얻은 개선: `--max-model-len`을 131072 → 16384로 낮추자 단일 스트림 TPOT이 약 10 % 개선되었고(중앙값 95.3 ms → 86.0 ms, 반복 측정 85.2 ms), 집계 처리량은 변하지 않아 `serve/glm-serve.sh`는 이제 16384를 출고값으로 합니다.
- 그 외에 시도한 것들은 모두 부팅 또는 적재에 실패했습니다: MTP 추측 디코딩(네이티브 자동 드래프트와 명시적 BF16 드래프트 디렉터리), fastsafetensors 로더, n-gram K=4. 수치와 항목별 한 줄 요약은 [README.ko.md](README.ko.md)를 참고하세요.

## Open question

어느 업스트림 vLLM 버전이 sm_121에서 GLM-5의 sparse MLA(BF16 또는 GLM 형태의 FP8)를 네이티브로 실행할 수 있는지 확인이 필요합니다. 이를 파악하면 이 오버레이를 언제 제거할 수 있을지 판단할 수 있습니다.
