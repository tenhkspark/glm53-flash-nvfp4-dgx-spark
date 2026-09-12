# DGX Spark 환경의 GLM-5.3-Flash NVFP4 — no-rope sparse MLA를 위한 단일 파일 vLLM 오버레이

공식 이미지는 기본 상태로는 DGX Spark(sm_121)에서 시작되지 않습니다. 이 단일 파일 오버레이가 이를 동작하게 만들었으며, 4대의 노드(두 쌍)에서 검증했습니다.

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh.md) · [한국어](README.ko.md)

## 왜 공개하는가

공식 오픈 모델이라면 검증을 거친 특정 하드웨어뿐만 아니라, 다운로드한 누구의 환경에서든 실제로 돌아가야 합니다. 그래서 무엇이 깨졌고 어디서 막혔는지, 그리고 타임스탬프와 정확한 오류 메시지를 그대로 남기며 각 단계를 어떻게 해결했는지 가감 없이 이곳에 정리해 둡니다. 다음 릴리스 때 누군가의 밤샘 작업을 줄여줄 수 있기를 바랍니다.

NVIDIA는 공식 vLLM 이미지(`vllm/vllm-openai:glm53-flash-arm64-cu130`, arm64 + CUDA 13 빌드(모델 카드의 검증 하드웨어는 GB200))와 함께 GLM-5.3-Flash의 공식 NVFP4 양자화 버전을 배포합니다. 그러나 DGX Spark(컴퓨팅 기능 sm_121)에서 해당 이미지가 비정상 종료됩니다. SM120 sparse-MLA 어텐션 백엔드에 GLM-5.3-Flash가 사용하는 정확한 어텐션 형태인 no-rope MLA용 코드 경로가 없기 때문입니다.

누락된 경로를 패치하기 위해 이 저장소에 단일 파일 Python 오버레이를 작성했습니다. C++/CUDA 변경 사항은 없으며 이미지를 다시 빌드할 필요도 없습니다. 교체할 원본 파일 위에 이 파일을 바인드 마운트하기만 하면 됩니다.

## 동작하지 않았던 것

공식 플래그를 사용해 두 대의 DGX Spark 노드(Ray, tensor parallel = 2)에 걸쳐 공식 이미지를 띄웠는데, 로딩 직후 KV cache 프로파일링 단계에서 죽어버립니다:

```
concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla
```

`--kv-cache-dtype bfloat16` 옵션을 강제해 보았으나 실패 지점만 다른 곳으로 옮겨갈 뿐이었습니다:

```
No valid attention backend found ... head_size=512, use_mla=True, use_sparse=True
```

정리하자면 FP8 KV는 캐시 라이터에서 크래시가 나고, BF16은 사용할 수 있는 백엔드 자체가 없는 상태가 됩니다. 플래그를 어떻게 조합해도 동작하지 않습니다. 노드를 띄우려 시도하면서 겪은 14가지 이슈는 [LOG.md](LOG.md)에 모두 기록해 두었습니다.

## 원인
- GLM-5.3-Flash는 NSA 스타일의 토큰 선택 인덱서를 사용하는 sparse MLA 구조입니다. 쿼리당 상위 k개(top-k) 선택된 토큰만 어텐션 연산에 포함되며, 어텐션 자체는 RoPE 없이 동작합니다 (`qk_rope_head_dim=0`, `qk_nope_head_dim=256`, `kv_lora_rank=512`).
- Compute capability 12.x 환경에서 `vllm/platforms/cuda.py`는 MLA 백엔드 선택지를 `TRITON_MLA`와 `FLASHINFER_MLA_SPARSE_SM120` 두 가지로 좁힙니다. 모델이 sparse 방식이므로 기본값으로 `FlashInferMLASparseSM120Backend`를 선택하게 됩니다.
- SM100 백엔드(`flashinfer_mla_sparse.py`)는 실제로 이 no-rope shape를 정상적으로 처리하지만, SM120 백엔드(`flashinfer_mla_sparse_sm120.py`)는 지원하지 못합니다. KV write 경로에서 `concat_and_cache_mla`를 호출하는데, 여기서 `fp8_ds_mla` 분기가 64 크기의 RoPE 텐서를 기대하기 때문입니다 (`pe_dim must be 64`). 하지만 RoPE를 쓰지 않으므로 해당 텐서 자체가 존재하지 않습니다.
- SM120 sparse 커널 자체(v32 / GLM-NSA decode 경로)도 d_qk=576 레이아웃(512 latent + 64 RoPE)을 엄격하게 요구합니다. SM100 경로를 그대로 가져와 쓸 수도 없는 상황인데, SM100 no-rope 커널(TRTLLM-GEN)이 sm_121을 완전히 거부하기 때문입니다 (`fmhaRunner.cuh`의 `Unsupported architecture`).
- 결론적으로, 해당 이미지는 no-rope 경로가 존재하는 GB200급 하드웨어(sm_100)에서만 테스트된 것이 분명합니다. DGX Spark(sm_121)는 두 백엔드 사이의 사각지대에 정확히 걸려 있는 상태입니다.

## 수정 방법: RoPE 차원 제로 채우기

커널이 576-wide query/key 레이아웃을 강제하지만, 해당 64 rope dimensions에는 실제로 rope 데이터가 들어갈 필요가 없습니다. 따라서 다음과 같이 처리했습니다.

- **KV side**: SM120 기본 `fp8_ds_mla` 캐시 레이아웃(토큰당 656-byte packed blob)을 유지하고 64 rope 슬롯을 0으로 채웠습니다. 직접 수정한 cache writer 오버라이드를 통해 `concat_and_cache_mla` 직전에 zero-width `k_pe` 텐서를 64-wide zero 텐서로 교체합니다.
- **Query side**: 512-wide latent query를 64개의 0으로 패딩하고 커널에 `qk_rope_head_dim=64`를 전달했습니다.

어떤 수에 0을 곱해도 0이므로, 추가된 차원은 내적 결과에 0만 더하게 됩니다. 결국 576-wide 커널은 rope를 사용하지 않는 네이티브 실행과 완전히 동일한 어텐션을 계산합니다. 근사치가 아니라 수학적으로 동일합니다.

fp8 KV 스케일링(GLM 모델용 `kv_scale_format=arbitrary_fp32`)은 그대로 유지했습니다.

서버에 적용하기 전, sm_121 환경에서 독립 단위 테스트를 통해 해당 커널 경로를 검증했습니다. 출력 결과는 BF16 레퍼런스와 최대 절대 오차 0.0024(단순 fp8 양자화 노이즈 수준) 내에서 일치했으며, 유효하지 않은(`-1`) top-k 인덱스 역시 정상적으로 마스킹(0.0031)되는 것을 확인했습니다.

## 저장소 구성

```
overlay/flashinfer_mla_sparse_sm120.py   오버레이(드롭인 대체용)
overlay.diff                            동일한 변경 사항을 담은 diff
serve/start-head.sh                     Ray head + vLLM 서버(노드 A에서 실행)
serve/start-worker.sh                   Ray worker(노드 B에서 실행)
serve/glm-serve.sh                      up / down / status / smoke / sync 래퍼
RUNBOOK.md                              복사해서 바로 쓰는 런북: 설정값, 명령어, 예상 출력, 주요 실패 사례
LOG.md                                  시작 로그: 15개 장애 요소 및 타임스탬프
```

## 빠른 시작

[RUNBOOK.md](RUNBOOK.md)는 AI 에이전트(또는 사람)가 바로 복사해서 실행할 수 있는 런북입니다. 입력해야 하는 값, 순서대로 실행할 5개의 명령어, 각 단계별 예상 출력값과 측정된 소요 시간, 그리고 가장 흔히 발생하는 5가지 실패 상황을 해결하는 방법을 정리했습니다.

## 사용법

DGX Spark 노드 2대를 다이렉트 링크로 연결했습니다 (나는 QSFP 포트를 사용하며, 직접 도달 가능한 IP 쌍이면 무엇이든 동작합니다. 스크립트가 Ray와 NCCL 트래픽을 이 점대점 주소로 고정합니다).

1. 양쪽 노드에 가중치와 이미지를 준비합니다. 공식 이미지(vLLM v0.28.1rc1.dev580+g385dce36b, flashinfer 0.6.18)를 기준으로 빌드하고 테스트했습니다:
```bash
docker pull vllm/vllm-openai:glm53-flash-arm64-cu130
# NVIDIA GLM-5.3-Flash-NVFP4 checkpoint, e.g. under ~/models/GLM-5.3-Flash-NVFP4
```

2. 양쪽 노드의 동일한 디렉터리(예: `~/vllm-overlay/`)에 `overlay/flashinfer_mla_sparse_sm120.py`를 배치합니다 (또는 `serve/glm-serve.sh sync-overlay` 실행).

3. head를 먼저 띄운 뒤 worker를 실행합니다 (`CHANGEME` 값을 채워 넣거나 사전에 export합니다):
```bash
# node A (head)
HEAD_IP=CHANGEME IFNAME=CHANGEME ./serve/start-head.sh

# node B (worker)
HEAD_IP=CHANGEME MY_IP=CHANGEME IFNAME=CHANGEME ./serve/start-worker.sh
```
`HEAD_IP`와 `MY_IP`는 점대점 링크의 엔드포인트이며, `IFNAME`은 해당 IP를 전달하는 인터페이스입니다. 콜드 스타트는 약 13분 소요됩니다 (191 GB의 가중치). 기본값인 600 s는 타임아웃이 발생하므로 스크립트에서 `VLLM_ENGINE_READY_TIMEOUT_S`를 3600 s로 늘려 두었습니다.

4. 순서가 중요합니다. 이전 실행에서 남아 있는 기존 worker가 있다면 head를 재시작하기 전에 반드시 종료하고, 그 이후에 worker를 다시 띄워야 합니다. 이전 실행의 잔여 worker가 새 Ray 클러스터에 조인을 시도하면 session-mismatch 에러와 함께 head가 크래시됩니다. `serve/glm-serve.sh up`을 사용하면 이 시퀀스를 자동으로 처리합니다 (`down` / `up` / `status` / `smoke`).

5. FlashInfer autotune은 명시적으로 꺼 두었습니다 (`--kernel-config '{"enable_flashinfer_autotune": false}'`). 현재 환경에서는 autotune을 켜 둘 경우 더미 패스 도중 rank 0에서 크래시가 발생했습니다.

### thinking 블록 건너뛰기

GLM-5.3 챗 템플릿은 열린 `<think>` 태그를 강제 삽입하며(`add_generation_prompt`가 `<|assistant|><think>`를 주입함) 별도의 `enable_thinking` 토글을 제공하지 않습니다. `max_tokens`를 작게 설정하면 모델이 사고 블록 내에서 전체 토큰 예산을 소진해 응답이 비어 있는 상태로 반환됩니다. 내가 사용하는 우회 방법은 빈 assistant 턴을 추가하고 continuation을 요청하는 방식입니다:

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

이렇게 하면 생성이 `<|assistant|><think></think>` 바로 뒤부터 시작되어 thinking 블록이 즉시 닫히고 출력이 실제 응답으로 직행합니다 (256 토큰 요청으로 테스트한 결과 183 토큰의 온전한 응답 텍스트를 얻었습니다). 주의할 점: `glm45` 추론 파서가 닫는 `</think>` 태그를 만나지 못하므로 전체 본문을 `reasoning`으로 분류하고 `content`를 비워 둘 수 있습니다. `content`를 먼저 읽고, 비어 있다면 `reasoning`을 가져오면 됩니다. `serve/glm-serve.sh smoke`에 이미 이 폴백 로직이 구현되어 있습니다.

## 측정 결과 (2× DGX Spark, Ray 기반 TP=2)
- 엔진 초기화에 105 s 소요되었고, KV 캐시는 노드당 7.69 GiB로 잡혔습니다 (`--max-model-len 131072`, fp8_ds_mla layout).
- 동시 요청 수 16으로 총 48개 요청을 실행했습니다: 실패 0건, 총합 61.5 tok/s를 기록했으며, 케이스 온도는 대략 64–66 °C 수준을 유지했습니다.
- 설정 변경 없이 후속 프로덕션 런을 위해 동시 요청 수를 32로 올렸습니다.

## 소요 시간

2026-09-12 심야에 첫 기동을 시도한 시점부터 같은 날 늦게 실제 프로덕션 런을 돌리기까지의 기록입니다 (모든 시각은 현지 시간 기준):

| Time | Milestone |
| --- | --- |
| small hours | 다이렉트 링크를 통해 가중치 복사 완료, 직접 빌드한 버전으로 기동 시도 |
| 06:50 | 6번의 기동 실패 후 자체 빌드 중단 — "동작하지 않았던 것"에 정리한 이슈에 봉착 |
| 07:10 | 모델 카드에서 공식 이미지 확인 |
| 07:40 | 공식 이미지에서도 정확히 동일한 문제 발생 확인 (TP=2) |
| 07:40–11:30 | 근본 원인 분석, 오버레이 구성, sm_121 환경에서 독립 단위 테스트 5개 작성 및 실행 |
| 11:30–11:50 | 오버레이 기동 1차 및 2차 시도 실패 |
| 12:32 | 오버레이 3차 시도: 드디어 서버 정상 기동 (엔진 초기화에 105 s 소요) |
| 12:35–12:47 | 검증 실행: 48개 요청, concurrency 16, 총합 61.5 tok/s 달성 |
| ~12:50 | 32 concurrent 환경에서 학습 말뭉치 대상 프로덕션 런 시작 |

총 기동 시도 횟수는 자체 빌드 6회, 공식 이미지 3회, 오버레이 3회였으며, 최종적으로 오버레이 3번째 시도에서 성공했습니다. 각 실패 지점과 대기 시간은 [LOG.md](LOG.md)에 기록해 두었습니다.

## 제작 방법
- **Planning and verification** — GLM-5.3 (Z.ai): 단계 계획을 보조하고 결과를 검증했습니다.
- **Implementation** — Z.ai 앱 내 GLM-5.3-Flash: 오버레이를 통한 설계 작업에는 reasoning effort Low를 사용했고, `serve/glm-serve.sh` 래퍼 작성 시에는 High로 올려서 적용했습니다.
- **Research and triage** — GLM-5.3 (Z.ai): 공식 문서를 확인하며 근본 원인(root cause)을 좁혀냈습니다.
- **Hardware** — four DGX Spark nodes (DGX Spark): 각 쌍 사이에 direct QSFP link를 연결해 두 쌍으로 분할 구성했습니다.
- **Software** — 공식 vLLM 이미지(`vllm/vllm-openai:glm53-flash-arm64-cu130`) 및 내가 직접 작성한 `serve/glm-serve.sh` 스크립트.
- 정책 결정과 툴 간 내용 전달은 내가 진행했고, 구현과 테스트는 모델들이 처리했습니다.
- 서드파티 레시피나 패치, 포크(fork)는 일절 사용하지 않았습니다. 모든 것은 공식 소스에서 직접 가져왔습니다: NVIDIA model card와 이미지, 해당 이미지 내에 이미 포함된 vLLM 소스 코드, 그리고 Z.ai의 공식 문서입니다.

## 로그

전체 시작 로그는 [LOG.md](LOG.md)에 보관해 두었습니다. 내가 마주친 순서대로 총 15개의 이슈를 타임스탬프 및 원본 오류 메시지와 함께 정리해 두었습니다. 사실상 serve 스크립트의 각 플래그가 왜 그렇게 설정되었는지에 대한 나의 작업 기록입니다.

## 감사의 말

Z.ai가 GLM-5.3-Flash를 오픈 모델로 공개하고, NVIDIA가 공식 NVFP4 체크포인트와 이를 구동하는 공식 vLLM 이미지를 배포해 준 덕분에 본 작업을 진행할 수 있었습니다. 감사합니다.
