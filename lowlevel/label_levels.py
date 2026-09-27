"""L1: the level (low / high) and the kind of meaning lost, for each of the 110 rated real bugs whose cause is known
(consensus K1-K6 and the 2 split items of the two pre-registered replays; K7 = cause not stated is left out).

The category (K1..K7) is the blind raters' consensus. The level and the kind are ONE session's judgment, made with
the hypothesis known, from the raters' evidence text and the issue title (lowlevel/rated_173.json). Rule:
  low   the meaning lost is a property of the data the GPU computes on (layout, stride, dtype, scale, which row or
        slot or page a value belongs to, a position frame, a cache's identity or validity, a buffer's lifetime), a
        numeric computation, the kernel or backend chosen, or execution order / collectives inside the engine
  high  the meaning lost is about text, tokens, requests, configuration values or responses before they become
        tensors or after results leave the model (tokenizer, chat template, request settings, parsers, response
        fields, placeholder provenance at the token level, adapter settings read at load, the caller's own input)
Run: python lowlevel/label_levels.py  ->  lowlevel/level_labels.json and the counts
"""
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

L = "low"
H = "high"
LABELS = {
    # --- K1: a meaning declared somewhere is lost at a boundary ---
    "diffusers#14379": (H, "요청 인자", "pipeline이 references 인자를 버림(멀티 GPU 장치 배치는 크게 실패)"),
    "sglang#21566": (L, "실행 순서·통신", "ROCm GEMM이 호출자의 스트림이 아니라 기본 스트림에서 돔"),
    "sglang#21696": (L, "배치·보폭", "Qwen3.5 투영 가중치가 융합 GDN Triton 커널이 가정한 interleaved 배치가 아님"),
    "sglang#21843": (L, "배치·보폭", "비연속 텐서를 연속으로 가정한 Triton 커널"),
    "sglang#23554": (L, "실행 순서·통신", "reduce-scatter 신호가 안 가 dense-MLP가 all-reduce를 두 번"),
    "sglang#23746": (L, "인덱스·대응 기준", "이종 TP PD에서 decode 랭크가 모두 KV 헤드 0에 대응, Mamba 상태 조각이 그룹 배치 무시"),
    "sglang#25055": (H, "응답 필드", "reasoning 분리가 logprobs에 전달 안 됨"),
    "sglang#26518": (L, "스케일·양자화 형식", "FP8 q/k/v 스케일이 TRTLLM MHA 커널로 안 넘어감"),
    "sglang#28502": (L, "위치 기준", "배치된 은닉 상태에 [1, seq] 기본 position_ids로 RoPE"),
    "sglang#29330": (L, "캐시·상태의 정체와 유효성", "Mamba2 prefill 초기 상태 복원이 잘못된 조건에 묶임"),
    "sglang#30989": (L, "자료형 약속", "fp32 편향을 bf16을 요구하는 라우팅 커널에 넘김 → NaN"),
    "sglang#31482": (L, "인덱스·대응 기준", "호스트 KV 페이지 번호를 장치 버퍼에 씀"),
    "sglang#31490": (L, "배치·보폭", "전치된 스케일 저장을 연속 메타데이터로 믿어 스케일이 엉뚱한 칸과 짝지어짐"),
    "sglang#31833": (L, "인덱스·대응 기준", "요청별 청크 수로 전역 청크 격자의 SSM 상태를 읽음"),
    "sglang#33245": (L, "스케일·양자화 형식", "융합 적재기가 패킹된 양자화 가중치 이름을 몰라 버림"),
    "sglang#34227": (L, "배치·보폭", "FSDP 경로가 헤드별 QKV 행 재배열 적재기를 건너뜀"),
    "sglang#36371": (L, "자료형 약속", "PD에서 Mamba 상태 자료형·항목 길이가 다른데 원본 길이를 보폭으로 씀"),
    "sglang#36423": (L, "스케일·양자화 형식", "융합 이름이 투영별 양자화 방식을 놓쳐 int8을 bf16에 그대로 복사"),
    "sglang#36886": (L, "인덱스·대응 기준", "DCP 가상 KV 위치를 랭크별 버퍼에 번역 없이 씀"),
    "sglang#38573": (L, "스케일·양자화 형식", "group_size 64 체크포인트에 커널이 chunk 128을 씀"),
    "sglang#40835": (H, "적재 설정 값", "LoRA 설정의 use_rslora를 읽지 않음"),
    "transformers#45356": (H, "토크나이저", "선언된 TikToken 토크나이저를 다른 백엔드로 바꿈"),
    "transformers#45812": (H, "토크나이저", "Granite를 GPT2Tokenizer로 라우팅"),
    "transformers#46489": (H, "토크나이저", "선언된 클래스가 tokenizer.json 파이프라인을 못 씀"),
    "transformers#46612": (L, "인덱스·대응 기준", "빔 탐색이 다른 이름의 캐시 행을 재배열하지 않음"),
    "transformers#48256": (L, "위치 기준", "오른쪽 패딩 prefill 뒤 Mamba 캐시가 마지막 유효 위치가 아닌 패딩 뒤 상태를 가짐"),
    "vllm#38643": (L, "배치·보폭", "FLA 커널이 time-first를 기대하는데 head-first를 받음"),
    "vllm#39179": (L, "자료형 약속", "FlashInfer가 요구하는 bf16 라우팅 편향을 블록 FP8 경로만 변환 안 함"),
    "vllm#39273": (L, "캐시·상태의 정체와 유효성", "추측 디코딩에서 수락 토큰 수가 SSM 메타데이터로 안 가 슬롯 0의 낡은 상태를 읽음"),
    "vllm#39407": (L, "스케일·양자화 형식", "가중치에 흡수된 활성 스케일을 다시 적용(이중 스케일)"),
    "vllm#41132": (H, "템플릿·요청 설정", "구조화 출력 게이트가 요청의 chat-template kwargs 없이 추론 파서를 만듦"),
    "vllm#41472": (L, "배치·보폭", "ROCm 페이지드 어텐션 커널이 표준 연속 KV 배치 공식을 박아 둠"),
    "vllm#41511": (L, "스케일·양자화 형식", "TP에서 스케일을 K로 안 나눠 커널이 모양으로 group 크기를 잘못 추정"),
    "vllm#42016": (L, "규약", "MRoPE 커널이 층이 선언한 짝짓기(is_neox_style=False)를 무시"),
    "vllm#43728": (H, "템플릿·요청 설정", "템플릿은 enable_thinking, 추론 파서는 thinking을 읽음"),
    "vllm#43996": (L, "캐시·상태의 정체와 유효성", "PD+추측 디코딩에서 접두 캐시 다듬기가 엉뚱한 블록을 버림"),
    "vllm#45562": (L, "자료형 약속", "플랫폼의 e4m3fnuz 대신 e4m3fn으로 KV 바이트를 해석"),
    "vllm#45734": (L, "인덱스·대응 기준", "은닉 상태 KV 그룹 번호를 0으로 박아 하이브리드에서 다른 그룹의 블록을 읽음"),
    "vllm#46261": (L, "배치·보폭", "시험 기준이 원래 bf16 가중치를, 커널은 다른 배치로 바뀐 가중치를 받음(시험 코드)"),
    "vllm#46585": (L, "배치·보폭", "시험이 SM90 GEMM에 요구되는 interleaved 배치 없이 가중치·스케일을 넣음(시험 코드)"),
    "vllm#47300": (L, "위치 기준", "FA4 마스크가 슬라이딩 윈도 제약을 버리고 상대 q_idx를 씀"),
    "vllm#47783": (L, "배치·보폭", "패킹된 KV 배치의 블록 보폭을 sparse-MLA decode가 모름"),
    "vllm#47986": (H, "파서", "도구 슬롯을 인자 텍스트로 찾아 다른 도구의 스키마로 풂"),
    "vllm#48058": (L, "배치·보폭", "XPU FP8 커널이 C-연속 가중치를 요구하는데 전치 뷰를 받음"),
    "vllm#48324": (L, "자료형 약속", "BF16 입력과 FP32 norm 가중치에 자료형 확인 없이 융합 연산 선택"),
    "vllm#48611": (L, "인덱스·대응 기준", "분할 뒤에도 전체 배치의 req_id_per_token을 넘김"),
    "vllm#48895": (L, "자료형 약속", "topk_weights를 자료형 확인 없이 fp32로 재해석"),
    "vllm#49070": (L, "규약", "Marlin MoE가 모델의 swigluoai alpha/beta/limit 대신 기본 SiLU 값을 씀"),
    "vllm#50332": (L, "실행 순서·통신", "DeepGemm 제외 설정이 FP8 MoE 백엔드 선택기에 안 감"),
    "vllm#51164": (H, "파서", "추론 파서가 엉뚱한 개행을 벗김"),
    "vllm#52071": (L, "실행 순서·통신", "동기 스케줄링에서 토큰이 돌아오는 단계 표시가 안 됨"),
    "vllm#52276": (L, "캐시·상태의 정체와 유효성", "NIXL 수신 실패를 완료로 보고해 빈 KV를 유효하다고 씀"),
    "vllm#54974": (L, "스케일·양자화 형식", "w13 융합에서 gate의 전역 스케일만 남겨 up 절반을 잘못 역양자화"),
    "vllm#55357": (L, "배치·보폭", "융합 PLE 커널이 비연속 state_indices 조각의 보폭을 무시"),
    "vllm#56380": (L, "인덱스·대응 기준", "블록 표에 물리 페이지 번호 대신 관리자 블록 번호"),
    "vllm#56655": (L, "캐시·상태의 정체와 유효성", "접두 캐시 키에 프롬프트 임베딩 마스크가 빠짐"),
    "vllm#56949": (L, "실행 순서·통신", "재구성 뒤 0으로 초기화된 랭크를 전역 정체로 읽어 가중치 전송 묶음을 잘못 고름"),
    "vllm#57740": (H, "자리표", "자리표를 출처가 아니라 값으로 묶음"),
    "vllm#58532": (L, "스케일·양자화 형식", "채널별 가중치 스케일을 텐서별 활성 플래그로 색인"),
    "vllm#58627": (L, "캐시·상태의 정체와 유효성", "NaN 행에서 슬롯 0만 쓰는데 재사용 버퍼의 모든 슬롯을 유효로 읽음"),
    # --- K2: arithmetic or numerics inside one component ---
    "diffusers#13425": (L, "산술·정밀도", "표준편차 0으로 나눔 → NaN"),
    "transformers#47475": (L, "산술·정밀도", "청크 사이 상태 점화식이 잘못된 축으로 합산"),
    "transformers#47885": (L, "산술·정밀도", "특징 추출의 동적 범위 바닥이 NaN 하나로 전체를 NaN으로(전처리)"),
    "transformers#48293": (L, "산술·정밀도", "라우터 cumsum 차원·logits 계산 오류"),
    "vllm#43301": (L, "산술·정밀도", "Mamba 커널의 bf16 중간 형변환"),
    "vllm#48231": (L, "산술·정밀도", "BF16→FP16 대체 뒤 비전 투영이 FP16에서 넘침"),
    "vllm#50427": (L, "산술·정밀도", "페이지드 KV 오프셋이 int32로 넘침"),
    "vllm#54035": (L, "산술·정밀도", "FA3 FP8 커널의 decode/prefill 타일 폭이 달라 수치가 다름"),
    "vllm#56578": (L, "산술·정밀도", "softcap의 exp 기반 tanh가 inf/inf"),
    "vllm#57017": (L, "산술·정밀도", "CUDA 그래프 패딩 행의 0/0"),
    # --- K3: memory or object lifetime ---
    "sglang#22701": (L, "수명·버퍼", "스트리머가 재사용하는 버퍼의 뷰를 붙들고 있음"),
    "sglang#29748": (L, "수명·버퍼", "초기화 안 된 출력에 여러 블록이 atomicAdd"),
    "vllm#41262": (L, "수명·버퍼", "블록 크기가 다른 KV 그룹이 같은 텐서를 공유"),
    "vllm#42182": (L, "수명·버퍼", "지연된 0 채우기 커널이 막 받은 KV를 지움"),
    "vllm#43163": (L, "수명·버퍼", "스트리머가 버퍼를 재사용해 내준 텐서가 덮임"),
    "vllm#50881": (L, "수명·버퍼", "CUDA 그래프 캡처 중 새 텐서 할당과 흩어 쓰기가 캐시를 망침"),
    "vllm#53488": (L, "수명·버퍼", "드래프터가 재사용 CUDA 그래프 출력 버퍼를 먼저 덮어씀"),
    "vllm#57136": (L, "수명·버퍼", "여러 인코더 그래프가 한 메모리 풀을 공유할 때 NaN"),
    # --- K5: hardware, driver, platform or compiler backend ---
    "diffusers#14368": (L, "하드웨어·백엔드", "MPS에서만 NaN"),
    "sglang#22760": (L, "하드웨어·백엔드", "inductor 조각 컴파일에서만 정밀도 틀림"),
    "sglang#24091": (L, "하드웨어·백엔드", "SM100 빌드의 Marlin W4A16 커널"),
    "sglang#28685": (L, "하드웨어·백엔드", "gfx950 hipcc 오컴파일"),
    "sglang#31011": (L, "하드웨어·백엔드", "oneCCL이 정수 MIN/MAX를 SUM으로"),
    "vllm#52644": (L, "하드웨어·백엔드", "ROCm FULL_DECODE_ONLY 그래프에서만"),
    "vllm#53211": (L, "하드웨어·백엔드", "XPU 그래프 캡처에 커널이 기록 안 됨"),
    "vllm#55560": (L, "하드웨어·백엔드", "CPU MoE를 inductor로 빌드할 때만 NaN"),
    "vllm#57064": (L, "하드웨어·백엔드", "ROCm AITER 융합 연산이 그래프 재생을 망침"),
    # --- K4: control flow, dispatch or parsing inside one component ---
    "sglang#23687": (L, "스케일·양자화 형식", "이름을 두 번 바꿔 FP8 weight_scale_inv를 버림"),
    "sglang#28180": (H, "요청 설정", "DFlash 검증 경로가 repetition_penalty를 무시"),
    "sglang#28792": (H, "요청 설정", "Anthropic 엔드포인트의 메시지 역할 검사"),
    "sglang#36938": (L, "인덱스·대응 기준", "건너뛴 요청에서 logprob 커서를 안 옮겨 다음 요청이 밀린 조각을 읽음"),
    "sglang#37187": (L, "실행 순서·통신", "DP 어텐션에서 MoE 합을 두 번 all-reduce"),
    "sglang#40735": (L, "실행 순서·통신", "MoE 분기가 FuseEP를 빠뜨려 all-reduce가 더 붙음"),
    "transformers#46032": (L, "실행 순서·통신", "seq_len > 1인데 한 단계 decode 경로로 분기"),
    "transformers#46710": (H, "토크나이저", "잘못된 토크나이저 클래스"),
    "vllm#39468": (H, "파서", "도구 호출 파서가 구분 토큰을 JSON에 흘림"),
    "vllm#42047": (H, "파서", "스트리밍 도구 파서의 부동소수 조각"),
    "vllm#43559": (L, "캐시·상태의 정체와 유효성", "접두 캐시 여유를 재귀 상태 그룹에도 적용"),
    "vllm#45691": (L, "실행 순서·통신", "LoRA 경로가 조건 없이 all-gather"),
    "vllm#46863": (H, "파서", "pythonic 파서가 JSON 형태 도구 호출을 모름"),
    "vllm#48217": (H, "파서", "스트리밍 파서의 초기 상태가 REASONING"),
    "vllm#49316": (H, "파서", "스트리밍에서 스키마 형 변환을 건너뜀"),
    "vllm#49412": (H, "파서", "스트리밍에서 공백 벗기기가 빠짐"),
    "vllm#52576": (L, "스케일·양자화 형식", "커널의 K 타일이 양자화 블록을 넘고 마스크가 없음"),
    "vllm#53051": (L, "실행 순서·통신", "1+k 토큰 prefill을 추측 디코딩 CUDA 그래프로 잘못 보냄"),
    "vllm#55927": (L, "배치·보폭", "fp8_fp4_paged_mqa_logits의 정렬 경계(seq_len % 4)"),
    # --- K6: the caller's own configuration or usage ---
    "diffusers#13411": (H, "사용자 입력", "호출자가 단조롭지 않은 sigma를 넣음"),
    "vllm#40080": (H, "모델 자체", "모델의 반복 경향"),
    # --- split (raters did not agree) ---
    "sglang#26745": (L, "수명·버퍼", "비전 가중치 없는 체크포인트의 초기화 안 된 파라미터 NaN이 슬롯 재사용으로 퍼짐"),
    "vllm#48898": (L, "배치·보폭", "K가 32로 정렬되지 않을 때 패딩된 블록 스케일 열이 낡은 메모리를 읽음(compile)"),
}


# For each low-level item: which general comparison would have exposed it without a rule written for it, and what it
# takes to trigger. Also one session's judgment from the same text. Mechanisms:
#   ref     the engine against a reference implementation of the same model or op, on the same input
#   mode    the engine against itself in another execution mode that must mean the same thing (eager / CUDA graph /
#           torch.compile, cache on / off, speculative decoding on / off, PD split / colocated, TP-DP-EP / one GPU,
#           batched / alone, padding side, chunk size, fusion on / off, overlap scheduling on / off, another backend)
#   type    a property the tensor itself carries (dtype, contiguity, stride) against what the kernel expects
#   load    the loaded parameters against the checkpoint's tensors (every tensor used, values as declared)
#   finite  values stay finite across a boundary
# Trigger: "1gpu" = reproducible on one consumer NVIDIA GPU like ours; otherwise what it needs.
EXPOSE = {
    "sglang#21566": ("mode", "ROCm"), "sglang#21696": ("ref mode", "1gpu"), "sglang#21843": ("ref type", "1gpu"),
    "sglang#23554": ("mode", "multi-GPU"), "sglang#23746": ("mode", "multi-GPU"), "sglang#26518": ("ref", "Blackwell"),
    "sglang#28502": ("mode", "1gpu"), "sglang#29330": ("ref mode", "1gpu"), "sglang#30989": ("type ref", "Blackwell"),
    "sglang#31482": ("mode", "multi-GPU"), "sglang#31490": ("ref", "ROCm"), "sglang#31833": ("mode", "1gpu"),
    "sglang#33245": ("load ref", "1gpu"), "sglang#34227": ("mode load", "multi-GPU"), "sglang#36371": ("type mode", "multi-GPU"),
    "sglang#36423": ("load type", "NPU"), "sglang#36886": ("mode", "multi-GPU"), "sglang#38573": ("ref", "Hopper"),
    "transformers#46612": ("mode", "1gpu"), "transformers#48256": ("mode", "1gpu"), "vllm#38643": ("ref", "1gpu"),
    "vllm#39179": ("type ref", "Blackwell"), "vllm#39273": ("mode", "1gpu"), "vllm#39407": ("ref", "1gpu"),
    "vllm#41472": ("ref", "ROCm"), "vllm#41511": ("mode", "multi-GPU"), "vllm#42016": ("ref", "1gpu"),
    "vllm#43996": ("mode", "multi-GPU"), "vllm#45562": ("type ref", "ROCm"), "vllm#45734": ("ref", "1gpu"),
    "vllm#46261": ("ref", "ROCm"), "vllm#46585": ("ref", "Hopper"), "vllm#47300": ("ref mode", "Hopper"),
    "vllm#47783": ("ref", "Hopper"), "vllm#48058": ("type", "XPU"), "vllm#48324": ("type mode", "multi-GPU"),
    "vllm#48611": ("ref", "Hopper"), "vllm#48895": ("type ref", "1gpu"), "vllm#49070": ("ref", "Hopper"),
    "vllm#50332": ("mode", "Blackwell"), "vllm#52071": ("mode", "multi-GPU"), "vllm#52276": ("mode", "multi-GPU"),
    "vllm#54974": ("ref", "Blackwell"), "vllm#55357": ("type mode", "1gpu"), "vllm#56380": ("ref", "ROCm"),
    "vllm#56655": ("mode", "1gpu"), "vllm#56949": ("mode", "multi-GPU"), "vllm#58532": ("ref", "1gpu"),
    "vllm#58627": ("finite ref", "Hopper"),
    "sglang#23687": ("load ref", "1gpu"), "sglang#36938": ("mode", "1gpu"), "sglang#37187": ("mode", "multi-GPU"),
    "sglang#40735": ("mode", "NPU"), "transformers#46032": ("mode", "1gpu"), "vllm#43559": ("mode", "1gpu"),
    "vllm#45691": ("mode", "multi-GPU"), "vllm#52576": ("ref", "1gpu"), "vllm#53051": ("mode", "1gpu"),
    "vllm#55927": ("ref", "Blackwell"),
    "sglang#26745": ("load finite", "1gpu"), "vllm#48898": ("mode", "Blackwell"),
    "diffusers#13425": ("finite", "1gpu"), "transformers#47475": ("mode ref", "1gpu"), "transformers#47885": ("finite", "1gpu"),
    "transformers#48293": ("mode ref", "1gpu"), "vllm#43301": ("mode ref", "1gpu"), "vllm#48231": ("finite mode", "1gpu"),
    "vllm#50427": ("none", "a very large KV cache"), "vllm#54035": ("mode", "Hopper"), "vllm#56578": ("finite ref", "1gpu"),
    "vllm#57017": ("finite mode", "1gpu"),
    "sglang#22701": ("load", "1gpu"), "sglang#29748": ("mode ref", "1gpu"), "vllm#41262": ("mode ref", "1gpu"),
    "vllm#42182": ("mode", "multi-GPU"), "vllm#43163": ("load", "1gpu"), "vllm#50881": ("mode", "1gpu"),
    "vllm#53488": ("mode", "1gpu"), "vllm#57136": ("mode", "1gpu"),
    "diffusers#14368": ("mode", "MPS"), "sglang#22760": ("mode", "1gpu"), "sglang#24091": ("mode ref", "Blackwell"),
    "sglang#28685": ("ref", "ROCm"), "sglang#31011": ("ref", "XPU"), "vllm#52644": ("mode", "ROCm"),
    "vllm#53211": ("mode", "XPU"), "vllm#55560": ("mode", "1gpu"), "vllm#57064": ("mode", "ROCm"),
}


def main():
    rows = json.load(open(os.path.join(HERE, "rated_173.json"), encoding="utf-8"))
    known = [r for r in rows if r["consensus"] != "K7"]
    missing = [r["issue"] for r in known if r["issue"] not in LABELS]
    extra = [k for k in LABELS if k not in {r["issue"] for r in known}]
    assert not missing and not extra, (missing, extra)
    lows = {k for k, v in LABELS.items() if v[0] == L}
    assert lows == set(EXPOSE), (sorted(lows - set(EXPOSE)), sorted(set(EXPOSE) - lows))
    out, by_cat, by_kind = [], defaultdict(Counter), Counter()
    for r in known:
        level, kind, why = LABELS[r["issue"]]
        mech, trig = EXPOSE.get(r["issue"], (None, None))
        out.append({**{k: r[k] for k in ("tag", "issue", "title", "consensus", "reproduced", "verdict")},
                    "level": level, "kind": kind, "why": why, "exposed_by": mech, "trigger": trig})
        by_cat[r["consensus"]][level] += 1
        if level == L:
            by_kind[(r["consensus"] if r["consensus"] in ("K2", "K3", "K5") else "K1/K4/split", kind)] += 1
    json.dump(out, open(os.path.join(HERE, "level_labels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    total = Counter(x["level"] for x in out)
    print("known-cause items:", len(out), dict(total))
    for c in sorted(by_cat):
        print(" ", c, dict(by_cat[c]))
    print("low-level kinds:")
    for (c, k), n in sorted(by_kind.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {n:3d}  {c:10s} {k}")
    rep = [x for x in out if x["reproduced"]]
    print("reproduced:", len(rep), Counter((x["consensus"] == "K1", x["level"]) for x in rep))
    low = [x for x in out if x["level"] == L]
    mechs = Counter(m for x in low for m in x["exposed_by"].split())
    print("low-level items exposed by (an item can count under several):", dict(mechs))
    print("  by mode alone or with others:", sum(1 for x in low if "mode" in x["exposed_by"].split()),
          "| by ref:", sum(1 for x in low if "ref" in x["exposed_by"].split()),
          "| by type/load/finite only:", sum(1 for x in low if not {"mode", "ref"} & set(x["exposed_by"].split())
                                             and x["exposed_by"] != "none"),
          "| none:", sum(1 for x in low if x["exposed_by"] == "none"))
    print("  trigger:", dict(Counter(x["trigger"] for x in low)))
    k1low = [x for x in low if x["consensus"] in ("K1", "K4", "split")]
    print("  meaning-loss low items (K1/K4/split):", len(k1low), "| 1gpu:", sum(1 for x in k1low if x["trigger"] == "1gpu"))


if __name__ == "__main__":
    main()
