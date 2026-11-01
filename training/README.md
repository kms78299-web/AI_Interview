# GANG2 — RECRUITVIEW Feature Extraction

MLP 학습 전 단계. RECRUITVIEW 데이터셋(`/local_datasets/RecruitView/`)의 각 면접
영상에서 Video / Audio / Text feature를 뽑아 저장한다.
**MLP 학습 코드는 여기 없음 — feature 추출 + 저장까지만.**

## ⚠️ 2026-10-05: embed/stats 두 트랙으로 개편

`extract_features.py`가 `--track embed --modality {audio,text,video} --encoder <name>`
CLI로 새로 바뀌었다. 아래 README의 "## 2~5" 섹션은 **예전 단일 파이프라인**
(`extract_features_legacy.py`로 이름만 바꿔서 그대로 남겨둠) 설명이라 지금 CLI랑
안 맞는 부분이 있음 — 최신 내용은 "## 6. Track 1: embed" 섹션 참고.

- **Track 1: embed** (MLP 입력용, frozen encoder forward만, 지금 구현됨)
  → `features/embed/{encoder}/{id}.npy`, 768차원
- **Track 2: stats** (해석 가능한 통계, LLM 피드백용, 아직 미구현 — 다음 작업)
  → `features/stats/{id}.json` 예정. 기존 MediaPipe(video_features.py)/
    librosa(audio_features.py)/`extract_text_stats_for_prompt`(text_features.py)
    코드를 재사용할 예정

## 폴더 구조

```
GANG2/
├── code/
│   ├── config.py                 # 경로 / feature 이름 / 하이퍼파라미터 / 인코더 레지스트리
│   ├── data_utils.py             # metadata.jsonl 로딩, transcript 정제, 라벨 벡터
│   ├── video_features.py         # Track 2(stats): OpenFace 우선, 없으면 MediaPipe
│   ├── audio_features.py         # librosa(stats) + wav2vec2 임베딩(embed) 둘 다
│   ├── text_features.py          # BERT/RoBERTa 임베딩(embed) + filler 등 통계(stats)
│   ├── video_embed.py            # Track 1(embed): VideoMAE video 임베딩 (신규)
│   ├── extract_features.py       # 새 CLI: --track embed/stats (지금 여기를 실행)
│   ├── extract_features_legacy.py # 예전 단일 파이프라인 (참고용으로 보존)
│   ├── qwen_utils.py             # Qwen2.5-Omni-7B 로딩/채팅/JSON 파싱
│   ├── interview_prompts.py      # dummy_data few-shot + 샘플 feature -> 프롬프트
│   ├── label_utils.py            # z-score 라벨 <-> 0~100 백분위 변환
│   └── analyze_interviews_llm.py # LLM 면접 분석 데모
├── scripts/
│   ├── run_extract.sh       # sbatch 스크립트 (feature 추출, 인자 그대로 전달)
│   └── run_llm_analysis.sh  # sbatch 스크립트 (LLM 면접 분석 데모)
├── logs/                    # extract_*_errors.log, slurm 로그
├── features/                # (신규) Track 1/2 결과물 — features/embed/{encoder}/, features/stats/
├── recruitview_features/    # (예전) extract_features_legacy.py 결과물 — video/audio/text 통계, labels.npy
├── audio_embeddings/        # (예전) extract_features_legacy.py의 wav2vec2 임베딩
├── llm_outputs/              # LLM 분석 결과 (.json) 이 저장되는 곳
├── requirements.txt
└── README.md
```

## 1. 환경 준비 (아직 설치 안 되어 있음 — 이 세션에서 직접 확인함)

```bash
conda create -n recruitview python=3.9 -y
conda activate recruitview
pip install -r requirements.txt
```

- **mediapipe는 반드시 `0.10.21`로 고정해서 설치할 것.** `pip install mediapipe`로
  그냥 깔면 최신 1.x가 잡히는데, 1.0부터 legacy Solutions API
  (`mp.solutions.face_mesh` 등)가 삭제돼서 이 프로젝트의 fallback 로직이
  `AttributeError: module 'mediapipe' has no attribute 'solutions'`로 죽는다.
  이미 잘못된 버전을 깐 상태라면:
  `pip install "mediapipe==0.10.21" "numpy<2"`
- `ffmpeg`은 이미 `/usr/bin/ffmpeg`에 설치되어 있어서 따로 안 깔아도 됨.
- OpenFace는 별도 설치 안 함. `FeatureExtraction` 바이너리가 PATH에 없으면
  자동으로 MediaPipe 기반 대체 로직을 탄다 (아래 "Video feature" 참고).
  OpenFace를 쓰고 싶으면 빌드 후 `FeatureExtraction`을 PATH에 넣거나
  `OPENFACE_BIN=/path/to/FeatureExtraction` 환경변수를 설정하면 됨.
- BERT 모델(`klue/bert-base`, `bert-base-uncased`)은 최초 실행 시
  HuggingFace Hub에서 자동 다운로드됨 (인터넷 필요, 이미 캐시되어 있으면
  오프라인도 가능).

## 2. 실행

```bash
# 로그인 노드에서 소량 테스트
cd code
python extract_features.py --limit 20

# 전체 실행 (sbatch)
sbatch scripts/run_extract.sh

# 전체 실행 (직접, GPU 노드에서)
bash scripts/run_extract.sh
```

주요 옵션 (`extract_features.py --help` 참고):
- `--metadata_path`, `--video_root`, `--output_dir`, `--embedding_dir`, `--log_dir`
- `--device {cuda,cpu}`
- `--limit N` : 앞에서 N개만 처리 (테스트용)
- `--overwrite` : 이미 완료된 샘플도 다시 추출

**중단 후 재실행 지원**: 샘플 `id`에 대해 `output_dir`의 `{id}_video.npy`,
`{id}_audio.npy`, `{id}_text.npy`와 `embedding_dir`의 `{id}_audio_embedding.npy`
까지 네 파일이 모두 있으면 이미 완료된 것으로 보고 스킵한다. 하나라도 실패하면
넷 다 저장하지 않으므로, 다음 실행에서 그 샘플은 다시 처음부터 재시도된다.
(wav2vec2 추가 전에 이미 뽑아둔 샘플은 `audio_embedding.npy`가 없어서 자동으로
다시 처리됨 — 정상.)

## 3. 결과물

`recruitview_features/` (해석 가능한 feature, LLM 프롬프트용) 안에:
- `{id}_video.npy` : 15차원 (`config.VIDEO_FEATURE_NAMES` 순서) — **해석 가능** (AU/gaze/pose)
- `{id}_audio.npy` : 32차원 (`config.AUDIO_FEATURE_NAMES` 순서, MFCC 13+13) — **해석 가능** (MFCC/pitch/rms/silence/speech_rate)
- `{id}_text.npy`  : 771차원 (BERT 768 + filler_word_count/vocab_diversity/answer_length) — 앞 768은 해석 불가(MLP용), 뒤 3개만 해석 가능
- `labels.npy`     : (N, num_labels) — **완료된 샘플 전체**를 매 실행마다 다시 스캔해서 생성
- `ids.npy`        : `labels.npy`의 각 행과 순서가 대응하는 샘플 id 목록
- `label_names.json` : `labels.npy` 컬럼 순서 (라벨 필드 이름)

`audio_embeddings/` (해석 불가능한 MLP 입력용 임베딩, 폴더 분리) 안에:
- `{id}_audio_embedding.npy` : 768차원, wav2vec2(`facebook/wav2vec2-base-960h`) 임베딩

**"해석 가능한 feature" vs "MLP용 임베딩" 구분**: video/audio 통계, text_stats(뒤 3개)는
숫자 하나하나가 무슨 뜻인지 사람이 읽을 수 있어서 `analyze_interviews_llm.py`가 LLM
프롬프트에 그대로 넣는다. 반대로 BERT 임베딩(text 앞 768)과 wav2vec2 임베딩
(audio_embedding)은 사전학습 모델이 뽑은 밀집 벡터라 사람이 값만 보고 의미를 알 수
없고, (가벼운) MLP의 입력으로만 쓴다. video는 아직 이 둘이 분리 안 돼 있어서 AU/gaze/
pose 벡터가 MLP 입력과 해석용을 겸하고 있음 (필요하면 나중에 VideoMAE 같은 사전학습
video 임베딩을 똑같이 추가할 수 있음, RECRUITVIEW 원논문도 VideoMAE/TimeSformer를 씀).

라벨 필드는 metadata.jsonl에서 `id`/`file_name`/`question`/`transcript`를
제외한 나머지 숫자 필드를 자동으로 모아서 정렬한 것이다. 실행 로그에 몇 개가
잡혔는지 출력되니, 실제 12개가 맞는지 첫 실행 때 꼭 확인할 것.

## 4. Feature 상세

### Video (`video_features.py`)
1순위 OpenFace(`FeatureExtraction -aus -gaze -pose`) → CSV에서
`AU06_r/AU12_r/AU45_r/AU04_r`, `gaze_angle_x/y`, `pose_Rx/Ry/Rz` 파싱.

OpenFace가 없으면 MediaPipe Face Mesh(`refine_landmarks=True`, iris 포함)로
프레임마다(`VIDEO_FRAME_SAMPLE_STRIDE=3`, 3프레임에 1번) 랜드마크를 뽑아:
- AU06/AU12/AU45/AU04는 **랜드마크 거리 기반 proxy 값**(실제 FACS AU 강도가
  아님 — 볼-눈 거리, 눈썹-눈 거리, 눈 종횡비, 입 너비/눈간거리 비율로 근사)
- gaze는 iris 중심의 눈 중심 대비 상대 위치
- pose는 `cv2.solvePnP` + 표준 3D 얼굴 모델로 추정한 pitch/yaw/roll

영상 전체에 대해 프레임별 값의 mean/std를 계산해서 저장한다.

### Audio (`audio_features.py`)
ffmpeg으로 mp4 → 16kHz mono wav 추출 후:
- **librosa 통계** (`extract_audio_features`, 해석용): MFCC 13개 mean/std,
  pitch(`librosa.pyin`) mean/std, RMS mean/std, `silence_ratio`
  (`librosa.effects.split` 기준 무음 비율), `speech_rate`(onset 개수 / 길이(초) —
  실제 ASR 없이 발화 빈도로 근사한 proxy)
- **wav2vec2 임베딩** (`extract_wav2vec2_embedding`, MLP용): `facebook/wav2vec2-base-960h`
  로 프레임별(~50Hz) hidden state를 뽑아 시간축 mean-pooling한 768차원 벡터.
  RECRUITVIEW 원논문(arXiv:2512.00450)이 쓰는 wav2vec2/HuBERT 계열 오디오
  임베딩과 같은 역할. (원래 self-supervised 버전인 `facebook/wav2vec2-base`를
  쓰려 했는데, 그 체크포인트는 safetensors가 없어서 transformers의
  CVE-2025-32434 보안 정책(`torch<2.6`이면 구형 `.bin` 로드 자체를 거부)에
  걸림 — safetensors가 있는 `-960h` 버전으로 바꿈. `Wav2Vec2Model`로 CTC head
  없이 encoder body만 쓰기 때문에 ASR 파인튜닝 여부는 임베딩 추출에 문제 없음.)

### Text (`text_features.py`)
- `data_utils.clean_transcript`로 `[00:01 - 00:11]` 형태 타임스탬프 제거
- 한글 포함 여부로 `klue/bert-base` / `bert-base-uncased` 자동 선택,
  attention-mask 기반 mean pooling으로 768차원 임베딩
- `filler_word_count`(um/uh/hmm 등, `config.FILLER_WORDS`), `vocab_diversity`
  (unique/total word), `answer_length`(단어 수)
- transcript가 비어있으면 에러로 처리하지 않고 0벡터로 채우며 로그에 경고 남김

## 5. 에러 처리 / 로그 (extract_features_legacy.py)

- `logs/extract_errors.log`에 `id / 예외타입 / 메시지 / traceback` 기록
- 콘솔에는 tqdm 진행바 + 스킵된 샘플만 한 줄씩 출력
- 실행이 끝나면 성공/스킵/에러 개수와 feature 통계(shape, mean, std)를 출력

## 5-1. Track 1: embed (`extract_features.py`, 지금 쓰는 CLI)

모달리티별로 frozen encoder에 forward만 해서 768차원 임베딩을 뽑는다.
fine-tuning 없음. 한 모달리티가 실패해도 다른 모달리티엔 영향 없게 완전히
따로 실행/저장된다.

```bash
# 사람이 직접 (로그인 노드 테스트용)
cd code
python extract_features.py --track embed --modality audio --encoder wav2vec2-base --limit 5
python extract_features.py --track embed --modality text  --encoder bert-base-uncased --limit 5
python extract_features.py --track embed --modality video --encoder videomae --limit 5

# sbatch로 (인자 그대로 전달됨)
sbatch scripts/run_extract.sh --track embed --modality video --encoder videomae --limit 5
```

| modality | `--encoder` | 실제 체크포인트 | 처리 |
|---|---|---|---|
| audio | `wav2vec2-base` | `facebook/wav2vec2-base-960h` | ffmpeg 16kHz mono 추출 → 마지막 hidden state 시간축 mean pooling → 768 |
| text | `bert-base-uncased` / `roberta-base` | 동일 | `clean_transcript`로 정제, max_length=512, attention-mask mean pooling → 768. 빈 transcript는 0벡터 |
| video | `videomae` | `MCG-NJU/videomae-base` | 영상에서 16프레임 균등 샘플링 → `VideoMAEImageProcessor`가 224×224로 리사이즈/정규화 → 시간/패치 토큰 전체 mean pooling → 768 |

**왜 `wav2vec2-base`가 `-960h`로 바뀌었나 / `deberta-v3-base`는 왜 없나**:
원래 요청받은 `facebook/wav2vec2-base`, `microsoft/deberta-v3-base`는 HF Hub에
safetensors가 없는 구형 `.bin`(pickle) 체크포인트뿐이라, transformers가
CVE-2025-32434 보안정책으로 "torch<2.6이면 그 포맷 로드 자체를 거부"하게 막혀있음
(`huggingface_hub.list_repo_files`로 직접 확인함). 지금 이 환경은 GPU 드라이버
제약으로 torch 2.5.1 고정이라, audio는 safetensors 있는 `-960h` 체크포인트로
바꾸되 `--encoder` 이름은 원래 요청대로 `wav2vec2-base`로 유지했고, text의
`deberta-v3-base`는 지원 목록에서 뺐음 (`bert-base-uncased`/`roberta-base`는
문제 없음).

저장: `features/embed/{encoder}/{id}.npy` (float32, shape=(768,)). resume
지원(이미 있으면 스킵, `--overwrite`로 강제 재실행), `--start`/`--end`로 구간
지정 가능, 에러는 `logs/extract_embed_{modality}_{encoder}_errors.log`에 기록.

**Track 2(stats)는 아직 구현 안 됨** — `--track stats`는 지금 에러를 내고 끝남.
다음 작업으로 기존 MediaPipe/librosa/텍스트 통계 코드를 `features/stats/{id}.json`
형태로 재배선할 예정.

## 6. LLM 면접 분석 데모 (`analyze_interviews_llm.py`)

MLP용 feature 추출과는 별개로, `Qwen/Qwen2.5-Omni-7B`
(`/ceph_data/kms78299/repos/Various_MLLM/Qwen2.5-Omni-7B/code/model_utils.py`
패턴 그대로)에 실제 샘플 몇 개를 넣어서 면접 점수를 예측해보는 데모.
**Qwen한테 숫자 통계만 텍스트로 설명해주는 게 아니라, 영상 파일 자체
(화면+오디오)를 직접 같이 넣어서** 모델이 실제로 보고 들은 걸 근거로
판단하게 한다. `{"type": "video", "path": ...}`(화면)와
`{"type": "audio", "path": ...}`(우리가 ffmpeg으로 직접 뽑은 wav)를 각각
별도 콘텐츠로 넣는다 — `qwen_utils.py` 참고.

```bash
# sbatch로 (5개 샘플)
sbatch scripts/run_llm_analysis.sh

# 샘플 개수 바꾸고 싶으면
sbatch scripts/run_llm_analysis.sh --num_samples 10

# 영상 프레임을 더/덜 촘촘히 보게 하고 싶으면 (기본 fps=1.0, max_frames=32)
sbatch scripts/run_llm_analysis.sh --fps 2 --max_frames 60

# 6개 점수+백분위 비교 대신, OCEAN 성격특성+채용추천까지 포함한 자유서술형
# 한국어 평가를 받고 싶으면 (STRICT JSON 없음, raw_output 그대로 저장/출력)
sbatch scripts/run_llm_analysis.sh --prompt_style hr_assessment

# 위와 같은 평가 항목이지만 MediaPipe/librosa 수치 통계를 아예 안 뽑고,
# Qwen이 영상만 보고 판단하게 하는 가장 순수한 training-free 버전
sbatch scripts/run_llm_analysis.sh --prompt_style video_only
```

`--prompt_style`:
- `scoring`(기본): few-shot calibration + STRICT JSON(6점수+category+feedback), 백분위 비교표 생성. `build_sample_features()`로 MediaPipe/librosa 수치를 뽑아서 프롬프트에 넣음
- `hr_assessment`: `interview_prompts.build_hr_assessment_prompt()` — Overview/OCEAN 성격특성/Key Behavioral Moments/Interview Performance/Improvements/Hiring Recommendation을 한국어 자유서술로 받음. JSON이 아니라서 점수/백분위 비교는 못 만들고 `raw_output`만 저장됨. 이 모드도 MediaPipe/librosa 수치를 프롬프트에 같이 넣음(템플릿에 `{AU06_mean}` 등 placeholder가 있어서)
- `video_only`: `interview_prompts.build_video_only_assessment_prompt()` — hr_assessment와 같은 평가 항목이지만 수치 통계 placeholder를 다 빼고 transcript+질문만 텍스트로 줌. `build_sample_features()`(MediaPipe/librosa/OpenFace) 호출 자체가 없어서 video_features/audio_features 의존성이 전혀 없음 — Qwen이 영상만 보고 판단하는 가장 순수한 형태
- `hr_assessment`/`video_only` 결과에는 metadata.jsonl에 이미 있는 Gemini 서술형 평가(`gemini_summary`)도 같이 저장/출력돼서, Qwen 결과랑 같은 샘플 기준으로 바로 비교할 수 있음

동작:
1. metadata.jsonl에서 영상 파일이 실제 존재하는 샘플을 `--num_samples`개 고름
2. 각 샘플의 transcript를 정제하고, `text_features.extract_text_stats_for_prompt`
   (filler_words/repeated_words 등 사람이 읽기 쉬운 통계), `audio_features`,
   `video_features`(위 Feature 추출 파이프라인과 동일한 코드)로 오디오/영상
   **수치** 통계를 뽑음 (정확한 필러워드 개수, pitch Hz 등 영상만 보고는 알기
   힘든 값들 — 아래 4번에서 영상과 같이 "교차검증용"으로 씀)
3. `/ceph_data/kms78299/dummy_data_v2-2.json`에 있는 5개 손으로 만든 예시
   (우수/평균/미흡/내용↑비언어↓/비언어↑내용↓ — labels + text/audio/video 패턴,
   실제 영상이 없는 손으로 만든 데이터라 여기는 텍스트로만 줌)를 few-shot
   calibration으로 프롬프트에 그대로 넣음
4. Qwen에게 **실제 영상 파일**(화면은 `{"type": "video", ...}`, 오디오는
   우리가 직접 뽑은 wav를 `{"type": "audio", ...}`로 별도 전달)과 2번의 수치
   통계를 프롬프트 텍스트로 같이 주고, 같은 포맷으로 6개 면접 점수
   (`interview_score`, `answer_score`, `speaking_skills`, `confidence_score`,
   `facial_expression`, `overall_performance`) + 카테고리 + reasoning +
   **`feedback`**(`strengths`/`improvements`/`summary` — 영상에서 실제로 본 것에
   근거한 구체적인 면접 코칭 피드백)을 JSON으로 예측시킴
5. 결과를 `--output_path`(기본 `llm_outputs/interview_analysis.json`)에
   샘플별 `{ground_truth, ground_truth_percentile, prediction, prediction_percentile,
   raw_output, error}` 리스트로 저장하고, 콘솔에 예측값 vs 실제 라벨(z-score +
   백분위) 비교표 + 강점/개선점/총평을 출력

주의:
- dummy_data는 "이 정도 feature 패턴이면 이 정도 점수대"라는 calibration
  예시일 뿐, 실제 모델 학습에 쓰는 라벨이 아님 (그건 metadata.jsonl 쪽 z-score).
- 영상을 통째로 넣으면 텍스트만 넣을 때보다 프롬프트 토큰 수/GPU 메모리/처리
  시간이 훨씬 늘어난다. `--fps`, `--max_frames`로 조절 가능 (기본값은 30초
  클립 기준으로 보수적으로 잡아둠 — 더 촘촘히 보게 하고 싶으면 늘리되 OOM
  조심할 것).
- 처음엔 `use_audio_in_video=True`로 영상 안 오디오를 Qwen이 자동으로 뽑게
  했는데, 두 가지 실제 버그를 겪었음: (1) `load_audio_from_video`를 같이 안
  주면 오디오 자리표시 토큰만 생기고 실제 데이터가 안 채워져서 `StopIteration`,
  (2) 그걸 고쳐도 내부적으로 mp4를 librosa로 직접 디코딩하려다
  "`torchcodec` 설치하라"는 에러로 죽음. 그래서 지금은 오디오를 우리가
  ffmpeg으로 직접 wav로 뽑아서 별도 `{"type": "audio", ...}` 콘텐츠로 넣는
  방식으로 우회함 (video/audio가 프레임 단위로 정확히 인터리빙되진 않지만,
  화면+음성 둘 다 온전히 모델에 들어가는 건 동일).
- video_stats(수치)는 OpenFace가 아니라 MediaPipe proxy일 가능성이 높아서,
  프롬프트 안에 "정확한 수치보다 상대적 패턴 위주로 보라"는 안내를 넣어둠 —
  대신 실제 영상을 직접 보여주니 이 한계를 어느 정도 보완할 수 있음.
- `parse_json_object`가 모델 출력에서 JSON을 못 뽑아내면 `prediction=None`,
  `error`에 이유가 남고 `raw_output`에 원문이 그대로 저장되니 실패 샘플은
  거기서 확인.

### z-score를 0~100으로 보고 싶을 때 (`label_utils.py`)

RECRUITVIEW 라벨은 1~100점 같은 절대 척도가 아니라, 임상 심리학자들의
27,000+ 쌍비교(pairwise comparison) 데이터에서 Multinomial Logit +
nuclear norm regularization으로 역산한 **상대적** 잠재 능력치라서
(arXiv:2512.00450, GitHub `AI4A-lab/RecruitView` README 참고), z-score를
"원래 몇 점이었는지"로 복원하는 공식 자체가 없다.

대신 `label_utils.build_label_distributions()`가 metadata.jsonl **전체**
(2011개) 샘플의 라벨 분포를 정렬해두고, `zscore_to_percentile()`이 그 안에서
해당 값이 상위 몇 %인지를 0~100 백분위로 계산한다 (min-max나 정규분포 가정
없이, "다른 지원자 대비 상대적으로 어디쯤이냐"는 원래 라벨의 의미를 그대로
보존). `analyze_interviews_llm.py`는 이 백분위를 z-score와 같이 출력/저장한다.
