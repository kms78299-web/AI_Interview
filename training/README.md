# GANG2 — RecruitView 면접 점수 모델 학습

RecruitView 면접 영상(2011개, 참가자 331명)으로 면접 점수 6개(interview, answer, speaking, confidence, facial_expression, overall)를 예측하는 모델을 학습한다.
서비스용 추론 패키지는 `model/`(Colab, Gradio API)에 있고, 여기에는 특징 추출, 학습, 기준표 생성, 분석 실험 코드가 있다.

## 모델 요약

- 입력: 고정된 인코더의 임베딩 3개(각 768차원). 영상은 VideoMAE(16프레임 균등 샘플, 토큰 평균), 음성은 wav2vec2-base-960h의 6번째 층(시간 평균), 대본은 bert-base-uncased(토큰 평균, 최대 512)다.
- 구조: MiniCRMF. 모달리티별 투영(768→256) → 모달리티 임베딩 + 1층 Transformer + attention pooling → 잔차 MLP → 공통 MLP → 점수별 출력층 6개.
- 학습: 라벨을 train으로 표준화하고 1.5σ 밖을 부드럽게 누름(soft winsorization). Huber loss, AdamW, OneCycleLR, dropout 0.3, weight decay 0.05, val macro Spearman으로 early stopping. seed 5개 앙상블.
- 분할: `user_no` 기준 70/15/15, seed 42 (`splits/split_seed42.json`, train 1411 / val 296 / test 304). 같은 사람은 한 split에만 있다. test는 모델 선택 후 한 번만 평가한다.

## 결과 (Spearman, 6개 점수 평균)

| 모델 | val | test |
|---|---|---|
| 처음 기본 모델 (wav2vec2 마지막 층, dropout 0.1) | 0.384 | 0.486 |
| **최종 앙상블** (`checkpoints/final_audio6_R1_ens5`) | **0.491** | **0.532** |
| 단일 모델 5개 평균 | 0.463 ± 0.015 | 0.504 ± 0.016 |

학습에 안 쓴 사람 99명으로 본 사용 방식별 정확도:

| 사용 방식 | Spearman |
|---|---|
| 사람 단위 (한 사람의 답변 평균) | 약 0.73 (3등분 상/중/하 일치 약 64%) |
| 답변 하나씩 | 약 0.53~0.57 |
| 같은 사람의 답변끼리 | 0.15~0.18 (길이 영향을 빼면 약 0.07) |

## 확인된 한계

- 라벨은 평가자의 쌍비교로 추정한 상대 점수이고, 답변 길이와 반비례한다(단어 수와 -0.49). 대부분 사람 단위 효과다(사람 평균끼리 -0.63, 같은 사람 안 -0.19).
- 모델은 답변 단위에서 라벨보다 길이에 더 기댄다(예측과 단어 수 -0.79). 임베딩만으로 단어 수를 맞히면 text 0.99, audio 0.96, video 0.45라서 입력에서 text를 빼도 길이 의존은 남는다.
- 그래서 서비스에서는 사람 단위 참고 신호로만 쓰고, 답변 비교와 내용 평가는 규칙 마커와 LLM이 맡는다(`model/API.md` 2장).

전달력 전용 모델 실험(`train_delivery.py`, speaking·confidence·facial 3개, val / test):

| 모델 | 원래 라벨 | 길이 뺀 라벨 | 단어 수와 상관 | 같은 사람 안 | 사람 단위 |
|---|---|---|---|---|---|
| 최종 모델 (text+audio+video) | 0.49 / 0.53 | 0.17 / 0.15 | -0.70 / -0.79 | 0.18 / 0.15 | 0.71 / 0.69 |
| audio+video, 원래 라벨 | 0.46 / 0.49 | 0.16 / 0.12 | -0.66 / -0.77 | 0.14 / 0.11 | 0.69 / 0.68 |
| audio+video, 길이 뺀 라벨로 학습 | 0.14 / 0.24 | 0.26 / 0.33 | +0.26 / +0.10 | 0.02 / -0.03 | 0.23 / 0.33 |

text를 빼도 성능은 거의 같고 길이 의존도 그대로다. 길이 뺀 라벨로 학습하면 길이 의존은 사라지지만 신호가 약하다.

같은 사람 답변 순위 학습 실험(`train_within_rank.py`, 회귀 + λ × 같은 사람 쌍 순위 손실, 6개 점수, val / test):
같은 사람 쌍 정확도(라벨 차이 0.5 초과)는 최종 모델 59.6% / 59.1%, '짧은 답변이 낫다' 규칙 56.2% / 54.9%였다.
순위 손실 비중을 높이면 val에서 57.8%(λ=0) → 56.7%(0.5) → 55.3%(1)로 떨어져 개선되지 않았다. 같은 사람 안에서 길이 외 신호는 약하게만 있다(규칙 대비 +3~4%p, test 95% 구간 +0.7~+7.4%p).

## 측정값과 평가자 라벨의 관계 (LLM 프롬프트 해석 기준)

`code/analyze_stat_labels.py`: train(232명)에서 기준을 정하고 val(50명)에서 방향을 확인했다. test는 쓰지 않았다.
주 지표는 사람 평균끼리, 사람 평균 단어 수(log)를 뺀 편 Spearman이다. 판정 기준(미리 정함): train bootstrap 95% 구간이 0을 포함하지 않고 |r| ≥ 0.15, val에서 같은 부호.

| 측정값 | 사람 단위, 길이 보정 (train, 6개 라벨 범위) | val | 같은 사람 안 | 판정 |
|---|---|---|---|---|
| 평균 음량 | -0.20 ~ -0.35 | -0.30 ~ -0.41 | ≈ 0 | 높을수록 낮게 평가 (6개 모두) |
| 음량 변동 | -0.11 ~ -0.30 | -0.16 ~ -0.35 | ≈ 0 | 4개 라벨 |
| 음높이 | -0.18 ~ -0.26 | -0.35 ~ -0.44 | ≈ 0 | 6개 모두 |
| 눈 깜빡임 | -0.15 ~ -0.21 | -0.17 ~ -0.36 | ≈ 0 | 6개 모두 |
| 입 모양 변화 | -0.16 ~ -0.25 | +0.01 ~ -0.15 | ≈ 0 | 5개 라벨 |
| 말 속도 | -0.16 ~ -0.26 | -0.07 ~ +0.05 | ≈ 0 | 5개 라벨이지만 val에서 거의 0 |
| 고개 좌우 | +0.08 ~ +0.17 | +0.10 ~ +0.28 | ≈ 0 | confidence 하나만 (여러 번 검정한 우연 가능) |
| 침묵 비율, 시선 이탈·흔들림, 머리 움직임, 고개 기울기, 얼굴 인식 | 약 -0.14 ~ +0.13 | | ≈ 0 | 관계 약함 |
| 어휘 다양성 | -0.04 ~ -0.09 (보정 전 +0.42 ~ +0.52) | | ≈ 0 | 길이 효과였음 |
| 군더더기 말 | | | | 판단 불가 (데이터셋 대본에서 제거됨) |

- 같은 사람 안 편상관은 모든 측정값·라벨에서 -0.07 ~ +0.03이다. 측정값으로 같은 사람의 답변끼리 우열을 가릴 근거는 없다.
- 경향이 있는 값(음량, 음높이, 눈 깜빡임, 입 모양)은 목소리·습관 같은 개인차나 녹화 환경의 차이일 수 있어서, 프롬프트에서는 방향을 코칭으로 쓰지 않고 "단정하지 않는 값"으로 넣었다(`session_prompt.py` 규칙 7).

## 실행 순서

환경은 conda `recruitview`(torch 2.5.1+cu121, transformers 5.17.0, mediapipe 0.10.21, librosa 0.11.0, scikit-learn 1.7.2, numpy 1.26)이다.
영상은 ariel-v7의 `/local_datasets/RecruitView`에만 있어서 **영상을 읽는 작업은 `-w ariel-v7`** 로 돌린다. 캐시된 임베딩만 쓰는 학습은 어느 노드에서나 된다.
경로는 `code/config.py`에 고정돼 있다.

| 순서 | 실행 | 결과 |
|---|---|---|
| 1 | `python code/build_meta.py` | `features/meta.csv` (id, user_no, 라벨) |
| 2 | `sbatch scripts/run_extract.sh --track embed --modality {text,video} --encoder {bert-base-uncased,videomae}` | `features/embed/{encoder}/{id}.npy` |
| 3 | `sbatch scripts/run_cache_raw.sh --modality audio` | `features/raw/audio_wave/{id}.npy` (16kHz 파형) |
| 4 | `sbatch scripts/run_extract_layers.sh --what audio` | `features/embed_layers/audio/wav2vec2/{id}.npy` (층별 평균·표준편차) |
| 5 | `sbatch scripts/run_final.sh` | `checkpoints/final_audio6_R1_ens5/` (seed 5개, scaler, 성능 기록) |
| 6 | `python code/make_calibration.py`, `make_session_reference.py`, `make_person_reference.py` | `model/` 패키지의 점수 보정·세션·사람 단위 기준표 |
| 7 | `sbatch scripts/run_extract_stats.sh` 후 `python code/extract_stats.py --reference` | `features/stats/{id}.json`, `stats_reference.json` (LLM 피드백용 통계와 분포) |
| 8 | `sbatch scripts/run_parity_reference.sh` | Colab 추론이 학습과 같은 값을 내는지 비교할 기준 |

실험용:
- `sbatch scripts/run_train_mini_crmf.sh --encoders ... --fusion {prefusion,concat}`: 처음 기본 모델과 fusion 비교
- `sbatch scripts/run_delivery.sh --label_mode {raw,length_resid}`: 전달력 전용 모델
- `sbatch scripts/run_within_rank.sh`: 같은 사람 답변 순위 학습
- `python code/analyze_stat_labels.py`: 측정값과 평가자 라벨의 관계 (CPU, 몇 분)

## 폴더

| 경로 | 내용 |
|---|---|
| `code/` | 특징 추출(`extract_features.py`, `extract_layers.py`, `cache_raw_inputs.py`), 학습(`train_mini_crmf.py`, `train_final_ensemble.py`, `train_delivery.py`, `train_within_rank.py`), 기준표(`make_*.py`), LLM용 통계와 프롬프트(`extract_stats.py`, `video_stats.py`, `segment_stats.py`, `session_prompt.py`) |
| `scripts/` | sbatch 래퍼 |
| `splits/` | 사람 단위 분할 |
| `features/`, `checkpoints/`, `logs/` | 생성물 (Git에 올리지 않음) |

`video_stats.py`, `audio_features.py`, `extract_stats.py`, `segment_stats.py`, `session_prompt.py`는 `model/code/`와 같은 파일이다. 한쪽을 고치면 다른 쪽도 맞춘다.
