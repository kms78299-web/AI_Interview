"""
RECRUITVIEW feature extraction 공통 설정.
경로/피처 이름 등을 한 곳에서 관리한다.
"""

import os

# ─────────────────────────────────────────────
# 기본 경로 (모두 argparse로 override 가능, extract_features.py 참고)
# ─────────────────────────────────────────────
DEFAULT_DATASET_ROOT = "/local_datasets/RecruitView"
# metadata.jsonl은 리포 안(GANG2/)에 직접 두고 씀. 영상 파일 자체는 용량이 커서
# 계속 /local_datasets/RecruitView 쪽(GPU 노드 로컬 데이터)을 기본 video_root로 둠.
DEFAULT_METADATA_PATH = "/ceph_data/kms78299/repos/GANG2/metadata.jsonl"
DEFAULT_VIDEO_ROOT = DEFAULT_DATASET_ROOT  # file_name 필드가 "videos/xxx.mp4" 형태라 root만 붙이면 됨

# 과제 설명에 저장 경로가 두 군데(/ceph_data/kms78299/recruitview_features/ 와
# /ceph_data/kms78299/repos/GANG2/recruitview_features/) 로 다르게 적혀 있었음.
# 리포 내부에 결과물을 두는 쪽(GANG2/recruitview_features)을 기본값으로 하고,
# 실행 시 --output_dir 로 얼마든지 바꿀 수 있게 해둠.
DEFAULT_OUTPUT_DIR = "/ceph_data/kms78299/repos/GANG2/recruitview_features"
DEFAULT_LOG_DIR = "/ceph_data/kms78299/repos/GANG2/logs"
# "해석 불가능한" 임베딩(MLP 입력용)은 "해석 가능한" feature(recruitview_features/,
# LLM 프롬프트용)랑 폴더 자체를 분리해둠. video/audio/text 통계 vs wav2vec2 임베딩을
# 헷갈리지 않게 물리적으로 나누는 목적.
DEFAULT_EMBEDDING_DIR = "/ceph_data/kms78299/repos/GANG2/audio_embeddings"

# metadata.jsonl 안에서 라벨이 아닌 필드들 (나머지 숫자 필드는 전부 라벨로 취급)
METADATA_NON_LABEL_KEYS = {"id", "file_name", "question", "transcript"}

# ─────────────────────────────────────────────
# Video (OpenFace / MediaPipe 공용) — 순서 고정. npy 저장 순서 == 이 리스트 순서.
# ─────────────────────────────────────────────
VIDEO_FEATURE_NAMES = [
    "AU06_mean", "AU06_std",
    "AU12_mean", "AU12_std",
    "AU45_mean", "AU45_std",
    "AU04_mean", "AU04_std",
    "gaze_x_mean", "gaze_y_mean", "gaze_std",
    "pose_rx_mean", "pose_ry_mean", "pose_rz_mean", "pose_std",
]

# 프레임 전체를 다 돌면 30초 영상도 꽤 느려서, N프레임마다 하나씩만 샘플링.
VIDEO_FRAME_SAMPLE_STRIDE = 3

# OpenFace FeatureExtraction 바이너리. PATH에 있으면 이름만, 없으면 절대경로로 지정.
# 환경변수 OPENFACE_BIN 으로도 override 가능.
OPENFACE_BIN = os.environ.get("OPENFACE_BIN", "FeatureExtraction")

# ─────────────────────────────────────────────
# Audio (librosa) — 순서 고정.
# ─────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MFCC = 13

AUDIO_FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(AUDIO_N_MFCC)]
    + [f"mfcc{i}_std" for i in range(AUDIO_N_MFCC)]
    + ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "silence_ratio", "speech_rate"]
)

# wav2vec2 — 이 32차원 librosa 통계와는 별개로, MLP 입력용 "무거운" 오디오
# 임베딩. hidden size가 768이라 BERT 임베딩 차원과 우연히 같음.
#
# facebook/wav2vec2-base(순수 self-supervised)를 쓰고 싶었지만, 이 체크포인트는
# HF Hub에 safetensors 없이 구형 pytorch_model.bin(pickle)만 있어서, transformers가
# CVE-2025-32434 보안 패치로 "torch<2.6이면 그 포맷 로드 자체를 거부"하게 막아놨음
# (지금 이 환경의 torch는 GPU 드라이버 제약 때문에 2.5.1+cu121로 고정 중이라 업그레이드
# 대신 모델을 바꿈). facebook/wav2vec2-base-960h는 safetensors가 있어서 문제 없음 —
# ASR(960h Librispeech) 파인튜닝된 체크포인트지만 Wav2Vec2Model로 CTC head 없이
# encoder body만 쓰므로 임베딩 추출 용도로는 문제 없음.
DEFAULT_WAV2VEC2_MODEL_ID = "facebook/wav2vec2-base-960h"
WAV2VEC2_EMBEDDING_DIM = 768

# ─────────────────────────────────────────────
# Text (BERT) — 순서 고정.
# ─────────────────────────────────────────────
TEXT_BERT_DIM = 768
TEXT_STAT_NAMES = ["filler_word_count", "vocab_diversity", "answer_length"]
TEXT_FEATURE_NAMES = [f"bert_{i}" for i in range(TEXT_BERT_DIM)] + TEXT_STAT_NAMES

TEXT_MODEL_KOREAN = "klue/bert-base"
TEXT_MODEL_ENGLISH = "bert-base-uncased"

FILLER_WORDS = ["um", "umm", "uh", "uhh", "uhm", "hmm", "hm", "er", "erm", "ah"]

# transcript 안의 "[00:01 - 00:11]" 같은 타임스탬프 패턴
TIMESTAMP_PATTERN = r"\[\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*\]"

# ─────────────────────────────────────────────
# LLM (Qwen2.5-Omni) 기반 면접 분석 — analyze_interviews_llm.py
# ─────────────────────────────────────────────
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
# 참고용 예시(few-shot calibration) 데이터. "우수/평균/미흡/내용↑비언어↓/비언어↑내용↓"
# 5개 카테고리별로 labels/text/audio/video 값이 어떻게 나오는지 보여주는 손으로 만든 예시.
DEFAULT_DUMMY_DATA_PATH = "/ceph_data/kms78299/dummy_data_v2-2.json"
DEFAULT_LLM_OUTPUT_PATH = "/ceph_data/kms78299/repos/GANG2/llm_outputs/interview_analysis.json"

# dummy_data_v2-2.json의 "labels"에 있는 6개 면접 평가 지표만 예측 대상으로 삼는다.
# (metadata.jsonl에는 이 6개 + 성격 특성 6개, 총 12개가 있지만 dummy_data 포맷은
# 면접 퍼포먼스 6개만 다룸)
INTERVIEW_LABEL_NAMES = [
    "interview_score",
    "answer_score",
    "speaking_skills",
    "confidence_score",
    "facial_expression",
    "overall_performance",
]
