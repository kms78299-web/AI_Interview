"""
Audio 특징 추출: librosa 통계(LLM 프롬프트용)와 wav2vec2 임베딩(MLP 입력용). 둘 다 ffmpeg으로 영상에서 오디오를 뽑아 쓴다.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

from config import AUDIO_FEATURE_NAMES, AUDIO_N_MFCC, AUDIO_SAMPLE_RATE, WAV2VEC2_EMBEDDING_DIM

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


def _extract_wav(video_path, wav_path, sr=AUDIO_SAMPLE_RATE):
    if shutil.which(FFMPEG_BIN) is None:
        raise RuntimeError(f"ffmpeg을 찾을 수 없음 (FFMPEG_BIN={FFMPEG_BIN})")
    cmd = [
        FFMPEG_BIN, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
        "-loglevel", "error", wav_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 오디오 추출 실패: {result.stderr.decode(errors='ignore')[:500]}")
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise RuntimeError(f"ffmpeg이 빈 wav를 생성함 (오디오 트랙이 없을 수 있음): {video_path}")


def extract_audio_features(video_path, sr=AUDIO_SAMPLE_RATE):
    import librosa

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = os.path.join(tmp_dir, "audio.wav")
        _extract_wav(video_path, wav_path, sr=sr)

        y, sr = librosa.load(wav_path, sr=sr, mono=True)
        if y.size == 0:
            raise RuntimeError(f"오디오 신호 길이가 0: {video_path}")

        feats = {}

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=AUDIO_N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        for i in range(AUDIO_N_MFCC):
            feats[f"mfcc{i}_mean"] = float(mfcc_mean[i])
            feats[f"mfcc{i}_std"] = float(mfcc_std[i])

        # Pitch (F0) — pyin, 무성 구간은 nan으로 나옴
        f0, voiced_flag, _voiced_prob = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
        if voiced_f0.size > 0:
            feats["pitch_mean"] = float(np.mean(voiced_f0))
            feats["pitch_std"] = float(np.std(voiced_f0))
        else:
            # 목소리가 감지되지 않은 구간(무음/노이즈뿐인 클립) -> 0으로 채움
            feats["pitch_mean"] = 0.0
            feats["pitch_std"] = 0.0

        # RMS (음량)
        rms = librosa.feature.rms(y=y)[0]
        feats["rms_mean"] = float(np.mean(rms))
        feats["rms_std"] = float(np.std(rms))

        # 침묵 비율: non-silent 구간 길이 합 / 전체 길이
        intervals = librosa.effects.split(y, top_db=30)
        non_silent_len = sum(end - start for start, end in intervals)
        feats["silence_ratio"] = float(1.0 - non_silent_len / y.size) if y.size > 0 else 1.0

        # 말 속도 proxy: onset 개수 / 길이(초). 실제 ASR 없이 음절 수를 셀 수
        # 없어서 onset(발화 시작점) 빈도로 근사한다.
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration_sec = y.size / sr
        feats["speech_rate"] = float(len(onset_frames) / duration_sec) if duration_sec > 0 else 0.0

        return np.array([feats[name] for name in AUDIO_FEATURE_NAMES], dtype=np.float32)


def load_wav2vec2(model_id, device):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id).to(device).eval()
    return feature_extractor, model


def extract_wav2vec2_embedding(video_path, feature_extractor, model, device, sr=AUDIO_SAMPLE_RATE):
    """wav2vec2 마지막 층의 hidden state를 시간축으로 평균내 768차원 벡터로 만든다."""
    import librosa
    import torch

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = os.path.join(tmp_dir, "audio.wav")
        _extract_wav(video_path, wav_path, sr=sr)
        y, sr = librosa.load(wav_path, sr=sr, mono=True)

    if y.size == 0:
        raise RuntimeError(f"오디오 신호 길이가 0: {video_path}")

    inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state[0]  # (T, WAV2VEC2_EMBEDDING_DIM)
    embedding = hidden.mean(dim=0).cpu().numpy().astype(np.float32)
    assert embedding.shape[0] == WAV2VEC2_EMBEDDING_DIM, (
        f"wav2vec2 hidden size가 예상({WAV2VEC2_EMBEDDING_DIM})과 다름: {embedding.shape[0]}. "
        "config.DEFAULT_WAV2VEC2_MODEL_ID를 바꿨으면 WAV2VEC2_EMBEDDING_DIM도 맞춰줘야 함."
    )
    return embedding


def extract_wav2vec2_layer_embedding(video_path, feature_extractor, model, device, layer=6, sr=AUDIO_SAMPLE_RATE):
    """지정한 층(0~12)의 hidden state를 시간축으로 평균낸 768차원 벡터. 마지막 층은 음성 인식 쪽으로 치우쳐서 중간 층을 쓴다."""
    import librosa
    import torch

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = os.path.join(tmp_dir, "audio.wav")
        _extract_wav(video_path, wav_path, sr=sr)
        y, sr = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        raise RuntimeError(f"오디오 신호 길이가 0: {video_path}")

    inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True).hidden_states
    if not 0 <= layer < len(hidden):
        raise ValueError(f"layer={layer}는 0~{len(hidden) - 1} 범위여야 함")
    emb = hidden[layer][0].mean(dim=0).cpu().numpy().astype(np.float32)
    assert emb.shape[0] == WAV2VEC2_EMBEDDING_DIM, f"hidden size {emb.shape[0]} != {WAV2VEC2_EMBEDDING_DIM}"
    return emb


def audio_vector_to_nested_dict(vec):
    """extract_audio_features()가 반환한 flat 32차원 벡터를 사람이 읽기 좋은
    중첩 dict로 바꾼다 (mfcc_mean/std를 13개짜리 리스트로 묶음). LLM 프롬프트
    표시용 — AUDIO_FEATURE_NAMES 순서(mean 13개 -> std 13개 -> 나머지)에 의존.
    """
    vec = [round(float(v), 4) for v in vec]
    return {
        "mfcc_mean": vec[0:AUDIO_N_MFCC],
        "mfcc_std": vec[AUDIO_N_MFCC:2 * AUDIO_N_MFCC],
        "pitch_mean": vec[2 * AUDIO_N_MFCC],
        "pitch_std": vec[2 * AUDIO_N_MFCC + 1],
        "rms_mean": vec[2 * AUDIO_N_MFCC + 2],
        "rms_std": vec[2 * AUDIO_N_MFCC + 3],
        "silence_ratio": vec[2 * AUDIO_N_MFCC + 4],
        "speech_rate": vec[2 * AUDIO_N_MFCC + 5],
    }
