"""
새 영상(RecruitView 데이터셋 바깥의 영상) 하나를 넣으면 interview 성과 점수
6개를 반환하는 end-to-end 추론 코드.

학습/검증에 쓴 extract_features.py 등은 metadata.jsonl(이미 transcript가 있는
RecruitView 전용 메타데이터)을 전제로 하는 배치 처리용 코드라서, 임의의 영상
파일 하나를 바로 넣을 수 있는 용도로는 못 씀 — 그래서 이 파일을 따로 둠.

RecruitView 샘플과 다른 점: transcript가 없으므로 Whisper로 STT를 먼저 돌려서
텍스트를 만든다 (학습 때는 이미 주어진 transcript를 그대로 썼음, STT 과정 자체가
없었음 — 이 차이를 꼭 알고 있을 것).

사용 1) 다른 코드(Gradio/FastAPI 등)에서 import:
    from inference import run_inference
    result = run_inference("my_interview.mp4")   # 첫 호출에서만 모델 전부 로드, 이후 재사용

    체크포인트 폴더는 아래 순서로 찾는다:
      run_inference(..., checkpoint_dir=...) 인자 > 환경변수 MINI_CRMF_CHECKPOINT_DIR >
      <이 파일 위치>/../checkpoints/ 아래에 mlp_best.pt가 든 폴더가 정확히 1개일 때 그 폴더

사용 2) 같은 사람의 영상 여러 개(세션, 예: 질문 5개 답변)를 한 번에:
    from inference import run_session_inference
    result = run_session_inference(["q1.mp4", "q2.mp4", "q3.mp4", "q4.mp4", "q5.mp4"])
    각 영상의 점수를 "본인 예측 50% + 세션 평균 50%"로 섞어서 보정한다(같은 사람의 라벨이 비슷하다는 점을 이용,
    검증에서 순위 정확도가 올랐음). 같은 사람의 영상만 넣어야 한다. 영상이 1개면 단일 영상 결과와 같다.
    반환: {"session": {전체 점수}, "videos": [영상별 점수], "explanation"}

사용 3) CLI:
    python inference.py --video_path my_interview.mp4

반환: {"scores": {타깃: {"raw", "percentile"(보정된 기대 백분위 0~100), "percentile_low/high"(80% 구간), "band"(상/중/하),
      "rank_percentile"(보정 전 예측 순위 백분위)}}, "explanation", "transcript"}
(JSON 직렬화 가능). 백분위는 checkpoint_dir의 percentile_reference.json(val 예측값 분포)
기준이며, 절대 점수가 아니라 RecruitView 참가자 집단 대비 상대 위치다. 파일이 없으면
raw만 반환.
"""

import argparse
import json
import math
import os
import pickle
import re
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    AUDIO_EMBED_ENCODERS,
    INTERVIEW_LABEL_NAMES,
    TEXT_EMBED_ENCODERS,
    VIDEO_EMBED_ENCODERS,
)
from data_utils import clean_transcript  # noqa: E402
from audio_features import extract_wav2vec2_embedding, extract_wav2vec2_layer_embedding, load_wav2vec2  # noqa: E402
from text_features import extract_text_embedding, load_bert  # noqa: E402
from video_embed import extract_videomae_embedding, load_videomae  # noqa: E402
from label_utils import percentile_to_band, zscore_to_percentile  # noqa: E402
from train_mini_crmf import MiniCRMF  # noqa: E402

_MODELS = None  # 한 번 로드한 모델 묶음 (run_inference가 재사용)
_LOCK = threading.Lock()  # Gradio 등에서 동시에 호출돼도 GPU/모델 상태가 꼬이지 않게


def _default_checkpoint_dir():
    env = os.environ.get("MINI_CRMF_CHECKPOINT_DIR")
    if env:
        return env
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    candidates = []
    if os.path.isdir(root):
        candidates = [d for d in sorted(os.listdir(root))
                      if os.path.isfile(os.path.join(root, d, "mlp_best.pt"))
                      or os.path.isfile(os.path.join(root, d, "ensemble.json"))]
    if len(candidates) == 1:
        return os.path.join(root, candidates[0])
    raise RuntimeError(
        f"체크포인트 폴더를 자동으로 정하지 못함 (후보 {len(candidates)}개, 위치 {root}). "
        "run_inference(checkpoint_dir=...) 인자나 환경변수 MINI_CRMF_CHECKPOINT_DIR로 지정할 것."
    )


def _build_mlp(cfg):
    return MiniCRMF(
        modality_dims=cfg["modality_dims"],
        d_model=cfg["d_model"],
        fusion=cfg["fusion"],
        target_names=INTERVIEW_LABEL_NAMES,
        dropout=cfg["dropout"],
        n_transformer_layers=cfg["transformer_layers"],
        n_heads=cfg["n_heads"],
    )


def load_checkpoint(checkpoint_dir, device):
    """단일 체크포인트(mlp_best.pt) 또는 앙상블(ensemble.json의 members)을 읽어 모델 리스트로 반환."""
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(checkpoint_dir, "scaler.pkl"), "rb") as f:
        scalers = pickle.load(f)

    ens_path = os.path.join(checkpoint_dir, "ensemble.json")
    if os.path.exists(ens_path):
        with open(ens_path) as f:
            files = [m["file"] for m in json.load(f)["members"]]
    else:
        files = ["mlp_best.pt"]

    mlps = []
    for name in files:
        model = _build_mlp(cfg).to(device)
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, name), map_location=device))
        model.eval()
        mlps.append(model)
    print(f"[INFO] 체크포인트 로드 완료: 모델 {len(mlps)}개(앙상블이면 예측 평균), fusion={cfg['fusion']}, "
          f"audio 층={cfg.get('audio_layer', '마지막(12)')}")
    return mlps, scalers, cfg


def load_models(checkpoint_dir=None, whisper_model="base", device=None):
    """모든 모델(Whisper, VideoMAE, Wav2Vec2, BERT, 학습된 MLP)을 한 번에 로드해서 dict로 반환."""
    import whisper

    checkpoint_dir = checkpoint_dir or _default_checkpoint_dir()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 사용 불가 -> CPU로 진행")
        device = "cpu"

    print(f"[INFO] 모델 로딩 시작 (device={device}, checkpoint={checkpoint_dir})")
    mlps, scalers, cfg = load_checkpoint(checkpoint_dir, device)
    models = {
        "device": device,
        "checkpoint_dir": checkpoint_dir,
        "cfg": cfg,
        "mlps": mlps,
        "scalers": scalers,
        "whisper": whisper.load_model(whisper_model, device=device),
    }
    models["videomae"] = load_videomae(VIDEO_EMBED_ENCODERS["videomae"], device)
    models["wav2vec2"] = load_wav2vec2(AUDIO_EMBED_ENCODERS["wav2vec2-base"], device)
    models["bert"] = load_bert(TEXT_EMBED_ENCODERS["bert-base-uncased"], device)
    _load_finetuned_encoders(models, checkpoint_dir)
    print("[INFO] 모델 로딩 완료")
    return models


def _load_finetuned_encoders(models, checkpoint_dir):
    """train_finetune.py로 학습한 체크포인트면 fine-tuning된 encoder 층 가중치를 덮어쓴다
    (frozen 체크포인트에는 finetuned_encoders.pt가 없어서 아무 일도 안 함)."""
    path = os.path.join(checkpoint_dir, "finetuned_encoders.pt")
    if not os.path.exists(path):
        return
    hf_models = {"video": models["videomae"][1], "audio": models["wav2vec2"][1], "text": models["bert"][1]}
    for modality, entry in torch.load(path, map_location="cpu").items():
        result = hf_models[modality].load_state_dict(entry["state"], strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"{modality}: fine-tuning 가중치가 encoder와 맞지 않음 (unexpected {result.unexpected_keys[:3]})")
        print(f"[INFO] fine-tuned {modality} encoder 층 {entry['layer_indices']} 가중치 로드 ({len(entry['state'])}개 텐서)")


def transcribe(video_path, whisper_model, device, language="en"):
    """Whisper로 STT. RecruitView 학습 데이터는 transcript가 이미 주어져 있어서
    이 단계가 없었지만, 새 영상은 transcript가 없으므로 여기서 직접 만든다.

    language 기본값 "en": 학습에 쓴 텍스트 인코더(bert-base-uncased)와 학습 데이터가
    영어라서. 영어가 아닌 발화를 그냥 넣으면 임베딩이 의미 없는 값이 된다."""
    result = whisper_model.transcribe(video_path, language=language, fp16=(device == "cuda"))
    return clean_transcript(result["text"])


def _get_models(models, checkpoint_dir, whisper_model, device):
    """models를 안 주면 모듈 전역 캐시를 쓰고, 없으면 처음 한 번 로드한다. _LOCK을 잡은 상태에서 호출할 것."""
    global _MODELS
    if models is None:
        if _MODELS is None:
            _MODELS = load_models(checkpoint_dir, whisper_model, device)
        models = _MODELS
    return models


def transcribe_videos(video_paths, models=None, checkpoint_dir=None, whisper_model="base", device=None, language="en"):
    """영상 여러 개 -> 대본 텍스트 리스트 (video_paths와 같은 순서). STT만 하고 점수 분석은 하지 않는다.

    UI가 텍스트를 먼저 보여 준 뒤 같은 텍스트를 run_session_inference(video_paths, transcripts=...)에 넘기면
    STT를 다시 돌리지 않고 그 텍스트 그대로 점수 분석에 쓴다."""
    video_paths = list(video_paths)
    if not video_paths:
        raise ValueError("video_paths가 비어 있음")
    with _LOCK:
        models = _get_models(models, checkpoint_dir, whisper_model, device)
        return [transcribe(p, models["whisper"], models["device"], language=language) for p in video_paths]


def run_inference(video_path, models=None, checkpoint_dir=None, whisper_model="base",
                  device=None, language="en", transcript=None):
    """영상 1개 -> 점수 dict (JSON 직렬화 가능).

    models를 안 주면 모듈 전역에 캐시된 모델을 쓰고, 없으면 이 호출에서 처음 로드한다
    (이후 호출은 재사용 — 서버처럼 반복 호출해도 매번 다시 로드하지 않음).

    transcript: 대본 텍스트를 직접 주면 Whisper STT를 건너뛴다(학습 때와 같은 대본으로 정확히 비교하는 검증용)."""
    with _LOCK:
        models = _get_models(models, checkpoint_dir, whisper_model, device)
        return _predict(video_path, models, language, transcript)


def _predict(video_path, models, language, transcript=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일 없음: {video_path}")
    device = models["device"]

    # 1. STT (학습 때는 없었던, 새 영상 전용 추가 단계)
    if transcript is None:
        clean_text = transcribe(video_path, models["whisper"], device, language=language)
    else:
        clean_text = clean_transcript(transcript)

    # 2. frozen encoder로 임베딩 추출 (학습 때와 동일한 함수/방식)
    videomae_processor, videomae_model = models["videomae"]
    video_emb = extract_videomae_embedding(video_path, videomae_processor, videomae_model, device)
    wav2vec2_extractor, wav2vec2_model = models["wav2vec2"]
    audio_layer = models["cfg"].get("audio_layer")  # 앙상블 체크포인트는 6(중간 층), 이전 체크포인트는 없음(마지막 층)
    if audio_layer is None:
        audio_emb = extract_wav2vec2_embedding(video_path, wav2vec2_extractor, wav2vec2_model, device)
    else:
        audio_emb = extract_wav2vec2_layer_embedding(video_path, wav2vec2_extractor, wav2vec2_model, device,
                                                     layer=audio_layer)
    bert_tokenizer, bert_model = models["bert"]
    text_emb = extract_text_embedding(clean_text, bert_tokenizer, bert_model, device)

    for name, emb in [("video", video_emb), ("audio", audio_emb), ("text", text_emb)]:
        if not np.all(np.isfinite(emb)):
            raise ValueError(f"{name} 임베딩에 NaN/inf 있음 — 영상/오디오/STT 결과를 확인할 것")

    # 3. 학습 때 쓴 scaler로 정규화 (train 통계 기준 scaler를 그대로 재사용)
    scalers = models["scalers"]
    batch = {}
    for name, emb in [("text", text_emb), ("audio", audio_emb), ("video", video_emb)]:
        scaled = scalers[name].transform(emb.reshape(1, -1)).astype(np.float32)
        batch[name] = torch.from_numpy(scaled).to(device)

    # 4. MLP forward
    with torch.no_grad():  # 앙상블이면 멤버 예측의 평균 (학습 때 평가한 방식과 같음)
        pred = np.mean([m(batch).cpu().numpy()[0] for m in models["mlps"]], axis=0)

    raw = dict(zip(INTERVIEW_LABEL_NAMES, [float(v) for v in pred]))
    result = build_result(raw, models["checkpoint_dir"])
    result["transcript"] = clean_text
    return result


def _load_json(checkpoint_dir, name):
    path = os.path.join(checkpoint_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _calibrate(p, raw, z80):
    """정규점수 선형 보정(make_calibration.py): raw -> (기대 백분위, 80% 구간 하한/상한, 상/중/하).
    m = a + b*(raw-mu)/sd (실제 점수의 정규점수 기대값), 기대 백분위 = Phi(m/sqrt(1+sigma^2)), 구간 = Phi(m +- z80*sigma)."""
    m = p["a"] + p["b"] * (raw - p["mu"]) / p["sd"]
    s = p["sigma"]
    pct = 100 * _phi(m / math.sqrt(1 + s * s))
    lo, hi = 100 * _phi(m - z80 * s), 100 * _phi(m + z80 * s)
    c1, c2 = p["band_cuts"]
    return pct, lo, hi, ("하" if pct <= c1 else ("중" if pct <= c2 else "상"))


def _make_entry(raw, name, cal_mode, ref_sorted, z80):
    """raw 하나 -> 점수 dict. calibration.json이 있으면 보정 백분위(+80% 구간), 없으면 예전 방식(예측 순위 백분위)으로 대체.
    rank_percentile = '기준 예측 분포에서의 순위'(보정 전 값, 참고/검증용)."""
    entry = {"raw": round(float(raw), 4), "percentile": None, "percentile_low": None, "percentile_high": None,
             "band": None, "rank_percentile": None}
    if ref_sorted is not None:
        pr = zscore_to_percentile(float(raw), np.array(ref_sorted[name]))
        entry["rank_percentile"] = int(round(pr))
        entry["percentile"], entry["band"] = entry["rank_percentile"], percentile_to_band(pr)
    if cal_mode is not None:
        pct, lo, hi, band = _calibrate(cal_mode["targets"][name], float(raw), z80)
        entry.update(percentile=int(round(pct)), percentile_low=int(round(lo)), percentile_high=int(round(hi)), band=band)
    return entry


def _calibration_text(cal, mode):
    cv = cal[mode]["cv"]
    return ("점수는 100점 만점의 절대 실력이 아니라 RecruitView 참가자 집단 안에서의 상대적 위치(백분위)입니다. "
            "모델 예측과 실제 점수의 상관이 약 0.5라서, 예측이 극단적일수록 실제 점수는 평균 쪽에 가깝다는 점(평균으로의 회귀)을 반영해 "
            "보정한 '기대 백분위'이고, percentile_low~percentile_high는 80% 확률로 실제 점수가 들어갈 구간입니다"
            f"(검증에서 실제 포함률 약 {cv['coverage_80']:.0%}). "
            "예: 70 (범위 38~92)는 집단에서 약 70번째 백분위(상위 30%)일 것으로 기대되지만 그 범위 안 어디든 가능하다는 뜻입니다. "
            "구간이 넓으므로 숫자 하나보다 상/중/하 구간과 함께 참고용으로만 사용하세요.")


def run_session_inference(video_paths, smoothing=None, transcripts=None, **kwargs):
    """같은 사람의 영상 여러 개 -> 세션 점수 + 영상별 점수.

    각 영상의 raw 예측을 smoothing*(자기 예측) + (1-smoothing)*(세션 평균)으로 보정한 뒤, "같은 방식으로 당긴 예측"으로 만든
    보정 곡선(calibration.json의 session)으로 기대 백분위와 80% 구간을 계산한다. 세션 점수는 영상별 값의 평균.
    smoothing 기본값은 체크포인트에 기록된 값(0.5). **같은 사람의 영상만 넣을 것.**
    transcripts: 영상별 대본 리스트(주면 STT 생략). 나머지 kwargs는 run_inference로 전달."""
    video_paths = list(video_paths)
    if not video_paths:
        raise ValueError("video_paths가 비어 있음")
    if transcripts is not None and len(transcripts) != len(video_paths):
        raise ValueError("transcripts 길이가 video_paths와 다름")
    results = [run_inference(p, transcript=(None if transcripts is None else transcripts[i]), **kwargs)
               for i, p in enumerate(video_paths)]
    models = kwargs.get("models") or _MODELS
    ckpt = models["checkpoint_dir"]
    report = _load_json(ckpt, "session_report.json") or {}
    cal = _load_json(ckpt, "calibration.json")
    cal_s = cal.get("session") if cal else None
    z80 = cal["z80"] if cal else 1.2816
    a = smoothing if smoothing is not None else report.get("smoothing", 0.5)

    if len(results) == 1:  # 세션 보정은 영상이 2개 이상일 때만 의미가 있음 (단일 영상 결과 그대로)
        r = results[0]
        return {"session": {"n_videos": 1, "smoothing": None, "scores": r["scores"]},
                "videos": [{"video": video_paths[0], "transcript": r["transcript"], "scores": r["scores"]}],
                "explanation": r["explanation"]}

    names = list(results[0]["scores"])
    raw = np.array([[r["scores"][n]["raw"] for n in names] for r in results])  # (영상 수, 타깃 6)
    adjusted = a * raw + (1 - a) * raw.mean(axis=0, keepdims=True)
    ref_data = _load_json(ckpt, "percentile_reference_session.json")
    ref = ref_data["predictions_sorted"] if ref_data else None

    videos, per_video = [], []
    for i, r in enumerate(results):
        scores = {}
        for j, n in enumerate(names):
            entry = _make_entry(adjusted[i, j], n, cal_s, ref, z80)
            entry["raw"], entry["smoothed_raw"] = round(float(raw[i, j]), 4), round(float(adjusted[i, j]), 4)
            scores[n] = entry
        per_video.append(scores)
        videos.append({"video": video_paths[i], "transcript": r["transcript"], "scores": scores,
                       "single_clip_percentile": {n: r["scores"][n]["percentile"] for n in names}})

    session_scores = {}
    for j, n in enumerate(names):
        entry = {"raw": round(float(raw[:, j].mean()), 4), "percentile": None, "percentile_low": None,
                 "percentile_high": None, "band": None}
        pcts = [v[n]["percentile"] for v in per_video]
        if pcts[0] is not None:
            mean_pct = float(np.mean(pcts))
            entry["percentile"] = int(round(mean_pct))
            if cal_s is not None:
                entry["percentile_low"] = int(round(np.mean([v[n]["percentile_low"] for v in per_video])))
                entry["percentile_high"] = int(round(np.mean([v[n]["percentile_high"] for v in per_video])))
                c1, c2 = cal_s["targets"][n]["band_cuts"]
                entry["band"] = "하" if mean_pct <= c1 else ("중" if mean_pct <= c2 else "상")
            else:
                entry["band"] = percentile_to_band(mean_pct)
        session_scores[n] = entry

    gain = ""
    if "test_single_clip_macro" in report:
        gain = (f" 검증에서 같은 사람 영상끼리 이렇게 보정하면 순위 정확도(Spearman)가 {report['test_single_clip_macro']:.2f}에서 "
                f"{report['test_session_smoothed_macro']:.2f}로 올랐습니다(test 데이터 기준, 영상이 사람당 평균 6개일 때).")
    head = (f"같은 사람의 영상 {len(results)}개를 함께 평가한 결과입니다. 각 영상의 점수는 '그 영상의 예측 {a:.0%} + 이 사람 전체 영상 평균 "
            f"{1 - a:.0%}'로 보정했습니다.{gain} ")
    if cal_s is not None:
        explanation = head + _calibration_text(cal, "session") + " 세션 점수와 범위는 영상별 값의 평균(근사)입니다."
    elif ref is not None:
        explanation = head + "점수는 100점 만점의 절대 실력이 아니라 RecruitView 참가자 집단 대비 상대적 위치(백분위)이며, 참고용으로만 사용하세요."
    else:
        explanation = head + "(calibration.json과 percentile_reference_session.json이 없어 백분위는 계산하지 못했습니다.)"
    return {"session": {"n_videos": len(results), "smoothing": a, "scores": session_scores},
            "videos": videos, "explanation": explanation}


MIN_WORDS, MIN_SECONDS = 13, 7.0  # 데이터셋 하위 5% 기준. 이보다 짧은 답변은 점수를 내지 않는다(아래 설명 참고)


def _too_short(text, video_path, min_words, min_seconds):
    """점수를 내면 안 되는 이유 리스트(비어 있으면 통과).

    학습 데이터에서 답변이 짧을수록 정답 점수가 높았고(13단어 미만은 정답 평균 백분위 85), 모델이 그 경향을 배워서
    아주 짧은 답변에 높은 점수를 준다. 이런 입력은 점수가 의미 없어서 점수 대신 이유를 돌려준다."""
    reasons = []
    n_words = len(re.findall(r"[A-Za-z가-힣']+", text or ""))
    if min_words and n_words < min_words:
        reasons.append(f"대본이 {n_words}단어로 너무 짧음(기준 {min_words}단어 이상)")
    if min_seconds:
        from extract_stats import _duration_sec
        duration = _duration_sec(video_path)
        if 0 < duration < min_seconds:
            reasons.append(f"영상이 {duration:.1f}초로 너무 짧음(기준 {min_seconds:.0f}초 이상)")
    return reasons


def analyze_session(video_paths, transcripts=None, questions=None, with_stats=True, guard=True,
                    min_words=MIN_WORDS, min_seconds=MIN_SECONDS, **kwargs):
    """영상 N개(질문 순서) -> run_session_inference 결과 + 영상별 통계(stats) + LLM 프롬프트(llm_prompt).

    transcripts: STT 텍스트(transcribe_videos 결과)를 주면 그대로 재사용한다. 없으면 STT부터 한다.
    questions: 질문 텍스트 리스트(선택, video_paths와 같은 순서). 주면 프롬프트에 질문이 들어간다.
    with_stats=False면 통계/프롬프트 없이 점수 분석만 한다.
    guard=True(기본): 너무 짧은 답변(min_words 단어 미만 또는 min_seconds초 미만)은 점수를 내지 않고
      scores=None, skipped=True, warning에 이유를 담는다. 이런 영상은 세션 점수 계산에서도 제외한다.
      그래도 with_stats면 통계와 llm_prompt(점수 없는 버전)는 만들어서 LLM이 내용을 평가할 수 있게 한다.
    통계 추출이 한 영상에서 실패해도 전체 분석은 유지하고, 그 영상은 stats=None, stats_error에 이유를 담는다."""
    video_paths = list(video_paths)
    if questions is not None and len(questions) != len(video_paths):
        raise ValueError("questions 길이가 video_paths와 다름")
    model_kwargs = {k: kwargs[k] for k in ("models", "checkpoint_dir", "whisper_model", "device", "language") if k in kwargs}

    skipped = {}
    if guard:
        if transcripts is None:
            transcripts = transcribe_videos(video_paths, **model_kwargs)
        for i, path in enumerate(video_paths):
            reasons = _too_short(transcripts[i], path, min_words, min_seconds)
            if reasons:
                skipped[i] = reasons
    valid = [i for i in range(len(video_paths)) if i not in skipped]

    if not skipped:
        result = run_session_inference(video_paths, transcripts=transcripts, **kwargs)
    else:
        sub = (run_session_inference([video_paths[i] for i in valid], transcripts=[transcripts[i] for i in valid], **kwargs)
               if valid else None)
        videos, vi = [], 0
        for i, path in enumerate(video_paths):
            if i in skipped:
                videos.append({"video": path, "transcript": clean_transcript(transcripts[i]), "scores": None,
                               "single_clip_percentile": None, "skipped": True, "warning": "; ".join(skipped[i])})
            else:
                videos.append(sub["videos"][vi])
                vi += 1
        note = (f"{len(skipped)}개 답변은 너무 짧아 점수를 내지 않았고 세션 점수 계산에서도 제외했습니다"
                "(이 모델은 학습 데이터의 경향상 아주 짧은 답변에 점수를 높게 주기 때문입니다).")
        result = {"session": ({**sub["session"], "n_skipped": len(skipped)} if sub else
                              {"n_videos": 0, "n_skipped": len(skipped), "smoothing": None, "scores": None}),
                  "videos": videos, "explanation": ((sub["explanation"] + " ") if sub else "") + note}
    if not with_stats:
        return result

    from extract_stats import build_feedback_prompt, extract_stats
    models = kwargs.get("models") or _MODELS
    checkpoint_dir = (models or {}).get("checkpoint_dir") or kwargs.get("checkpoint_dir") or _default_checkpoint_dir()
    reference = _load_json(checkpoint_dir, "stats_reference.json")
    for i, video in enumerate(result["videos"]):
        try:
            stats = extract_stats(video_paths[i], video["transcript"])
            video["stats"] = stats
            video["llm_prompt"] = build_feedback_prompt(
                stats, {"scores": video["scores"]} if video["scores"] else None,
                None if questions is None else questions[i], reference)
        except Exception as e:  # 얼굴 미검출 등: 이 영상의 통계만 포기
            video["stats"], video["llm_prompt"] = None, None
            video["stats_error"] = f"{type(e).__name__}: {e}"
    return result


def build_result(raw_scores, checkpoint_dir):
    """모델 raw 출력 -> 보정 백분위(0~100) + 80% 구간 + 상/중/하 + 설명문.

    우선순위: calibration.json(보정, make_calibration.py) > percentile_reference.json(예측 순위) > raw만 반환."""
    ref_data = _load_json(checkpoint_dir, "percentile_reference.json")
    ref = ref_data["predictions_sorted"] if ref_data else None
    cal = _load_json(checkpoint_dir, "calibration.json")
    cal_single = cal.get("single") if cal else None
    z80 = cal["z80"] if cal else 1.2816
    scores = {n: _make_entry(v, n, cal_single, ref, z80) for n, v in raw_scores.items()}

    if cal_single is not None:
        explanation = _calibration_text(cal, "single")
    elif ref is None:
        explanation = ("calibration.json과 percentile_reference.json이 없어 0~100 변환을 하지 못했고 raw 값만 반환합니다. "
                       "raw는 절대 점수가 아니라 평균 0 근처의 상대적 z-score 성격 값입니다.")
    else:
        explanation = ("점수는 100점 만점의 절대 실력이 아니라 RecruitView 참가자 집단 대비 상대적 위치(백분위, 예측 순위 기준)입니다. "
                       "모델 정확도가 중간 수준이라 개별 점수는 오차가 큽니다. 상/중/하 구간과 함께 참고용으로만 사용하세요.")
    return {"scores": scores, "explanation": explanation}


def _load_test_spearman(checkpoint_dir):
    path = os.path.join(checkpoint_dir, "metrics_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["test_spearman_agg"]["spearman_macro"]["mean"]


def parse_args():
    p = argparse.ArgumentParser(description="새 영상 1개 -> interview 성과 점수 6개 추론")
    p.add_argument("--video_path", required=True, help="추론할 영상 파일 경로 (mp4 등)")
    p.add_argument("--checkpoint_dir", default=None,
                    help="mlp_best.pt, config.json, scaler.pkl이 있는 폴더 (생략 시 자동 탐색)")
    p.add_argument("--whisper_model", default="base",
                    help="STT용 Whisper 모델 크기: tiny/base/small/medium/large (클수록 정확하지만 느림)")
    p.add_argument("--language", default="en", help="STT 언어 코드 (학습 데이터가 영어라 기본 en)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    result = run_inference(args.video_path, checkpoint_dir=args.checkpoint_dir,
                           whisper_model=args.whisper_model, device=args.device, language=args.language)

    print("\n[RESULT]")
    for name, v in result["scores"].items():
        if v["percentile"] is None:
            print(f"  {name}: raw={v['raw']}")
        else:
            print(f"  {name}: {v['percentile']}점 ({v['band']})  [raw={v['raw']}]")
    print("\n[STT]\n" + result["transcript"][:300])
    print("\n[설명]\n" + result["explanation"])
    print("\n[JSON]\n" + json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
