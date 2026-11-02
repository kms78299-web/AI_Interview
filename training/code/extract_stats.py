"""
Track 2 (stats): LLM 피드백용 해석 가능한 통계 추출 + 프롬프트 만들기.

영상 1개 -> {audio, text, video 통계}를 하나의 dict로 모은다. 서비스에서는 extract_stats(video_path, transcript)를 부르면 되고,
데이터셋 전체(2011개)는 아래 CLI로 features/stats/{id}.json에 저장한다. 저장한 값의 분포를 기준표(stats_reference.json)로 만들어 두면
프롬프트에서 "이 값이 RecruitView 참가자 중 상위 몇 %인지"까지 LLM에게 줄 수 있다 (임의 기준값 대신 데이터 분포로 해석).

사용 (영상 파일은 ariel-v7에만 있으므로 sbatch -w ariel-v7 로 실행, run_extract_stats.sh 참고):
    python extract_stats.py --limit 5            # 소량 확인
    python extract_stats.py                       # 전체 추출 -> features/stats/{id}.json
    python extract_stats.py --reference           # 저장된 값으로 stats_reference.json 만들기 (CPU, 영상 불필요)
"""

import argparse
import json
import logging
import os
import sys
import traceback

import numpy as np
from tqdm import tqdm

from config import DEFAULT_FEATURES_DIR, DEFAULT_LOG_DIR, DEFAULT_METADATA_PATH, DEFAULT_VIDEO_ROOT
from data_utils import clean_transcript, load_metadata, video_path_for_item

AUDIO_KEYS = ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "silence_ratio", "speech_rate"]


def _duration_sec(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if n else 0.0


def extract_stats(video_path, transcript):
    """영상 + 대본(STT 결과 그대로) -> {"audio": {...}, "text": {...}, "video": {...}} (사람이 읽을 수 있는 값)."""
    from audio_features import extract_audio_features
    from config import AUDIO_FEATURE_NAMES
    from text_features import extract_text_stats_for_prompt
    from video_stats import extract_video_stats

    vec = extract_audio_features(video_path)
    audio = {k: float(v) for k, v in zip(AUDIO_FEATURE_NAMES, vec) if k in AUDIO_KEYS}
    text = extract_text_stats_for_prompt(transcript)
    duration = _duration_sec(video_path)
    audio["duration_sec"] = round(duration, 1)
    audio["speaking_rate_wpm"] = round(text["answer_length"] / duration * 60, 1) if duration > 0 else 0.0
    return {"audio": audio, "text": text, "video": extract_video_stats(video_path)}


# ───────────── 기준표 (데이터셋 분포) ─────────────

NUMERIC_KEYS = {
    "audio": AUDIO_KEYS + ["duration_sec", "speaking_rate_wpm"],
    "text": ["filler_count", "answer_length", "vocab_diversity"],
    "video": ["face_detected_ratio", "gaze_off_center_mean", "gaze_away_ratio", "gaze_x_std", "gaze_y_std",
              "head_yaw_mean_abs", "head_pitch_mean", "head_roll_std", "head_motion", "blink_rate_per_min",
              "mouth_open_std", "mouth_width_std"],
}


def make_reference(stats_dir, out_path):
    """features/stats/*.json -> 값별 정렬된 분포. 프롬프트에서 백분위로 바꿀 때 쓴다."""
    files = [f for f in os.listdir(stats_dir) if f.endswith(".json")]
    if not files:
        raise SystemExit(f"{stats_dir}에 통계 json이 없음 — 먼저 python extract_stats.py")
    rows = [json.load(open(os.path.join(stats_dir, f), encoding="utf-8")) for f in files]
    ref = {"n_samples": len(rows), "values_sorted": {}}
    for group, keys in NUMERIC_KEYS.items():
        for k in keys:
            vals = sorted(float(r[group][k]) for r in rows if k in r.get(group, {}))
            if vals:
                ref["values_sorted"][f"{group}.{k}"] = vals
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ref, f)
    print(f"[INFO] 기준표 저장: {out_path} (샘플 {len(rows)}개, 항목 {len(ref['values_sorted'])}개)")


def to_percentile(reference, group, key, value):
    vals = reference["values_sorted"].get(f"{group}.{key}")
    if not vals:
        return None
    return int(round(100 * np.searchsorted(vals, value, side="right") / len(vals)))


# ───────────── LLM 프롬프트 ─────────────

STAT_LABELS = {
    "audio": {"speaking_rate_wpm": "말 속도(분당 단어 수)", "silence_ratio": "침묵 비율",
              "pitch_mean": "평균 음높이(Hz)", "rms_mean": "평균 음량", "rms_std": "음량 변동", "duration_sec": "답변 길이(초)"},
    "text": {"answer_length": "답변 단어 수", "filler_count": "군더더기 말(um, like 등) 횟수",
             "vocab_diversity": "어휘 다양성(서로 다른 단어 비율)"},
    "video": {"gaze_away_ratio": "시선이 정면에서 크게 벗어난 시간 비율", "gaze_x_std": "시선의 좌우 흔들림",
              "head_motion": "머리 움직임", "head_yaw_mean_abs": "고개를 좌우로 돌린 정도", "head_roll_std": "고개 갸웃거림",
              "blink_rate_per_min": "분당 눈 깜빡임(근사)", "mouth_width_std": "입 모양 변화(표정 변화의 근사)",
              "face_detected_ratio": "얼굴이 인식된 프레임 비율"},
}


# 데이터셋 기준표로 위치를 말하면 안 되는 항목: filler_count는 데이터셋 대본에서 군더더기 말이 이미 제거돼 거의 항상 0이고,
# face_detected_ratio는 전부 1.0이라 분포 정보가 없다. (값 자체는 프롬프트에 그대로 준다.)
# 제외한 항목: pitch_std(음높이 추적이 옥타브를 건너뛰는 오류가 섞임), gaze_y_std/gaze_off_center_mean(눈 감은 프레임에서 계산이 불안정).
NO_PERCENTILE = {"filler_count", "face_detected_ratio"}


FEEDBACK_JSON_EXAMPLE = json.dumps({
    "content_rating": "적절 | 부분적 | 부적절 중 하나",
    "content_reason": "내용 평가 이유 한 줄",
    "overall_band": "상 | 중 | 하 중 하나",
    "band_reason": "종합 평가 이유 한 줄 (모델 점수와 다르면 왜 다른지)",
    "text": {"strength": "...", "improvement": "..."},
    "audio": {"strength": "...", "improvement": "..."},
    "video": {"strength": "...", "improvement": "..."},
    "summary": "종합 조언 2~3문장",
}, ensure_ascii=False, indent=2)


def build_feedback_prompt(stats, model_result=None, question=None, reference=None):
    """통계(+선택: 모델 결과, 질문, 기준표) -> LLM에 줄 프롬프트 문자열.

    LLM에게 지키게 하는 원칙: (1) 관찰된 값만 근거로, (2) 값이 점수의 '원인'이라고 단정하지 않음, (3) 값의 의미 범위를 넘는 추측 금지."""
    lines = []
    for group in ("audio", "text", "video"):
        for key, label in STAT_LABELS[group].items():
            if key not in stats[group]:
                continue
            v = stats[group][key]
            pct = to_percentile(reference, group, key, v) if reference and key not in NO_PERCENTILE else None
            lines.append(f"- [{group}] {label}: {round(v, 3) if isinstance(v, float) else v}"
                         + (f" (참가자 {pct}번째 백분위, 100에 가까울수록 값이 큰 편)" if pct is not None else ""))
    t = stats["text"]
    extras = []
    if t.get("filler_words"):
        extras.append(f"자주 쓴 군더더기 말: {', '.join(t['filler_words'])}")
    if t.get("repeated_words"):
        extras.append(f"반복된 단어: {', '.join(t['repeated_words'])}")
    result_lines = []
    if model_result:
        for name, s in model_result["scores"].items():
            rng = (f", 80% 범위 {s['percentile_low']}~{s['percentile_high']}"
                   if s.get("percentile_low") is not None else "")
            result_lines.append(f"- {name}: 참가자 집단 내 {s['percentile']}번째 백분위({s['band']}){rng}")
    return "\n".join([
        "당신은 면접 연습 피드백을 쓰는 코치입니다. 아래는 면접 답변 영상에서 자동으로 측정한 값입니다.",
        *(["질문: " + question] if question else []),
        "", "[답변 대본]", t["transcript"] or "(대본 없음)", "",
        "[측정값]", *lines, *extras, "",
        *(["[참고: 예측 모델 결과 — 참가자 집단 대비 상대 위치이며 절대 점수가 아님, 정확도는 중간 수준. "
           "이 모델은 학습 데이터의 경향상 답변이 짧을수록 점수를 높게 주는 편이라, 내용의 적절성은 판단하지 못한다]", *result_lines, ""]
          if result_lines else ["[참고: 이 답변은 너무 짧거나 분석 조건에 맞지 않아 예측 모델 점수는 제공되지 않음]", ""]),
        "[작성 규칙]",
        "1. 측정값과 대본에서 실제로 관찰되는 것만 근거로 쓴다. 값에 없는 내용(감정, 성격, 합격 가능성)은 추측하지 않는다.",
        "2. 측정값이 점수의 '원인'이라고 단정하지 않는다. 예측 모델은 이 측정값을 직접 쓰지 않았다. '이런 패턴이 관찰되었다'로 표현한다.",
        "3. 얼굴 인식 비율이 낮거나(0.8 미만) 대본이 비어 있으면 해당 부분 해석은 신뢰하기 어렵다고 밝힌다.",
        "4. content_rating: 질문에 맞는 답인지 본다. 질문이 주어지지 않았으면 대본 자체의 완결성만 본다. "
        "적절=질문에 직접 답하고 내용이 있음 / 부분적=일부만 답하거나 구체성이 부족함 / "
        "부적절=질문을 이해하지 못했다고 답하거나, 질문과 무관하거나, 평가할 내용이 거의 없음.",
        "5. overall_band(상/중/하): 답변의 종합 수준(질문 적합성, 구체성, 전달의 안정감)에 대한 당신의 판단이다. "
        "content_rating이 '부적절'이면 반드시 '하', '부분적'이면 '상'은 줄 수 없다(중 이하). 확신이 없으면 '중'. "
        "예측 모델 점수는 참고만 하고, 이 모델은 내용 적절성을 못 보므로 점수가 높아도 내용이 나쁘면 '하'를 준다. "
        "점수와 overall_band가 크게 다르면 band_reason에 이유를 쓴다.",
        "6. 텍스트(내용), 음성(속도·쉼·음량·음높이), 영상(시선·고개·표정) 각각 강점 1개와 개선점 1개. 근거가 부족한 영역은 '관찰된 값이 부족함'이라고 쓴다.",
        "7. 한국어, 친근하지만 단정적이지 않은 말투.",
        "",
        "[출력 형식] 아래 JSON 하나만 출력한다(설명 문장, 코드블록 표시 없이). 키 이름과 값의 종류를 정확히 지킨다.",
        FEEDBACK_JSON_EXAMPLE,
    ])


# ───────────── 배치 CLI ─────────────

def run_batch(args):
    out_dir = os.path.join(args.features_dir, "stats")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = logging.getLogger("extract_stats")
    logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, "extract_stats_errors.log")))
    items = load_metadata(args.metadata_path)[args.start:args.end]
    if args.limit is not None:
        items = items[: args.limit]
    print(f"[INFO] 대상 샘플 {len(items)}개 -> {out_dir}")
    n_ok = n_skip = n_err = 0
    for item in tqdm(items, desc="stats"):
        out_path = os.path.join(out_dir, f"{item['id']}.json")
        if not args.overwrite and os.path.exists(out_path):
            n_skip += 1
            continue
        try:
            stats = extract_stats(video_path_for_item(item, args.video_root), clean_transcript(item.get("transcript")))
            stats["id"] = item["id"]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False)
            n_ok += 1
        except Exception as e:
            n_err += 1
            logger.error(f"id={item['id']}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            tqdm.write(f"[SKIP] id={item['id']}: {type(e).__name__}: {e}")
    print(f"[INFO] 완료: 성공 {n_ok}, 스킵 {n_skip}, 에러 {n_err}")


def main():
    p = argparse.ArgumentParser(description="LLM 피드백용 통계 추출")
    p.add_argument("--metadata_path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR)
    p.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reference", action="store_true", help="저장된 통계로 stats_reference.json 만들기")
    args = p.parse_args()
    if args.reference:
        make_reference(os.path.join(args.features_dir, "stats"), os.path.join(args.features_dir, "stats_reference.json"))
        return
    run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
