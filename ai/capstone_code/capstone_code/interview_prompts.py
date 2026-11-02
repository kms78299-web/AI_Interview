"""
면접 분석 프롬프트 구성.

dummy_data_v2-2.json에 있는 5개 손으로 만든 예시(우수/평균/미흡/내용↑비언어↓/
비언어↑내용↓)를 few-shot calibration으로 그대로 프롬프트에 넣고, 그 아래에
실제 샘플의 transcript + text/audio/video feature(수치 통계)를 붙인다.
calibration 예시는 실제 영상이 없는 손으로 만든 데이터라 텍스트로만 주지만,
실제 분석 대상 후보자는 analyze_interviews_llm.py가 이 프롬프트(텍스트)와
함께 영상 파일 자체({"type": "video", "path": ...})도 Qwen한테 같이 넘겨서,
모델이 숫자 통계만 보고 짐작하는 게 아니라 실제 화면(표정/자세)과 음성(톤/속도)을
직접 보고 듣고 판단하게 한다.
"""

import json

from config import INTERVIEW_LABEL_NAMES

_META_KEYS = {"_note", "_label_stats"}


def load_calibration_examples(dummy_data_path):
    with open(dummy_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k not in _META_KEYS}


def _format_example(name, entry):
    labels_str = json.dumps(entry.get("labels", {}), ensure_ascii=False)
    text = entry.get("text", {})
    audio = entry.get("audio", {})
    video = entry.get("video", {})
    return (
        f"### Example: {name} ({entry.get('label', '')})\n"
        f"labels: {labels_str}\n"
        f"transcript: \"{text.get('transcript', '')}\"\n"
        f"text_stats: filler_count={text.get('filler_count')}, "
        f"vocab_diversity={text.get('vocab_diversity')}, "
        f"answer_length={text.get('answer_length')}, "
        f"repeated_words={text.get('repeated_words')}\n"
        f"audio_stats: pitch_mean={audio.get('pitch_mean')}, pitch_std={audio.get('pitch_std')}, "
        f"rms_mean={audio.get('rms_mean')}, silence_ratio={audio.get('silence_ratio')}, "
        f"speech_rate={audio.get('speech_rate')}\n"
        f"video_stats: AU06_mean(smile-cheek)={video.get('AU06_mean')}, "
        f"AU12_mean(smile-lip)={video.get('AU12_mean')}, AU45_mean(blink)={video.get('AU45_mean')}, "
        f"AU04_mean(brow-furrow)={video.get('AU04_mean')}, gaze_std={video.get('gaze_std')}, "
        f"pose_std={video.get('pose_std')}\n"
    )


def build_calibration_block(calibration_examples):
    blocks = [_format_example(name, entry) for name, entry in calibration_examples.items()]
    return "\n".join(blocks)


def build_analysis_prompt(calibration_examples, sample):
    """sample: {"id", "question", "text": {...}, "audio": {...}, "video": {...}}

    text/audio/video는 각각 text_features.extract_text_stats_for_prompt(),
    audio_features.audio_vector_to_nested_dict(), dict(zip(VIDEO_FEATURE_NAMES, vec))
    로 만들어진 dict를 그대로 넣으면 된다.
    """
    calibration_block = build_calibration_block(calibration_examples)
    text = sample["text"]
    audio = sample["audio"]
    video = sample["video"]

    labels_schema = ", ".join(f'"{name}": <float>' for name in INTERVIEW_LABEL_NAMES)

    return f"""You are an expert interview coach and multimodal behavior analyst.

TASK:
You will see labeled CALIBRATION EXAMPLES below. Each example has:
- labels: 6 interview performance scores (z-scores, mean=0, std~1, roughly -10..+10;
  positive = above average, negative = below average)
- transcript + text_stats (filler words, vocabulary diversity, repeated words)
- audio_stats (pitch, energy/rms, silence ratio, speech rate)
- video_stats (facial action unit proxies for smiling/blinking/brow-furrow, gaze
  stability, head-pose stability)

Use these examples to calibrate what feature patterns correspond to which score
ranges (e.g. high AU06/AU12 + low AU45/AU04 + low gaze_std/pose_std + few filler
words => high scores; the opposite => low scores; strong content but poor
non-verbal delivery, or vice versa, can pull sub-scores apart).

CALIBRATION EXAMPLES:
{calibration_block}

Now analyze this NEW candidate. You are given their actual interview video
(watch the visuals and listen to the audio directly — this is the primary
evidence) together with measured stats below (precise numbers you cannot read
off the video by eye, e.g. exact filler-word count or pitch in Hz). Use the
video for your holistic impression (facial expression, posture, tone, pacing)
and the stats to double-check/quantify that impression. Predict the same 6
scores, AND give the candidate concrete, actionable interview coaching
feedback based on what you actually saw/heard in the video (not generic
advice) — specific strengths to keep and specific things to improve next time.

Question asked: "{sample.get('question', '')}"

transcript: "{text.get('transcript', '')}"
text_stats: filler_words={text.get('filler_words')}, filler_count={text.get('filler_count')}, \
repeated_words={text.get('repeated_words')}, answer_length={text.get('answer_length')}, \
vocab_diversity={text.get('vocab_diversity')}

audio_stats: pitch_mean={audio.get('pitch_mean')}, pitch_std={audio.get('pitch_std')}, \
rms_mean={audio.get('rms_mean')}, rms_std={audio.get('rms_std')}, \
silence_ratio={audio.get('silence_ratio')}, speech_rate={audio.get('speech_rate')}

video_stats: AU06_mean={video.get('AU06_mean')}, AU06_std={video.get('AU06_std')}, \
AU12_mean={video.get('AU12_mean')}, AU12_std={video.get('AU12_std')}, \
AU45_mean={video.get('AU45_mean')}, AU45_std={video.get('AU45_std')}, \
AU04_mean={video.get('AU04_mean')}, AU04_std={video.get('AU04_std')}, \
gaze_x_mean={video.get('gaze_x_mean')}, gaze_y_mean={video.get('gaze_y_mean')}, \
gaze_std={video.get('gaze_std')}, pose_rx_mean={video.get('pose_rx_mean')}, \
pose_ry_mean={video.get('pose_ry_mean')}, pose_rz_mean={video.get('pose_rz_mean')}, \
pose_std={video.get('pose_std')}

Note: video_stats here come from a MediaPipe landmark-based proxy, not real FACS AU
intensities, so treat exact magnitudes loosely and focus on the relative pattern
(smile-like vs. tense/blinking-heavy, stable vs. jittery gaze/pose).

STRICT OUTPUT (return ONLY this raw JSON object, no markdown, no extra text):
{{
  "category": "<one short Korean label like 우수/평균/미흡/내용↑ 비언어↓/비언어↑ 내용↓, or your own>",
  "labels": {{{labels_schema}}},
  "reasoning": "<2-3 sentences explaining the scores, citing specific evidence you directly observed in the video/audio plus the measured stats>",
  "feedback": {{
    "strengths": ["<specific thing the candidate did well, citing what you saw/heard>", "..."],
    "improvements": ["<specific, actionable suggestion for next time, tied to a concrete observation>", "..."],
    "summary": "<1-2 sentence overall coaching takeaway in Korean, written directly to the candidate>"
  }}
}}
"""
