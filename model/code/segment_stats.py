"""
답변을 문장 구간(Whisper segments)으로 나누고, 구간마다 전달 관련 측정값과 규칙 마커를 붙인다 (구간별 피드백용).

마커는 고정 기준이다. 연출 녹화(일부러 옆 보기, 침묵 넣기 등)로 잡히는지 확인하고 기준값을 조정한다.
  gaze_away     시선이 정면에서 크게 벗어난 상태가 GAZE_AWAY_MIN_SEC 이상 연속
  face_missing  얼굴이 FACE_MISSING_MIN_SEC 이상 연속으로 안 잡힘
  long_pause    침묵이 LONG_PAUSE_SEC 이상 (이 구간 직전의 문장 사이 공백 포함)
  pace_slow/pace_fast  구간 말 속도가 WPM_SLOW 미만 / WPM_FAST 초과 (PACE_MIN_SEC 이상 구간만)
  fillers       군더더기 말(um, uh 등)이 FILLER_MIN회 이상
"""

import re

import numpy as np

from config import FILLER_WORDS
from video_stats import GAZE_AWAY_THRESHOLD

MIN_SEGMENT_SEC = 3.0
GAZE_AWAY_MIN_SEC = 1.0
FACE_MISSING_MIN_SEC = 1.0
LONG_PAUSE_SEC = 2.0
WPM_SLOW, WPM_FAST, PACE_MIN_SEC = 100, 180, 3.0
FILLER_MIN = 2

_WORD_RE = re.compile(r"[A-Za-z가-힣']+")
_FILLER_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b", re.IGNORECASE)


def merge_whisper_segments(raw_segments, min_sec=MIN_SEGMENT_SEC):
    """Whisper segments -> [{"id": "S1", "start", "end", "text"}]. min_sec보다 짧은 문장은 앞 문장에 붙인다(첫 문장은 뒤에)."""
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
            for s in raw_segments if s["text"].strip()]
    merged = []
    for s in segs:
        if merged and s["end"] - s["start"] < min_sec:
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] += " " + s["text"]
        else:
            merged.append(dict(s))
    if len(merged) > 1 and merged[0]["end"] - merged[0]["start"] < min_sec:
        first = merged.pop(0)
        merged[0]["start"] = first["start"]
        merged[0]["text"] = first["text"] + " " + merged[0]["text"]
    return [{"id": f"S{k}", "start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
            for k, s in enumerate(merged, 1)]


def _runs(times, mask, step):
    """mask가 연속으로 True인 샘플 묶음 -> [(시작, 길이)]. 길이는 마지막 샘플 뒤 한 간격까지 포함한다."""
    runs, start, last = [], None, None
    for t, m in zip(times, mask):
        if m:
            start = t if start is None else start
            last = t
        elif start is not None:
            runs.append((start, last - start + step))
            start = None
    if start is not None:
        runs.append((start, last - start + step))
    return runs


def _video_part(timeline, start, end):
    idx = [k for k, t in enumerate(timeline["times"]) if start <= t < end]
    if not idx:
        return {}, []
    times = [timeline["times"][k] for k in idx]
    feats = [timeline["feats"][k] for k in idx]
    face = [f is not None for f in feats]
    off = [f is not None and float(np.hypot(f["gx"], f["gy"])) > GAZE_AWAY_THRESHOLD for f in feats]
    step = timeline["step"]
    stats = {"face_detected_ratio": round(float(np.mean(face)), 3)}
    if any(face):
        stats["gaze_away_ratio"] = round(float(np.sum(off) / np.sum(face)), 3)
        nose = np.stack([f["nose_xy"] for f in feats if f is not None])
        stats["head_motion"] = round(float(np.linalg.norm(np.diff(nose, axis=0), axis=1).mean()), 4) if len(nose) > 1 else 0.0
    markers = []
    for s, dur in _runs(times, off, step):
        if dur >= GAZE_AWAY_MIN_SEC:
            markers.append({"type": "gaze_away", "start": round(s, 2), "dur": round(dur, 1),
                            "detail": f"시선이 {dur:.1f}초간 정면에서 벗어남"})
    for s, dur in _runs(times, [not f for f in face], step):
        if dur >= FACE_MISSING_MIN_SEC:
            markers.append({"type": "face_missing", "start": round(s, 2), "dur": round(dur, 1),
                            "detail": f"얼굴이 {dur:.1f}초간 화면에서 잡히지 않음"})
    return stats, markers


def _audio_part(audio, window_start, start, end):
    """침묵은 window_start(직전 문장 끝)부터 보고, 음량과 침묵 비율은 이 문장 구간 안에서 본다."""
    stats, markers = {}, []
    inside = [(max(a, start), min(b, end)) for a, b in audio["silences"] if b > start and a < end]
    stats["silence_ratio"] = round(sum(b - a for a, b in inside) / max(end - start, 1e-6), 3)
    m = (audio["rms_times"] >= start) & (audio["rms_times"] < end)
    if m.any():
        stats["rms_mean"] = round(float(audio["rms"][m].mean()), 4)
    pauses = [(max(a, window_start), min(b, end)) for a, b in audio["silences"] if b > window_start and a < end]
    longest = max(pauses, key=lambda p: p[1] - p[0], default=None)
    if longest:
        stats["longest_pause_sec"] = round(longest[1] - longest[0], 2)
        if longest[1] - longest[0] >= LONG_PAUSE_SEC:
            markers.append({"type": "long_pause", "start": round(longest[0], 2), "dur": round(longest[1] - longest[0], 1),
                            "detail": f"{longest[1] - longest[0]:.1f}초간 말이 멈춤"})
    return stats, markers


def _text_part(seg):
    dur = seg["end"] - seg["start"]
    n_words = len(_WORD_RE.findall(seg["text"]))
    fillers = len(_FILLER_RE.findall(seg["text"]))
    stats = {"words": n_words, "fillers": fillers}
    markers = []
    if dur >= PACE_MIN_SEC:
        wpm = n_words / dur * 60
        stats["wpm"] = round(wpm, 1)
        if wpm < WPM_SLOW or wpm > WPM_FAST:
            kind, word = ("pace_slow", "느림") if wpm < WPM_SLOW else ("pace_fast", "빠름")
            markers.append({"type": kind, "start": seg["start"], "dur": round(dur, 1),
                            "detail": f"말 속도가 분당 {wpm:.0f}단어로 {word}"})
    if fillers >= FILLER_MIN:
        markers.append({"type": "fillers", "start": seg["start"], "dur": round(dur, 1),
                        "detail": f"군더더기 말 {fillers}회"})
    return stats, markers


def build_segment_features(segments, video_tl=None, audio_tl=None):
    """구간마다 측정값(stats), 규칙 마커(markers), rule_flag(issue: 마커 있음 / stable: 측정 가능했고 마커 없음 / unknown)."""
    out = []
    for i, seg in enumerate(segments):
        stats, markers = {}, []
        st, mk = _text_part(seg)
        stats.update(st)
        markers += mk
        if video_tl is not None:
            st, mk = _video_part(video_tl, seg["start"], seg["end"])
            stats.update(st)
            markers += mk
        if audio_tl is not None:
            st, mk = _audio_part(audio_tl, segments[i - 1]["end"] if i else 0.0, seg["start"], seg["end"])
            stats.update(st)
            markers += mk
        measured = video_tl is not None and audio_tl is not None
        out.append({**seg, "stats": stats, "markers": sorted(markers, key=lambda m: m["start"]),
                    "rule_flag": "issue" if markers else ("stable" if measured else "unknown")})
    return out
