"""Validate run_session_inference results and adapt them without converting scores."""

import copy
import json
import math


SCORE_NAMES = (
    "interview_score",
    "answer_score",
    "speaking_skills",
    "confidence_score",
    "facial_expression",
    "overall_performance",
)


def validate_score(score, location):
    if not isinstance(score, dict):
        raise ValueError(f"{location}: 점수 객체가 필요합니다.")
    for field in ("raw", "percentile", "band"):
        if field not in score:
            raise ValueError(f"{location}.{field}: 값이 없습니다.")
    raw = score["raw"]
    if type(raw) not in (int, float) or not math.isfinite(raw):
        raise ValueError(f"{location}.raw: 유한한 숫자가 필요합니다.")
    if "smoothed_raw" in score and (
        type(score["smoothed_raw"]) not in (int, float)
        or not math.isfinite(score["smoothed_raw"])
    ):
        raise ValueError(f"{location}.smoothed_raw: 유한한 숫자가 필요합니다.")
    percentile = score["percentile"]
    if percentile is not None and (
        type(percentile) not in (int, float)
        or not math.isfinite(percentile)
        or not 0 <= percentile <= 100
    ):
        raise ValueError(f"{location}.percentile: 0~100 또는 null이 필요합니다.")
    band = score["band"]
    if band is not None and (not isinstance(band, str) or not band.strip()):
        raise ValueError(f"{location}.band: 구간 문자열 또는 null이 필요합니다.")


def validate_scores(scores, location):
    if not isinstance(scores, dict) or set(scores) != set(SCORE_NAMES):
        raise ValueError(f"{location}: 6개 평가 항목이 모두 필요합니다.")
    for name in SCORE_NAMES:
        validate_score(scores[name], f"{location}.{name}")


def validate_analysis_result(result):
    """Accept the model's session/videos contract, including null percentiles."""
    if not isinstance(result, dict):
        raise ValueError("run_session_inference의 session, videos, explanation 객체가 필요합니다.")
    if "status" in result and result["status"] != "success":
        raise ValueError("성공하지 않은 모델 응답입니다.")
    if "is_dummy" in result and type(result["is_dummy"]) is not bool:
        raise ValueError("is_dummy: true 또는 false가 필요합니다.")
    session = result.get("session")
    if not isinstance(session, dict) or type(session.get("n_videos")) is not int or session["n_videos"] != 5:
        raise ValueError("session.n_videos: 5가 필요합니다.")
    smoothing = session.get("smoothing")
    if "smoothing" not in session or (
        smoothing is not None and (
            type(smoothing) not in (int, float) or not math.isfinite(smoothing) or not 0 <= smoothing <= 1
        )
    ):
        raise ValueError("session.smoothing: 0~1 또는 null이 필요합니다.")
    validate_scores(session.get("scores"), "session.scores")
    videos = result.get("videos")
    if not isinstance(videos, list) or len(videos) != 5:
        raise ValueError("videos: 질문 순서의 영상별 결과 5개가 필요합니다.")
    for index, video in enumerate(videos):
        location = f"videos[{index}]"
        if not isinstance(video, dict) or not isinstance(video.get("video"), str) or not video["video"].strip():
            raise ValueError(f"{location}.video: 영상 파일 경로가 필요합니다.")
        validate_scores(video.get("scores"), f"{location}.scores")
        if not isinstance(video.get("transcript"), str):
            raise ValueError(f"{location}.transcript: 문자열이 필요합니다.")
        if video.get("stats") is not None and not isinstance(video["stats"], dict):
            raise ValueError(f"{location}.stats: 통계 객체 또는 null이 필요합니다.")
        if video.get("llm_prompt") is not None and not isinstance(video["llm_prompt"], str):
            raise ValueError(f"{location}.llm_prompt: 문자열 또는 null이 필요합니다.")
        if "single_clip_percentile" in video:
            percentiles = video["single_clip_percentile"]
            if not isinstance(percentiles, dict) or set(percentiles) != set(SCORE_NAMES):
                raise ValueError(f"{location}.single_clip_percentile: 6개 항목이 필요합니다.")
            for value in percentiles.values():
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100):
                    raise ValueError(f"{location}.single_clip_percentile: 0~100 또는 null이 필요합니다.")
    if not isinstance(result.get("explanation"), str):
        raise ValueError("explanation: 문자열이 필요합니다.")
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("분석 결과는 NaN/Infinity 없는 JSON이어야 합니다.") from error


def prepare_analysis_result(model_result, video_paths):
    """Preserve native scores and expose compatibility aliases for the web API."""
    validate_analysis_result(model_result)
    expected = [path.name for path in video_paths]
    # The API guarantees input order; remote video paths are temporary identifiers.
    if len(expected) != 5:
        raise ValueError("제출한 영상 5개가 필요합니다.")
    result = copy.deepcopy(model_result)
    answers = [
        {**copy.deepcopy(video), "question_index": index, "filename": filename}
        for index, (video, filename) in enumerate(zip(result["videos"], expected), start=1)
    ]
    result.update({
        "status": "success",
        "is_dummy": result.get("is_dummy", False),
        "video_count": len(answers),
        # These are aliases of the model's session scores, never recalculated means.
        "average_scores": copy.deepcopy(result["session"]["scores"]),
        "overall_score": copy.deepcopy(result["session"]["scores"]["overall_performance"]),
        "answers": answers,
        "combined_transcript": "\n\n".join(
            f"질문 {answer['question_index']}: {answer['transcript']}" for answer in answers
        ),
    })
    return result


def make_mock_result(video_paths):
    """Fixed fake values; never called as a fallback for a Gradio failure."""
    videos = []
    for index, path in enumerate(video_paths, start=1):
        scores = {
            name: {
                "raw": round((index - 3) * 0.1 + position * 0.02, 4),
                "percentile": 40 + index * 5 + position,
                "band": "중",
            }
            for position, name in enumerate(SCORE_NAMES)
        }
        for score in scores.values():
            score["smoothed_raw"] = score["raw"]
        videos.append({
            "video": str(path),
            "scores": scores,
            "transcript": f"[더미] 질문 {index}의 화면 확인용 답변입니다. 실제 STT 결과가 아닙니다.",
            "single_clip_percentile": {name: score["percentile"] for name, score in scores.items()},
        })
    averages = {
        name: {
            "raw": round(sum(video["scores"][name]["raw"] for video in videos) / 5, 4),
            "percentile": round(sum(video["scores"][name]["percentile"] for video in videos) / 5, 1),
            "band": "중",
        }
        for name in SCORE_NAMES
    }
    return {
        "status": "success",
        "is_dummy": True,
        "session": {"n_videos": len(videos), "smoothing": 0.5, "scores": averages},
        "videos": videos,
        "explanation": "화면 개발용 가짜 결과입니다. 모델 추론과 백분위 계산을 실행하지 않았습니다.",
    }
