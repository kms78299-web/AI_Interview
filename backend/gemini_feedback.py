"""Gemini feedback for model-provided prompts. Never log keys or prompt content."""
import json
import os
import re
import urllib.error
import urllib.request
from session_feedback import SESSION_SCHEMA, validate_session_feedback

DEFAULT_MODEL = "gemini-3.1-flash-lite"
TEXT_FIELDS = ("content_reason", "band_reason", "summary")
SECTIONS = ("text", "audio", "video")
STRING = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {
        "content_rating": {"type": "string", "enum": ["적절", "부분적", "부적절"]},
        "overall_band": {"type": "string", "enum": ["상", "중", "하"]},
        **{name: STRING for name in TEXT_FIELDS},
        **{name: {"type": "object", "properties": {"strength": STRING, "improvement": STRING},
                  "required": ["strength", "improvement"]} for name in SECTIONS},
    },
    "required": ["content_rating", "content_reason", "overall_band", "band_reason", *SECTIONS, "summary"],
}


class FeedbackError(Exception):
    pass


def model_name():
    name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", name):
        raise FeedbackError("Gemini 모델 설정을 확인해 주세요.")
    return name


def validate_feedback(value):
    if not isinstance(value, dict) or value.get("content_rating") not in ("적절", "부분적", "부적절") or value.get("overall_band") not in ("상", "중", "하"):
        raise ValueError("피드백 평가 형식이 올바르지 않습니다.")
    for name in TEXT_FIELDS:
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise ValueError("피드백 설명이 누락됐습니다.")
    for name in SECTIONS:
        if not isinstance(value.get(name), dict) or any(not isinstance(value[name].get(k), str) or not value[name][k].strip() for k in ("strength", "improvement")):
            raise ValueError("피드백 강점·개선점이 누락됐습니다.")
    if (value["content_rating"] == "부적절" and value["overall_band"] != "하") or (value["content_rating"] == "부분적" and value["overall_band"] == "상"):
        raise ValueError("내용 평가와 종합 평가가 일치하지 않습니다.")
    # Whitelist fields rather than returning arbitrary generated JSON to the UI.
    return {name: value[name] for name in SCHEMA["required"]}


def generate_feedback(prompt, schema=None, validator=None):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise FeedbackError("Gemini API 키가 설정되지 않았어요.")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "responseFormat": {"text": {"mimeType": "APPLICATION_JSON", "schema": schema or SCHEMA}}},
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name()}:generateContent",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"}, method="POST",
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:
            # Do not expose provider response bodies, URLs, keys, or prompts.
            if error.code == 429:
                message = "Gemini 요청 한도에 도달했어요. 잠시 후 다시 요청해 주세요."
            elif error.code in (500, 502, 503, 504):
                message = "Gemini 서버가 혼잡하거나 일시적으로 응답하지 않아요. 잠시 후 다시 요청해 주세요."
            else:
                message = "Gemini 호출에 실패했어요. API 키·모델 설정과 이용 가능 여부를 확인해 주세요."
            raise FeedbackError(message) from None
        except (OSError, ValueError):
            raise FeedbackError("Gemini에 연결하지 못했어요. 잠시 후 다시 요청해 주세요.") from None
        try:
            candidate = body["candidates"][0]
            if candidate.get("finishReason") != "STOP":
                raise ValueError("Incomplete response")
            parts = candidate["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return (validator or validate_feedback)(json.loads(text))
        except (KeyError, IndexError, TypeError, ValueError):
            if attempt == 1:
                raise FeedbackError("Gemini 피드백 형식을 확인하지 못했어요. 다시 요청해 주세요.") from None


def generate_session_feedback(prompt):
    return generate_feedback(prompt, schema=SESSION_SCHEMA, validator=validate_session_feedback)
