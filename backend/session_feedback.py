"""Validate session coaching and resolve model-owned evidence to local recordings."""
import copy
import math


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def enum(*values):
    return {"type": ["string", "null"] if None in values else "string", "enum": list(values)}


def array(items, **limits):
    return {"type": "array", "items": items, **limits}


STRING = {"type": "string"}
NULLABLE_STRING = {"type": ["string", "null"]}
BAND = enum("상", "중", "하", None)
REF = obj({"question_id": STRING, "segment_id": STRING})
ITEM = obj({"modality": enum("audio", "video", "multimodal"), "feedback": STRING,
            "suggestion": NULLABLE_STRING, "scope": enum("single_answer", "multiple_answers"),
            "related_question_ids": array(STRING), "evidence": array(REF, maxItems=2)})
ANSWER = obj({"question_id": STRING, "status": enum("evaluated", "insufficient_data"),
              "content_band": BAND, "coaching_feedback": STRING,
              "relevance": obj({"rating": enum("적절", "부분적", "부적절", "평가불가"), "reason": STRING}),
              "specificity": STRING,
              "accuracy": obj({"status": enum("supported", "contradicted", "unverified"), "reason": STRING}),
              "strength": NULLABLE_STRING, "improvement": NULLABLE_STRING})
SESSION_SCHEMA = obj({"status": enum("ready", "partial", "insufficient_data"), "overall_band": BAND,
                      "band_reason": STRING, "summary": STRING,
                      "answer_feedback": array(ANSWER, minItems=5, maxItems=5),
                      "delivery_feedback": obj({"summary": STRING,
                          "strengths": array(ITEM, maxItems=2), "improvements": array(ITEM, maxItems=2)}),
                      "limitations": array(STRING)})


def validate_shape(value, schema):
    kind = schema["type"]
    if value is None:
        if isinstance(kind, list) and "null" in kind:
            return None
        raise ValueError("필수 평가 값이 없습니다.")
    if isinstance(kind, list):
        kind = next(t for t in kind if t != "null")
    if kind == "object":
        if not isinstance(value, dict) or any(k not in value for k in schema["required"]):
            raise ValueError("세션 피드백 필드가 누락됐습니다.")
        return {k: validate_shape(value[k], s) for k, s in schema["properties"].items()}
    if kind == "array":
        if not isinstance(value, list) or not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 1000):
            raise ValueError("피드백 목록 개수가 올바르지 않습니다.")
        return [validate_shape(x, schema["items"]) for x in value]
    if not isinstance(value, str) or ("enum" in schema and value not in schema["enum"]):
        raise ValueError("피드백 값의 형식이 올바르지 않습니다.")
    return value


def validate_session_feedback(value):
    report = validate_shape(value, SESSION_SCHEMA)
    expected = {f"Q{i}" for i in range(1, 6)}
    answers = report["answer_feedback"]
    if {a["question_id"] for a in answers} != expected:
        raise ValueError("질문 5개의 피드백이 중복되거나 누락됐습니다.")
    if report["status"] == "insufficient_data" and report["overall_band"] is not None:
        raise ValueError("자료 부족 상태에는 세션 등급을 줄 수 없습니다.")
    if report["status"] == "ready" and (report["overall_band"] is None or any(a["status"] != "evaluated" for a in answers)):
        raise ValueError("완료 상태와 평가 자료가 일치하지 않습니다.")
    if not report["band_reason"].strip() or not report["summary"].strip():
        raise ValueError("전체 평가 이유가 비어 있습니다.")
    for answer in answers:
        if not answer["coaching_feedback"].strip():
            raise ValueError("답변 코칭이 비어 있습니다.")
        if answer["status"] == "evaluated" and answer["content_band"] is None:
            raise ValueError("평가된 답변의 등급이 없습니다.")
        if answer["status"] == "insufficient_data" and (answer["content_band"] is not None
                or answer["relevance"]["rating"] != "평가불가"
                or answer["strength"] is not None or answer["improvement"] is not None):
            raise ValueError("자료 부족 답변의 평가가 일치하지 않습니다.")
        # No verification source is supplied by the current service.
        if answer["accuracy"]["status"] != "unverified":
            raise ValueError("확인 자료 없이 내용의 사실 여부를 단정할 수 없습니다.")
    for kind in ("strengths", "improvements"):
        for item in report["delivery_feedback"][kind]:
            related = item["related_question_ids"]
            if not related or len(set(related)) != len(related) or not set(related) <= expected:
                raise ValueError("종합 피드백의 관련 질문이 올바르지 않습니다.")
            if (len(related) == 1) != (item["scope"] == "single_answer") or not item["feedback"].strip():
                raise ValueError("피드백의 관찰 범위가 올바르지 않습니다.")
            if kind == "improvements" and (not item["suggestion"] or not item["suggestion"].strip()):
                raise ValueError("개선 피드백에는 구체적인 조언이 필요합니다.")
    return report


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_segments(segments):
    if not isinstance(segments, list):
        raise ValueError("문장 구간 목록이 필요합니다.")
    seen = set()
    for segment in segments:
        if (not isinstance(segment, dict) or not isinstance(segment.get("id"), str)
                or not segment["id"].strip() or segment["id"] in seen
                or not isinstance(segment.get("text"), str)
                or not finite(segment.get("start")) or not finite(segment.get("end"))
                or not 0 <= segment["start"] < segment["end"]):
            raise ValueError("문장 구간의 ID·시간·텍스트가 올바르지 않습니다.")
        seen.add(segment["id"])
        if "markers" in segment and not isinstance(segment["markers"], list):
            raise ValueError("구간 마커 목록이 올바르지 않습니다.")
        if "rule_flag" in segment and segment["rule_flag"] not in ("issue", "stable", "unknown"):
            raise ValueError("구간 관찰 상태가 올바르지 않습니다.")
    return segments


def normalize_feedback(raw, result, answers):
    """Map Q1..Q5 to trusted local question IDs; never trust generated times or URLs."""
    report = copy.deepcopy(validate_session_feedback(raw))
    ids = {f"Q{i}": a["question_id"] for i, a in enumerate(answers, 1)}
    lookup = {}
    for i, video in enumerate(result["videos"], 1):
        for segment in validate_segments(video.get("segments") or []):
            lookup[(f"Q{i}", segment["id"])] = segment
    dropped, clips = [], {}
    # Resolve improvements first, so conflicting evidence has improvement priority.
    improvement_refs = {(r["question_id"], r["segment_id"])
                        for item in report["delivery_feedback"]["improvements"] for r in item["evidence"]}
    for kind in ("improvements", "strengths"):
        for item in report["delivery_feedback"][kind]:
            valid = []
            for ref in item["evidence"]:
                key = (ref["question_id"], ref["segment_id"])
                segment = lookup.get(key)
                if (segment is None or ref["question_id"] not in item["related_question_ids"]
                        or (kind == "strengths" and (segment.get("rule_flag") == "issue" or key in improvement_refs))):
                    dropped.append({**ref, "kind": kind, "reason": "missing_or_conflicting_evidence"})
                    continue
                marker = (segment.get("markers") or [None])[0]
                start, end = segment["start"], segment["end"]
                if marker:
                    if not isinstance(marker, dict) or not finite(marker.get("start")) or not finite(marker.get("dur")):
                        dropped.append({**ref, "kind": kind, "reason": "invalid_marker"})
                        continue
                    start, end = marker["start"], marker["start"] + marker["dur"]
                duration = answers[int(ref["question_id"][1:]) - 1].get("duration_seconds")
                if (not 0 <= start < end or (finite(duration) and end > duration + .2)):
                    dropped.append({**ref, "kind": kind, "reason": "outside_recording"})
                    continue
                if finite(duration):
                    end = min(end, duration)
                if start >= end:
                    dropped.append({**ref, "kind": kind, "reason": "outside_recording"})
                    continue
                mapped = {"question_id": ids[ref["question_id"]], "segment_id": ref["segment_id"]}
                if mapped not in valid:
                    valid.append(mapped)
                clips[(mapped["question_id"], mapped["segment_id"])] = {
                    **mapped, "start": round(start, 2), "end": round(end, 2),
                    "playback_start": round(max(0, start - 1), 2),
                    "playback_end": round(min(end, max(0, start - 1) + 5), 2),
                    "label": "improvement" if kind == "improvements" else "strength",
                    "video_url": f'/api/interviews/{result["session_id"]}/answers/{mapped["question_id"]}/video',
                }
            item["evidence"] = valid
            item["related_question_ids"] = [ids[q] for q in item["related_question_ids"]]
    by_id = {a["question_id"]: a for a in report["answer_feedback"]}
    report["answer_feedback"] = [dict(by_id[q], question_id=local) for q, local in ids.items()]
    report.update(schema_version="1.0", session_id=result["session_id"], is_dummy=False,
                  evidence_segments=list(clips.values()))
    if dropped:
        report["limitations"].append("일부 근거 구간을 녹화와 연결할 수 없어 제외했어요.")
    return report, dropped
