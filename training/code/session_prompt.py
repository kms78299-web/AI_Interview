"""
세션(한 사용자의 답변 전체) 단위 LLM 프롬프트와, LLM이 고른 근거 구간을 실제 시간으로 바꾸는 도우미.

출력 형식은 팀 데이터 전달 규약(docs/feedback-data-contract.md 7장, 14장)을 따른다. 질문별로는 내용 등급(content_band)과 코칭 문단,
음성·영상은 답변 전체를 합친 좋은 점/개선점과 각각의 대표 장면(evidence)을 쓴다.
LLM은 시간을 직접 쓰지 않고 (question_id, segment_id)만 고른다. 시간은 resolve_evidence가 서버가 만든 구간과 마커에서 붙인다.
question_id는 영상 순서대로 "Q1", "Q2"… 이며, 백엔드가 순서로 실제 질문 ID와 연결한다.
"""

import json

from extract_stats import NO_PERCENTILE, STAT_LABELS, _clock, to_percentile

SESSION_OUTPUT_EXAMPLE = {
    "status": "ready | partial | insufficient_data 중 하나",
    "overall_band": "상 | 중 | 하 | null",
    "band_reason": "세션 전체 등급의 이유",
    "summary": "전체 요약 2~3문장",
    "answer_feedback": [
        {"question_id": "Q1", "status": "evaluated | insufficient_data",
         "content_band": "상 | 중 | 하 | null",
         "coaching_feedback": "좋았던 점을 짚고, 답변을 어떻게 재구성하거나 구체화할지 이어 쓰는 한 문단",
         "relevance": {"rating": "적절 | 부분적 | 부적절 | 평가불가", "reason": "..."},
         "specificity": "구체성 설명",
         "accuracy": {"status": "supported | contradicted | unverified", "reason": "..."},
         "strength": "내용 강점 한 줄 또는 null", "improvement": "내용 개선점 한 줄 또는 null"}
    ],
    "delivery_feedback": {
        "summary": "음성·영상 종합 평가",
        "strengths": [{"modality": "audio | video", "feedback": "...", "suggestion": None,
                       "scope": "single_answer | multiple_answers", "related_question_ids": ["Q1"],
                       "evidence": [{"question_id": "Q1", "segment_id": "S2"}]}],
        "improvements": [{"modality": "audio | video", "feedback": "...", "suggestion": "구체적인 개선 방법",
                          "scope": "single_answer | multiple_answers", "related_question_ids": ["Q2", "Q4"],
                          "evidence": [{"question_id": "Q2", "segment_id": "S3"}]}]
    },
    "limitations": ["자료 누락, 측정 한계 등"],
}


def _stat_lines(stats, reference):
    lines = []
    for group in ("audio", "text", "video"):
        for key, label in STAT_LABELS[group].items():
            if key not in (stats or {}).get(group, {}):
                continue
            v = stats[group][key]
            pct = to_percentile(reference, group, key, v) if reference and key not in NO_PERCENTILE else None
            lines.append(f"  - {label}: {round(v, 3) if isinstance(v, float) else v}"
                         + (f" ({pct}번째 백분위)" if pct is not None else ""))
    return lines


def _answer_block(k, video, question, model_info="full"):
    qid = f"Q{k + 1}"
    lines = [f"### {qid}" + (f" — 질문: {question}" if question else "")]
    if video.get("skipped"):
        lines.append(f"(상태: 답변이 너무 짧아 모델 점수에서 제외됨 — {video.get('warning', '')})" if model_info != "none"
                     else f"(상태: 답변이 매우 짧음 — {video.get('warning', '')})")
    if video.get("stats_error"):
        lines.append(f"(측정 일부 실패: {video['stats_error']} — 실패한 영역은 평가하지 않는다고 밝힐 것)")
    segments = video.get("segments") or []
    lines.append("[대본]" + (" (구간 ID와 시간)" if segments else " (구간 정보 없음: 이 답변은 근거 구간을 인용할 수 없음)"))
    if segments:
        lines += [f"[{g['id']} {_clock(g['start'])}–{_clock(g['end'])}] {g['text']}" for g in segments]
    else:
        lines.append(video.get("transcript") or "(대본 없음)")
    return qid, lines, segments


MODEL_INFO_LEVELS = ("full", "output_only", "none")


def build_session_prompt(result, questions=None, reference=None, model_info="full"):
    """analyze_session 결과 -> 세션 단위 프롬프트 문자열 (답변 전체를 한 번에 평가).

    model_info: 모델 정보를 얼마나 넣을지 (LLM 평가 방식 비교 실험용).
      "full"        모델 출력 + 학습 정보 + 성능·편향 정보와 사용 제한 (서비스 기본, session.llm_prompt)
      "output_only" 모델 출력 + 학습 정보만 (성능, 편향, 사용 제한 없음)
      "none"        모델 정보 없음 (측정값과 대본만)"""
    if model_info not in MODEL_INFO_LEVELS:
        raise ValueError(f"model_info는 {MODEL_INFO_LEVELS} 중 하나")
    videos = result["videos"]
    blocks, marker_lines, ids = [], [], []
    for k, video in enumerate(videos):
        qid, lines, segments = _answer_block(k, video, None if questions is None else questions[k], model_info)
        ids.append(qid)
        stat_lines = _stat_lines(video.get("stats"), reference)
        blocks += lines + (["[측정값] (참가자 대비 백분위, 100에 가까울수록 값이 큰 편)"] + stat_lines if stat_lines else
                           ["[측정값] 없음 (음성·영상은 평가하지 않음)"]) + [""]
        for g in segments:
            if g.get("markers"):
                marker_lines.append(f"- {qid} {g['id']}: " + "; ".join(m["detail"] for m in g["markers"]))

    delivery = (result.get("session") or {}).get("delivery")
    model_lines = []
    has_model = bool(delivery and delivery.get("scores", {}).get("overall_performance"))
    if has_model and model_info in ("output_only", "full"):  # full은 output_only에 성능·편향 정보와 사용 제한을 더한 것
        model_lines = [
            "[참고: 예측 모델의 사람 단위 전반적 인상]",
            f"- {delivery['scores']['overall_performance']['band']} (RecruitView 평가자 판단을 학습한 모델이 이 사람의 답변 전체를 다른 지원자와 비교한 위치)"]
    if has_model and model_info == "full":
        model_lines += [
            "- 이 모델은 사람 전체 단위로만 평가자 판단과 맞는다(Spearman 약 0.73). 같은 사람의 답변끼리는 거의 구분하지 못하므로 "
            "답변 비교, 장면 선택, 세션 등급 결정에 쓰지 않는다. 내용은 보지 못하고, 간결하게 말하는 사람을 높게 보는 경향이 있다."]
    if model_lines:
        model_lines.append("")

    rules = [
        "1. 대본과 측정값, 구간별 측정 관찰에 있는 것만 근거로 쓴다. 감정, 성격, 합격 가능성은 추측하지 않는다. 숫자 점수나 백분위를 새로 만들지 않는다.",
        "2. answer_feedback는 위 답변 순서대로 정확히 " + f"{len(ids)}개({', '.join(ids)})를 쓴다. "
        "content_band(상/중/하)는 그 답변의 텍스트 내용만 본다: 상=질문에 직접 답하고 구체적 근거와 구성이 충분함 / "
        "중=대체로 답하지만 근거나 설명 보완이 필요함 / 하=질문 핵심에 충분히 답하지 못하거나 핵심 설명이 크게 부족함. "
        + ("음성·영상이나 예측 모델 인상은 반영하지 않는다. " if model_info == "full" else "음성·영상은 반영하지 않는다. ") +
        "coaching_feedback는 좋았던 점을 짚고 답변을 어떻게 재구성하거나 구체화할지 "
        "자연스럽게 이어 쓴 한 문단이다(제목을 붙여 나열하지 않는다). relevance는 질문에 직접 답했는지(적절/부분적/부적절, 대본이 비면 평가불가), "
        "specificity는 구체적 사례나 근거가 있는지, accuracy는 대본만으로 사실 확인이 안 되면 unverified로 둔다. "
        "대본이 비거나 평가할 수 없으면 status=insufficient_data, content_band=null, strength/improvement는 null.",
        "3. 음성·영상은 답변 전체를 함께 검토해 delivery_feedback 하나로 쓴다. 좋은 점(strengths)과 개선점(improvements)은 각각 0~2개. "
        "scope는 관련 답변이 1개면 single_answer, 2개 이상이면 multiple_answers이고, 여러 답변의 경향이라고 쓸 때는 그 답변들에 실제 관찰이 있어야 한다. "
        "개선점에는 suggestion(구체적인 개선 방법)을 쓴다.",
        "4. overall_band(세션 전체 상/중/하): 상=대부분의 답변이 질문에 직접 답하고 구체적이며 전달에 큰 방해가 적음 / "
        "중=대체로 답하지만 구체성이나 전달에 보완할 점이 있음 / 하=여러 답변에서 질문에 충분히 답하지 못했거나 전달에 큰 보완이 필요함. "
        "자료가 부족하면 null이고 status=insufficient_data. 짧은 답변, 한 번의 시선 이탈, 목소리 특성 하나만으로 하를 주지 않는다. "
        + ("예측 모델의 인상을 등급으로 옮기지 않는다." if model_info == "full" else ""),
        "5. delivery_feedback의 각 항목에는 그 특징이 가장 잘 드러나는 대표 장면을 evidence 첫 번째로 1개 넣는다(최대 2개). "
        "위 대본에 있는 (question_id, segment_id)만 쓰고 시간은 쓰지 않는다. 개선점 장면은 구간별 측정 관찰에 걸린 구간을 우선 고르고, "
        "좋은 점 장면은 측정 관찰에 걸리지 않은 구간에서 측정값이 뒷받침할 때만 고른다. 근거 장면이 없으면 빈 배열로 둔다.",
        "6. 구간 정보가 없는 답변은 장면을 인용할 수 없다. 장면 없이 피드백만 쓴다.",
        "7. 참고: RecruitView 평가자 데이터에서는 말 속도가 빠를수록, 음높이가 높을수록 낮게 평가되는 경향이 있었고, "
        "시선 이탈 비율은 평가와 거의 관계가 없었다. 시선 관련 내용은 면접 태도에 대한 일반적인 조언으로만 쓴다.",
        "8. 한국어, 친근하지만 단정적이지 않은 말투. JSON 객체 하나만 출력한다(설명 문장, 코드블록 표시 없이).",
    ]
    return "\n".join([
        f"당신은 영어 면접 연습 피드백을 쓰는 코치입니다. 아래는 한 지원자가 질문 {len(ids)}개에 답한 영상의 대본과 자동 측정값입니다. "
        "답변 전체를 함께 보고 평가하세요.",
        "",
        *blocks,
        "[구간별 측정 관찰] (자동 측정 기준에 걸린 구간만 표시)",
        *(marker_lines or ["- 기준에 걸린 구간 없음"]), "",
        *model_lines,
        "[작성 규칙]", *rules, "",
        "[출력 형식]",
        json.dumps(SESSION_OUTPUT_EXAMPLE, ensure_ascii=False, indent=2),
    ])


def resolve_evidence(feedback, result, lead_sec=1.0):
    """LLM 응답의 delivery_feedback 대표 장면(evidence)을 실제 시간으로 바꾼다. 없는 ID는 버리고 dropped에 남긴다.

    시간: 그 구간에 마커가 있으면 첫 마커 시간(start ~ start + dur), 없으면 문장 구간 전체(start ~ end). play_start = max(0, start - lead_sec).
    좋은 점(strengths)의 장면 중 마커가 걸린 구간(rule_flag == "issue")은 쓰지 않는다. 항목마다 첫 번째 유효 장면이 대표 장면이다."""
    lookup = {}
    for k, video in enumerate(result["videos"]):
        for g in video.get("segments") or []:
            lookup[(f"Q{k + 1}", g["id"])] = g

    out = {"strengths": [], "improvements": [], "dropped": []}
    for kind in ("strengths", "improvements"):
        for n, item in enumerate((feedback.get("delivery_feedback") or {}).get(kind, [])):
            scenes = []
            for ref in item.get("evidence", []):
                g = lookup.get((ref.get("question_id"), ref.get("segment_id")))
                if g is None or (kind == "strengths" and g.get("rule_flag") == "issue"):
                    out["dropped"].append({**ref, "item": f"{kind}[{n}]"})
                    continue
                m = (g.get("markers") or [None])[0]
                start, end = (m["start"], m["start"] + m["dur"]) if m else (g["start"], g["end"])
                scenes.append({**ref, "video_index": int(ref["question_id"][1:]) - 1, "start": round(start, 2),
                               "end": round(end, 2), "play_start": round(max(0.0, start - lead_sec), 2),
                               "segment_text": g["text"], "marker": m["detail"] if m else None})
            out[kind].append({"item_index": n, "modality": item.get("modality"),
                              "representative": scenes[0] if scenes else None, "scenes": scenes})
    return out
