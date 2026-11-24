import copy
import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import main
import test_analysis
from analysis_results import make_mock_result
from session_feedback import validate_session_feedback, normalize_feedback, validate_segments


def sample_session_feedback():
    return {"status": "ready", "overall_band": "중", "band_reason": "구체적인 설명을 더할 수 있어요.",
        "summary": "핵심 경험을 구체화해 보세요.",
        "answer_feedback": [{"question_id": f"Q{i}", "status": "evaluated", "content_band": "중",
            "coaching_feedback": "질문에 답했어요. 본인의 행동과 결과를 덧붙여 보세요.",
            "relevance": {"rating": "적절", "reason": "질문에 답했어요."}, "specificity": "구체화할 수 있어요.",
            "accuracy": {"status": "unverified", "reason": "확인 자료가 없어요."},
            "strength": "질문에 답했어요.", "improvement": "경험을 덧붙여 보세요."} for i in range(1, 6)],
        "delivery_feedback": {"summary": "전달 관찰을 살펴봤어요.", "strengths": [], "improvements": [
            {"modality": "audio", "feedback": "긴 쉼이 관찰됐어요.", "suggestion": "핵심 단어를 정리해 보세요.",
             "scope": "single_answer", "related_question_ids": ["Q1"],
             "evidence": [{"question_id": "Q1", "segment_id": "S1"}]}]}, "limitations": []}


class SessionValidationTests(unittest.TestCase):
    def test_rejects_duplicate_questions_scope_and_unsupported_accuracy(self):
        for mutate in (lambda r: r["answer_feedback"][1].update(question_id="Q1"),
                       lambda r: r["answer_feedback"][0]["accuracy"].update(status="supported"),
                       lambda r: r["delivery_feedback"]["improvements"][0].update(scope="multiple_answers"),
                       lambda r: r.update(status="insufficient_data")):
            raw = sample_session_feedback(); mutate(raw)
            with self.assertRaises(ValueError):
                validate_session_feedback(raw)

    def test_marker_resolution_id_mapping_and_invalid_evidence_filter(self):
        raw = sample_session_feedback()
        raw["delivery_feedback"]["improvements"][0]["evidence"].insert(0, {"question_id": "Q1", "segment_id": "missing"})
        result = {"session_id": "a"*32, "videos": [{"segments": [{"id": "S1", "start": 10, "end": 17,
            "text": "hello", "rule_flag": "issue", "markers": [{"start": 12.1, "dur": 2.9}]}]}] + [{}]*4}
        answers = [{"question_id": f"local{i}", "duration_seconds": 60} for i in range(1, 6)]
        report, dropped = normalize_feedback(raw, result, answers)
        clip = report["evidence_segments"][0]
        self.assertEqual((clip["question_id"], clip["start"], clip["end"], clip["playback_start"]), ("local1", 12.1, 15.0, 11.1))
        self.assertEqual(clip["playback_end"], 15.0)
        self.assertEqual(report["answer_feedback"][0]["question_id"], "local1")
        self.assertEqual(len(dropped), 1)
        self.assertNotIn("missing", json.dumps(report["evidence_segments"]))

    def test_issue_or_improvement_conflicts_are_never_strength_clips(self):
        raw = sample_session_feedback()
        strength = copy.deepcopy(raw["delivery_feedback"]["improvements"][0]); strength["suggestion"] = None
        raw["delivery_feedback"]["strengths"] = [strength]
        result = {"session_id": "a"*32, "videos": [{"segments": [{"id": "S1", "start": 0, "end": 3,
            "text": "hello", "rule_flag": "stable", "markers": []}]}] + [{}]*4}
        report, _ = normalize_feedback(raw, result, [{"question_id": f"q{i}"} for i in range(1, 6)])
        self.assertEqual(report["delivery_feedback"]["strengths"][0]["evidence"], [])
        self.assertEqual(report["evidence_segments"][0]["playback_start"], 0)

    def test_bad_segment_times_and_duplicate_ids_are_rejected(self):
        for segments in ([{"id": "S1", "start": float("nan"), "end": 3, "text": "hello"}],
                         [{"id": "S1", "start": 0, "end": 3, "text": "hello"}]*2):
            with self.assertRaises(ValueError):
                validate_segments(segments)


class SessionEndpointTests(unittest.TestCase):
    setUp = test_analysis.AnalysisTests.setUp

    def test_preview_is_explicit_read_only_and_rejects_changed_recordings(self):
        payload = {"video_signature": main.video_signature(self.paths), "label": "test"}
        main.write_json(self.session_dir/"feedback-preview.json", payload)
        with patch.dict(main.os.environ, {"INTERVIEW_ENABLE_PREVIEW": "0"}):
            with self.assertRaises(HTTPException):
                main.get_feedback_preview(self.session_id)
        with patch.dict(main.os.environ, {"INTERVIEW_ENABLE_PREVIEW": "1"}):
            response = main.get_feedback_preview(self.session_id)
            self.assertEqual(json.loads(response.body), payload)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.paths[0].write_bytes(b"changed")
            with self.assertRaises(HTTPException) as error:
                main.get_feedback_preview(self.session_id)
        self.assertEqual(error.exception.status_code, 409)
        self.assertFalse((self.session_dir/"result.json").exists())

    def analyze_fixture(self):
        response = make_mock_result(self.paths)
        response["is_dummy"] = False
        response["session"]["llm_prompt"] = "exact session prompt\n  unchanged"
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "gradio"}), patch.object(main, "call_gradio", return_value=response):
            return main.analyze_interview(self.session_id)

    def test_single_session_call_and_cache(self):
        self.analyze_fixture()
        with patch.object(main, "generate_session_feedback", return_value=sample_session_feedback()) as generate:
            first = main.create_interview_feedback(self.session_id)
            second = main.create_interview_feedback(self.session_id)
        generate.assert_called_once_with("exact session prompt\n  unchanged")
        self.assertEqual(first["feedback_v1"], second["feedback_v1"])
        self.assertEqual(len(first["feedback_v1"]["answer_feedback"]), 5)
        self.assertTrue(all(v.get("feedback") is None for v in first["videos"]))

    def test_failed_gemini_keeps_analysis_and_can_retry(self):
        before = self.analyze_fixture()
        with patch.object(main, "generate_session_feedback", side_effect=main.FeedbackError("unavailable")):
            with self.assertRaises(HTTPException):
                main.create_interview_feedback(self.session_id)
        self.assertEqual(main.get_interview_result(self.session_id), before)
        with patch.object(main, "generate_session_feedback", return_value=sample_session_feedback()):
            self.assertEqual(main.create_interview_feedback(self.session_id)["feedback_status"], "ready")

    def test_re_recording_blocks_stale_feedback_without_call(self):
        self.analyze_fixture(); self.paths[0].write_bytes(b"new recording")
        with patch.object(main, "generate_session_feedback") as generate:
            with self.assertRaises(HTTPException) as error:
                main.create_interview_feedback(self.session_id)
        self.assertEqual(error.exception.status_code, 409)
        generate.assert_not_called()

    def test_old_text_cache_is_upgraded_to_segments(self):
        cache = {"texts": ["old"]*5, "provider": "gradio", "video_signature": main.video_signature(self.paths)}
        main.write_json(self.session_dir/"transcripts.json", cache)
        stt = [{"text": "new", "segments": [{"id": "S1", "start": 0, "end": 3, "text": "new"}]}]*5
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "gradio"}), patch.object(main, "call_gradio_stt", return_value=stt) as call:
            result = main.transcribe_interview(self.session_id)
            main.transcribe_interview(self.session_id)
        call.assert_called_once()
        self.assertEqual(result["answers"][0]["segments"][0]["id"], "S1")

    def test_changed_analysis_during_gemini_is_not_overwritten(self):
        self.analyze_fixture()
        def generate(prompt):
            changed = main.get_interview_result(self.session_id)
            changed["analyzed_at"] = "newer analysis"
            main.write_json(self.session_dir/"result.json", changed)
            return sample_session_feedback()
        with patch.object(main, "generate_session_feedback", side_effect=generate):
            with self.assertRaises(HTTPException) as error:
                main.create_interview_feedback(self.session_id)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(main.get_interview_result(self.session_id)["analyzed_at"], "newer analysis")


if __name__ == "__main__":
    unittest.main()
