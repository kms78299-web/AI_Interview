import copy
import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from gemini_feedback import generate_feedback, validate_feedback, FeedbackError


def sample_feedback():
    return {"content_rating": "적절", "content_reason": "질문에 답변했습니다.",
            "overall_band": "중", "band_reason": "구체적인 설명이 더 필요합니다.",
            **{key: {"strength": "강점 예시", "improvement": "개선점 예시"} for key in ("text", "audio", "video")},
            "summary": "구체적인 경험을 덧붙여 보세요."}


def response(value):
    return io.BytesIO(json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(value)}]}}]}).encode())


class FeedbackTests(unittest.TestCase):
    def test_session_prompt_uses_session_schema_and_only_one_successful_call(self):
        from gemini_feedback import generate_session_feedback
        from test_session_feedback import sample_session_feedback
        prompt = "exact session prompt\n  unchanged"
        def fake(request, timeout):
            payload = json.loads(request.data)
            self.assertEqual(payload["contents"][0]["parts"][0]["text"], prompt)
            schema = payload["generationConfig"]["responseFormat"]["text"]["schema"]
            self.assertIn("answer_feedback", schema["required"])
            self.assertNotIn("content_rating", schema["required"])
            return response(sample_session_feedback())
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}), patch("gemini_feedback.urllib.request.urlopen", side_effect=fake) as call:
            self.assertEqual(generate_session_feedback(prompt), sample_session_feedback())
        call.assert_called_once()

    def test_exact_prompt_json_mode_and_key_in_header(self):
        prompt = "모델 원본 프롬프트\n공백도 그대로  유지"
        def fake(request, timeout):
            payload = json.loads(request.data)
            self.assertEqual(payload["contents"], [{"role": "user", "parts": [{"text": prompt}]}])
            self.assertEqual(payload["generationConfig"]["temperature"], 0.4)
            self.assertEqual(payload["generationConfig"]["responseFormat"]["text"]["mimeType"], "APPLICATION_JSON")
            self.assertNotIn("test-secret", request.full_url)
            self.assertEqual(request.get_header("X-goog-api-key"), "test-secret")
            return response(sample_feedback())
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}), patch("gemini_feedback.urllib.request.urlopen", side_effect=fake):
            self.assertEqual(generate_feedback(prompt), sample_feedback())

    def test_invalid_json_schema_retries_once(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}), patch("gemini_feedback.urllib.request.urlopen", side_effect=[response({}), response(sample_feedback())]) as call:
            self.assertEqual(generate_feedback("test"), sample_feedback())
            self.assertEqual(call.call_count, 2)

    def test_rating_rules_and_missing_sections_rejected(self):
        for mutate in (lambda x: x.update(content_rating="부적절", overall_band="상"),
                       lambda x: x.update(content_rating="부분적", overall_band="상"),
                       lambda x: x["video"].update(strength="")):
            value = copy.deepcopy(sample_feedback()); mutate(value)
            with self.assertRaises(ValueError):
                validate_feedback(value)

    def test_provider_errors_do_not_leak_key_or_body(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}), patch("gemini_feedback.urllib.request.urlopen", side_effect=urllib.error.HTTPError("https://example.com/test-secret", 429, "private body test-secret", {}, None)) as call:
            with self.assertRaises(FeedbackError) as error:
                generate_feedback("private prompt")
            self.assertNotIn("test-secret", str(error.exception))
            self.assertEqual(call.call_count, 1)

    def test_missing_key_never_calls_provider(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), patch("gemini_feedback.urllib.request.urlopen") as call:
            with self.assertRaises(FeedbackError):
                generate_feedback("test")
            call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
