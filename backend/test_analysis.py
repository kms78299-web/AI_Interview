import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

import main
from analysis_results import make_mock_result, prepare_analysis_result, validate_analysis_result


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.upload_dir = Path(self.temp.name)
        self.session_id = "a" * 32
        self.session_dir = self.upload_dir / self.session_id
        self.session_dir.mkdir()
        self.manifest = {
            "session_id": self.session_id,
            "status": "uploaded",
            "answers": [
                {
                    "order": index,
                    "question_id": f"q{index}",
                    "question": f"Question {index}",
                    "category": "test",
                    "status": "saved",
                    "video_filename": f"{index:02d}_q{index}.webm",
                    "transcript": None,
                }
                for index in range(1, 6)
            ],
            "combined_transcript": None,
        }
        self.paths = []
        for answer in self.manifest["answers"]:
            path = self.session_dir / answer["video_filename"]
            path.write_bytes(b"fixture-not-real-video")
            self.paths.append(path)
        (self.session_dir / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        self.addCleanup(patch.stopall)
        patch.object(main, "UPLOAD_DIR", self.upload_dir).start()

    def test_mock_persists_results_and_transcripts_without_gradio(self):
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "mock"}), patch.object(main, "Client") as client:
            result = main.analyze_interview(self.session_id)
        client.assert_not_called()
        self.assertTrue(result["is_dummy"])
        self.assertEqual(result["provider"], "mock")
        self.assertEqual(result["answers"][0]["question_id"], "q1")
        self.assertEqual(main.get_interview_result(self.session_id), result)
        saved_manifest = json.loads((self.session_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_manifest["status"], "analyzed")
        self.assertEqual(saved_manifest["answers"][0]["transcript"], result["answers"][0]["transcript"])
        self.assertEqual(saved_manifest["combined_transcript"], result["combined_transcript"])

    def test_real_response_keeps_scores_and_question_order(self):
        response = make_mock_result(self.paths)
        del response["is_dummy"]
        del response["status"]
        response["session"]["scores"]["overall_performance"]["raw"] = -1.23456789
        original = copy.deepcopy(response)
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "gradio"}), patch.object(main, "call_gradio", return_value=response) as call:
            result = main.analyze_interview(self.session_id)
        call.assert_called_once_with(self.paths)
        self.assertEqual(result["session"], response["session"])
        self.assertEqual(result["average_scores"], response["session"]["scores"])
        self.assertEqual(result["overall_score"], response["session"]["scores"]["overall_performance"])
        self.assertEqual(result["explanation"], response["explanation"])
        for index, video in enumerate(response["videos"]):
            for key, value in video.items():
                self.assertEqual(result["videos"][index][key], value)
                self.assertEqual(result["answers"][index][key], value)
            self.assertEqual(result["videos"][index]["question_id"], f"q{index + 1}")
        self.assertEqual(response, original)
        self.assertEqual(main.get_interview_result(self.session_id), result)
        self.assertEqual(result["provider"], "gradio")
        self.assertFalse(result["is_dummy"])

    def test_bad_response_does_not_save_or_replace_manifest(self):
        original_manifest = (self.session_dir / "manifest.json").read_bytes()
        response = make_mock_result(self.paths)
        response["videos"][0]["scores"]["interview_score"]["percentile"] = 101
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "gradio"}), patch.object(main, "call_gradio", return_value=response):
            with self.assertRaises(HTTPException) as error:
                main.analyze_interview(self.session_id)
        self.assertEqual(error.exception.status_code, 502)
        self.assertFalse((self.session_dir / "result.json").exists())
        self.assertEqual((self.session_dir / "manifest.json").read_bytes(), original_manifest)

    def test_null_percentiles_allowed_but_legacy_and_nonfinite_scores_rejected(self):
        response = make_mock_result(self.paths)
        for score in response["session"]["scores"].values():
            score.update(percentile=None, band=None)
        for answer in response["videos"]:
            for score in answer["scores"].values():
                score.update(percentile=None, band=None)
        validate_analysis_result(response)
        legacy = copy.deepcopy(response)
        legacy["session"]["scores"]["interview_score"] = 0.2
        with self.assertRaises(ValueError):
            validate_analysis_result(legacy)
        response["videos"][0]["scores"]["interview_score"]["raw"] = float("nan")
        with self.assertRaises(ValueError):
            validate_analysis_result(response)

    def test_gradio_cache_paths_preserve_file_mapping(self):
        response = make_mock_result(self.paths)
        for index, video in enumerate(response["videos"]):
            video["video"] = f"/tmp/gradio/cache/{self.paths[index].name}"
        response["videos"][0]["video"] = "C:\\cache\\" + self.paths[0].name
        result = prepare_analysis_result(response, self.paths)
        self.assertEqual([answer["filename"] for answer in result["answers"]], [p.name for p in self.paths])

    def test_mismatched_video_count_is_rejected(self):
        for case in ("missing", "count"):
            with self.subTest(case=case):
                response = make_mock_result(self.paths)
                if case == "missing":
                    response["videos"].pop()
                else:
                    response["session"]["n_videos"] = 4
                with self.assertRaises(ValueError):
                    prepare_analysis_result(response, self.paths)

    def test_invalid_smoothed_scores_and_non_json_values_are_rejected(self):
        for value in (True, float("nan"), float("inf"), "0.2"):
            with self.subTest(value=value):
                response = make_mock_result(self.paths)
                response["videos"][0]["scores"]["answer_score"]["smoothed_raw"] = value
                with self.assertRaises(ValueError):
                    validate_analysis_result(response)
        response = make_mock_result(self.paths)
        response["unexpected"] = {1, 2}
        with self.assertRaises(ValueError):
            validate_analysis_result(response)

    def test_gradio_failure_does_not_fall_back_to_fake_result(self):
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "gradio"}), patch.object(
            main, "call_gradio", side_effect=HTTPException(status_code=502, detail="offline")
        ), patch.object(main, "make_mock_result") as mock_result:
            with self.assertRaises(HTTPException):
                main.analyze_interview(self.session_id)
        mock_result.assert_not_called()
        self.assertFalse((self.session_dir / "result.json").exists())

    def test_stt_is_cached_then_reused_for_scoring_in_question_order(self):
        texts = [f"Answer {i}" for i in range(5)]
        model_result = make_mock_result(self.paths)
        model_result["is_dummy"] = False
        for video, text in zip(model_result["videos"], texts):
            video["transcript"] = text
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "gradio"}), patch.object(
            main, "call_gradio_stt", return_value=texts
        ) as stt, patch.object(main, "call_gradio", return_value=model_result) as analyze:
            intermediate = main.transcribe_interview(self.session_id)
            self.assertEqual(main.transcribe_interview(self.session_id), intermediate)
            self.assertFalse((self.session_dir / "result.json").exists())
            result = main.analyze_interview(self.session_id)
        stt.assert_called_once_with(self.paths)
        analyze.assert_called_once_with(self.paths, texts)
        self.assertEqual([a["question_id"] for a in intermediate["answers"]], [f"q{i}" for i in range(1, 6)])
        self.assertEqual([v["transcript"] for v in result["videos"]], texts)
        self.assertFalse(intermediate["is_dummy"])

    def test_bad_stt_is_not_saved_and_unchanged_files_only_are_reused(self):
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "gradio"}), patch.object(main, "call_gradio_stt", return_value=["missing"]):
            with self.assertRaises(HTTPException):
                main.transcribe_interview(self.session_id)
        self.assertFalse((self.session_dir / "transcripts.json").exists())
        texts = ["", "two", "three", "four", "five"]
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "gradio"}), patch.object(main, "call_gradio_stt", return_value=texts) as stt:
            main.transcribe_interview(self.session_id)
            self.paths[0].write_bytes(b"replacement-video")
            with self.assertRaises(HTTPException) as error:
                main.analyze_interview(self.session_id)
            self.assertEqual(error.exception.status_code, 409)
            main.transcribe_interview(self.session_id)
            self.assertEqual(stt.call_count, 2)

    def test_gradio_endpoints_receive_language_and_transcript_json(self):
        texts = ["", "two", "three", "four", "five"]
        with patch.dict(main.os.environ, {"GRADIO_API_URL": "https://test.gradio.live"}), patch.object(main, "Client") as client, patch.object(main, "handle_file", side_effect=str):
            main.call_gradio_stt(self.paths)
            client.return_value.predict.assert_called_with([str(p) for p in self.paths], "en", api_name="/stt")
            main.call_gradio(self.paths, texts)
            client.return_value.predict.assert_called_with([str(p) for p in self.paths], json.dumps(texts, ensure_ascii=False), "", True, api_name="/analyze")

    def test_temporary_paths_and_nullable_llm_data_preserve_index_mapping(self):
        response = make_mock_result(self.paths)
        for index, video in enumerate(response["videos"]):
            video.update(video="/tmp/model/generated.webm", stats=None if index == 1 else {"duration": 12.3},
                         llm_prompt=None if index == 1 else f"Prompt {index}")
        with patch.dict(main.os.environ, {"GRADIO_ANALYSIS_MODE": "gradio"}), patch.object(main, "call_gradio", return_value=response):
            result = main.analyze_interview(self.session_id)
        self.assertEqual([v["question_id"] for v in result["videos"]], [f"q{i}" for i in range(1, 6)])
        self.assertEqual(result["videos"][0]["llm_prompt"], "Prompt 0")
        self.assertIsNone(result["videos"][1]["llm_prompt"])
        self.assertIsNone(result["videos"][1]["stats"])
        self.assertEqual(main.get_interview_result(self.session_id), result)


if __name__ == "__main__":
    unittest.main()
