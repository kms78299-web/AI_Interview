"""Local, single-question entry point for the existing Colab model."""

import json
import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from analysis_results import make_mock_result, validate_scores
from main import CHUNK_SIZE, MAX_UPLOAD_SIZE, QUESTION_FILE, call_gradio


app = FastAPI(title="RecruitView Model Lab")
QUESTION = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))["required"]["text"]
VIDEO_TYPES = {"video/webm": ".webm", "video/mp4": ".mp4", "video/quicktime": ".mov"}
logger = logging.getLogger(__name__)


@app.get("/api/demo")
def demo_config():
    return {"question": QUESTION, "configured": bool(os.environ.get("GRADIO_API_URL", "").strip())}


@app.get("/api/demo/example")
def demo_example():
    example = make_mock_result([Path("example.webm")])["videos"][0]
    return {"is_example": True, "elapsed_ms": None, "scores": example["scores"],
            "transcript": "I’m a computer science student who enjoys building tools that make everyday tasks easier. In my last project, I worked with a small team to build an interview practice app.",
            "warning": None}


def read_prediction(result):
    if not isinstance(result, dict) or result.get("is_dummy") or result.get("status", "success") != "success":
        raise ValueError("Expected a live model response")
    videos = result.get("videos")
    if not isinstance(videos, list) or len(videos) != 1 or not isinstance(videos[0], dict):
        raise ValueError("Expected a single clip")
    clip = videos[0]
    transcript = clip.get("transcript")
    scores = clip.get("scores")
    warning = clip.get("warning")
    if not isinstance(transcript, str):
        raise ValueError("Expected a transcript")
    if scores is None:
        if clip.get("skipped") is not True or not isinstance(warning, str) or not warning.strip():
            raise ValueError("Expected the reason the clip was skipped")
    else:
        validate_scores(scores, "videos[0].scores")
    prediction = {"scores": scores, "transcript": transcript, "warning": warning}
    json.dumps(prediction, allow_nan=False)
    return prediction


@app.post("/api/demo/analyze")
def analyze_demo(video: UploadFile = File(...)):
    mime = (video.content_type or "").split(";", 1)[0].lower()
    suffix = VIDEO_TYPES.get(mime)
    if not suffix and mime in {"", "application/octet-stream"}:
        extension = Path(video.filename or "").suffix.lower()
        if extension in VIDEO_TYPES.values():
            suffix = extension
    if not suffix:
        raise HTTPException(415, "WebM, MP4 또는 MOV 영상을 올려 주세요.")
    if not video.size:
        raise HTTPException(400, "비어 있는 영상입니다.")
    if video.size > MAX_UPLOAD_SIZE:
        raise HTTPException(413, "영상은 100MB까지 업로드할 수 있습니다.")
    with TemporaryDirectory(prefix="recruitview-demo-") as folder:
        path = Path(folder) / f"answer{suffix}"
        with path.open("wb") as output:
            shutil.copyfileobj(video.file, output, CHUNK_SIZE)
        started = perf_counter()
        try:
            result = call_gradio([path], questions=[QUESTION])
        except HTTPException as error:
            if error.status_code != 502:
                raise
            logger.warning("Colab model request failed")
            raise HTTPException(502, "Colab 모델에 연결하거나 분석하는 데 실패했어요. Colab 실행 상태와 GRADIO_API_URL을 확인해 주세요.") from error
        elapsed_ms = round((perf_counter() - started) * 1000)
        try:
            prediction = read_prediction(result)
        except (TypeError, ValueError) as error:
            raise HTTPException(502, "모델에서 영상 한 개의 분석 결과를 받지 못했어요. 최신 serve_gradio.py로 실행했는지 확인해 주세요.") from error
    return {"is_example": False, "elapsed_ms": elapsed_ms, **prediction}


app.mount("/", StaticFiles(directory=Path(__file__).with_name("model_demo"), html=True), name="demo")
