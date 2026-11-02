import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from gradio_client import Client, handle_file

from analysis_results import make_mock_result, prepare_analysis_result


app = FastAPI(title="RecruitView API")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
QUESTION_FILE = BASE_DIR / "questions.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CONTENT_TYPES = {
    "video/webm",
    "video/webm;codecs=vp8,opus",
    "video/webm;codecs=vp9,opus",
}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/interviews")
def create_interview() -> dict:
    with QUESTION_FILE.open(encoding="utf-8") as question_file:
        bank = json.load(question_file)

    selected = [bank["required"]]
    selected.extend(random.choice(questions) for questions in bank["categories"].values())
    questions = [
        {**question, "order": order}
        for order, question in enumerate(selected, start=1)
    ]

    session_id = uuid4().hex
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir()
    manifest = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "questions_created",
        "questions": questions,
        "answers": [
            {
                "order": question["order"],
                "question_id": question["id"],
                "category": question["category"],
                "question": question["text"],
                "status": "pending",
                "video_filename": None,
                "duration_seconds": None,
                "transcript": None,
            }
            for question in questions
        ],
        "combined_transcript": None,
    }
    with (session_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    return {"session_id": session_id, "questions": questions}


@app.post("/api/interviews/{session_id}/answers/{question_id}")
async def upload_answer(
    session_id: str,
    question_id: str,
    video: UploadFile = File(...),
    duration_seconds: float = Form(...),
) -> dict[str, str | int | float]:
    if len(session_id) != 32 or any(character not in "0123456789abcdef" for character in session_id):
        raise HTTPException(status_code=404, detail="면접 세션을 찾을 수 없습니다.")

    session_dir = UPLOAD_DIR / session_id
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="면접 세션을 찾을 수 없습니다.")

    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    answer = next(
        (item for item in manifest["answers"] if item["question_id"] == question_id),
        None,
    )
    if answer is None:
        raise HTTPException(status_code=404, detail="이 세션에 없는 질문입니다.")
    if duration_seconds < 0 or duration_seconds > 61:
        raise HTTPException(status_code=400, detail="답변 시간이 올바르지 않습니다.")
    if video.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="WebM 영상만 업로드할 수 있습니다.")

    filename = f'{answer["order"]:02d}_{question_id}.webm'
    saved_path = session_dir / filename
    size = 0
    try:
        with saved_path.open("wb") as output:
            while chunk := await video.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="영상은 100MB까지 업로드할 수 있습니다.")
                output.write(chunk)
    except Exception:
        saved_path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    if size == 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="비어 있는 영상입니다.")

    answer.update(
        {
            "status": "saved",
            "video_filename": filename,
            "duration_seconds": round(duration_seconds, 2),
            "size_bytes": size,
        }
    )
    if all(item["status"] == "saved" for item in manifest["answers"]):
        manifest["status"] = "uploaded"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    return {
        "status": "saved",
        "session_id": session_id,
        "question_id": question_id,
        "filename": filename,
        "duration_seconds": round(duration_seconds, 2),
        "size_bytes": size,
    }


@app.get("/api/interviews/{session_id}/answers/{question_id}/video")
def get_answer_video(session_id: str, question_id: str) -> FileResponse:
    if len(session_id) != 32 or any(character not in "0123456789abcdef" for character in session_id):
        raise HTTPException(status_code=404, detail="면접 세션을 찾을 수 없습니다.")

    session_dir = UPLOAD_DIR / session_id
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="면접 세션을 찾을 수 없습니다.")

    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    answer = next(
        (item for item in manifest["answers"] if item["question_id"] == question_id),
        None,
    )
    if answer is None or answer["status"] != "saved" or not answer["video_filename"]:
        raise HTTPException(status_code=404, detail="저장된 답변 영상을 찾을 수 없습니다.")

    video_path = session_dir / answer["video_filename"]
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="저장된 답변 영상을 찾을 수 없습니다.")
    return FileResponse(video_path, media_type="video/webm")


def analysis_inputs(session_id: str):
    if len(session_id) != 32 or any(character not in "0123456789abcdef" for character in session_id):
        raise HTTPException(status_code=404, detail="면접 세션을 찾을 수 없습니다.")

    session_dir = UPLOAD_DIR / session_id
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="면접 세션을 찾을 수 없습니다.")

    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    answers = sorted(manifest["answers"], key=lambda item: item["order"])
    if len(answers) != 5 or any(item["status"] != "saved" for item in answers):
        raise HTTPException(status_code=409, detail="저장된 답변 영상 5개가 모두 필요합니다.")

    video_paths = [session_dir / item["video_filename"] for item in answers]
    if any(not path.is_file() for path in video_paths):
        raise HTTPException(status_code=404, detail="저장된 답변 영상 파일을 찾을 수 없습니다.")

    return session_dir, manifest_path, manifest, answers, video_paths


def video_signature(video_paths):
    return [{"filename": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in video_paths]


def validate_transcripts(texts):
    if not isinstance(texts, list) or len(texts) != 5 or not all(isinstance(text, str) for text in texts):
        raise HTTPException(status_code=502, detail="STT 결과는 질문 순서의 답변 텍스트 5개여야 합니다.")


def cached_transcripts(session_dir, video_paths, mode):
    path = session_dir / "transcripts.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        cache = json.load(source)
    if cache.get("video_signature") != video_signature(video_paths) or cache.get("provider") != mode:
        return None
    validate_transcripts(cache.get("texts"))
    return cache


@app.post("/api/interviews/{session_id}/transcripts")
def transcribe_interview(session_id: str) -> dict:
    session_dir, _, _, answers, video_paths = analysis_inputs(session_id)
    mode = os.environ.get("GRADIO_ANALYSIS_MODE", "gradio").strip().lower()
    if mode not in ("mock", "gradio"):
        raise HTTPException(status_code=500, detail="GRADIO_ANALYSIS_MODE은 gradio 또는 mock이어야 합니다.")
    cache = cached_transcripts(session_dir, video_paths, mode)
    if cache is None:
        signature = video_signature(video_paths)
        texts = ([video["transcript"] for video in make_mock_result(video_paths)["videos"]]
                 if mode == "mock" else call_gradio_stt(video_paths))
        validate_transcripts(texts)
        if video_signature(video_paths) != signature:
            raise HTTPException(status_code=409, detail="변환 중 영상이 변경됐어요. 답변 텍스트를 다시 요청해 주세요.")
        cache = {"texts": texts, "provider": mode, "is_dummy": mode == "mock",
                 "video_signature": signature}
        write_json(session_dir / "transcripts.json", cache)
    return {"status": "ready", "session_id": session_id,
            "provider": mode, "is_dummy": cache["is_dummy"],
            "answers": [{"question_id": answer["question_id"], "transcript": text}
                        for answer, text in zip(answers, cache["texts"])]}


@app.post("/api/interviews/{session_id}/analyze")
def analyze_interview(session_id: str) -> dict:
    session_dir, manifest_path, manifest, answers, video_paths = analysis_inputs(session_id)

    analysis_mode = os.environ.get("GRADIO_ANALYSIS_MODE", "gradio").strip().lower()
    if analysis_mode == "mock":
        result = make_mock_result(video_paths)
        provider = "mock"
    elif analysis_mode == "gradio":
        cache = cached_transcripts(session_dir, video_paths, analysis_mode)
        if (session_dir / "transcripts.json").is_file() and cache is None:
            raise HTTPException(status_code=409, detail="영상 정보가 변경됐어요. 답변 텍스트를 다시 불러온 뒤 분석해 주세요.")
        result = call_gradio(video_paths, cache["texts"]) if cache else call_gradio(video_paths)
        provider = "gradio"
    else:
        raise HTTPException(status_code=500, detail="GRADIO_ANALYSIS_MODE은 gradio 또는 mock이어야 합니다.")

    try:
        result = prepare_analysis_result(result, video_paths)
    except ValueError as error:
        raise HTTPException(status_code=502, detail=f"분석 결과 JSON 형식이 올바르지 않습니다: {error}") from error

    # Attach trusted local question metadata in the same order as uploaded files.
    enriched_answers = []
    for local_answer, analyzed_answer in zip(answers, result["answers"]):
        enriched_answers.append({
            **analyzed_answer,
            "question_id": local_answer["question_id"],
            "question": local_answer["question"],
            "category": local_answer["category"],
            "filename": local_answer["video_filename"],
        })

    response = {
        **result,
        "session_id": session_id,
        "provider": provider,
        "answers": enriched_answers,
        "videos": [
            {**video, **{
                key: answer[key]
                for key in ("question_index", "question_id", "question", "category", "filename")
            }}
            for video, answer in zip(result["videos"], enriched_answers)
        ],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "aggregation_method": "model_session_scores",
        "aggregation_note": "run_session_inference의 session.scores를 그대로 제공합니다. 점수 표시 방식은 아직 결정되지 않았습니다.",
    }
    write_json(session_dir / "result.json", response)

    for local_answer, analyzed_answer in zip(answers, enriched_answers):
        local_answer["transcript"] = analyzed_answer["transcript"]
    manifest["combined_transcript"] = response["combined_transcript"]
    manifest["status"] = "analyzed"
    manifest["analysis"] = {
        "result_filename": "result.json",
        "provider": provider,
        "is_dummy": response["is_dummy"],
        "analyzed_at": response["analyzed_at"],
    }
    write_json(manifest_path, manifest)
    return response


def write_json(path: Path, data: dict) -> None:
    temporary_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def call_gradio(video_paths: list[Path], transcripts=None) -> dict:
    return gradio_predict(video_paths, "/analyze", json.dumps(transcripts, ensure_ascii=False) if transcripts is not None else "", "", True)


def call_gradio_stt(video_paths: list[Path]):
    return gradio_predict(video_paths, "/stt", "en")


def gradio_predict(video_paths: list[Path], api_name: str, *extra):
    gradio_api_url = os.environ.get("GRADIO_API_URL", "").strip().rstrip("/")
    if not gradio_api_url:
        raise HTTPException(
            status_code=503,
            detail="GRADIO_API_URL이 설정되지 않았습니다. Colab의 gradio.live 주소를 설정해 주세요.",
        )
    try:
        gradio_timeout_seconds = float(os.environ.get("GRADIO_TIMEOUT_SECONDS", "300"))
    except ValueError as error:
        raise HTTPException(status_code=500, detail="GRADIO_TIMEOUT_SECONDS 설정이 올바르지 않습니다.") from error

    try:
        client = Client(
            gradio_api_url,
            verbose=False,
            httpx_kwargs={"timeout": gradio_timeout_seconds},
        )
        result = client.predict(
            [handle_file(str(path)) for path in video_paths],
            *extra,
            api_name=api_name,
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Colab Gradio 분석 서버 호출에 실패했습니다: {error}",
        ) from error

    return result


@app.get("/api/interviews/{session_id}/result")
def get_interview_result(session_id: str) -> dict:
    if len(session_id) != 32 or any(character not in "0123456789abcdef" for character in session_id):
        raise HTTPException(status_code=404, detail="면접 세션을 찾을 수 없습니다.")
    result_path = UPLOAD_DIR / session_id / "result.json"
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="저장된 분석 결과가 없습니다.")
    with result_path.open(encoding="utf-8") as result_file:
        return json.load(result_file)


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...)) -> dict[str, str | int]:
    if video.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="WebM 영상만 업로드할 수 있습니다.")

    saved_name = f"{uuid4().hex}.webm"
    saved_path = UPLOAD_DIR / saved_name
    size = 0

    try:
        with saved_path.open("wb") as output:
            while chunk := await video.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="영상은 100MB까지 업로드할 수 있습니다.")
                output.write(chunk)
    except Exception:
        saved_path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    if size == 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="비어 있는 영상입니다.")

    return {
        "status": "saved",
        "filename": saved_name,
        "content_type": video.content_type,
        "size_bytes": size,
    }
