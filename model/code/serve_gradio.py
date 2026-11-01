"""
Colab에서 Gradio 서버를 띄워 로컬 백엔드(FastAPI)가 HTTP로 호출하게 하는 파일.

API 2개 (api_name):
  /stt      영상 N개(질문 순서) -> 대본 텍스트 N개 (같은 순서의 JSON 리스트). STT만 한다.
  /analyze  영상 N개 + 위 텍스트 -> {"session", "videos", "explanation"} (run_session_inference와 같은 형식).
            텍스트를 주면 STT를 다시 돌리지 않고 그대로 점수 분석에 쓴다. 텍스트를 비우면 STT부터 한다.

Colab에서:
    !pip install -r requirements_colab.txt
    %cd model/code
    !python serve_gradio.py --share          # 출력되는 https://xxxx.gradio.live 주소를 백엔드에 설정

로컬 백엔드에서 호출 (pip install gradio_client):
    from gradio_client import Client, handle_file
    client = Client("https://xxxx.gradio.live")
    files = [handle_file(p) for p in video_paths]                      # 질문 순서대로
    texts = client.predict(files, api_name="/stt")                     # ["1번 답변", ..., "5번 답변"]
    result = client.predict(files, json.dumps(texts), api_name="/analyze")   # session / videos / explanation
영상 순서가 곧 질문 순서이고, 반환되는 텍스트와 videos도 같은 순서다. 질문 ID 연결은 백엔드가 한다.
"""

import argparse
import json

import gradio as gr

import inference


def _paths(files):
    if not files:
        raise gr.Error("영상 파일이 없습니다.")
    # gradio 버전에 따라 경로 문자열 또는 {"path": ...} / 파일 객체(.name)로 올 수 있어 모두 처리
    return [f if isinstance(f, str) else (f["path"] if isinstance(f, dict) else f.name) for f in files]


def stt_api(files, language="en"):
    return inference.transcribe_videos(_paths(files), language=language)


def analyze_api(files, transcripts_json=""):
    paths = _paths(files)
    transcripts = None
    if transcripts_json and transcripts_json.strip():
        try:
            transcripts = json.loads(transcripts_json)
        except json.JSONDecodeError as e:
            raise gr.Error(f"transcripts_json이 JSON이 아님: {e}")
        if not isinstance(transcripts, list) or not all(isinstance(t, str) for t in transcripts):
            raise gr.Error("transcripts_json은 문자열 리스트여야 함")
        if len(transcripts) != len(paths):
            raise gr.Error(f"텍스트 {len(transcripts)}개와 영상 {len(paths)}개의 개수가 다름")
    return inference.run_session_inference(paths, transcripts=transcripts)


def build_app():
    with gr.Blocks(title="AI Interview model") as demo:
        gr.Markdown("영상은 **질문 순서대로** 올리세요. `/stt`는 텍스트만, `/analyze`는 점수 분석(텍스트를 주면 STT 생략).")
        files = gr.File(label="영상 (질문 순서)", file_count="multiple", type="filepath")
        language = gr.Textbox(label="STT 언어", value="en")
        transcripts = gr.Textbox(label="transcripts_json (비우면 STT부터)", lines=3)
        stt_out = gr.JSON(label="STT 결과")
        result_out = gr.JSON(label="분석 결과")
        gr.Button("STT만").click(stt_api, [files, language], stt_out, api_name="stt")
        gr.Button("점수 분석").click(analyze_api, [files, transcripts], result_out, api_name="analyze")
    return demo.queue(default_concurrency_limit=1)  # 모델이 하나라 요청을 한 번에 하나씩 처리


def main():
    p = argparse.ArgumentParser(description="Gradio 서버 (STT / 점수 분석 API)")
    p.add_argument("--share", action="store_true", help="Colab 등에서 외부 접속용 공개 주소(gradio.live) 생성")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--preload", action="store_true", help="서버 시작 때 모델을 미리 로드 (첫 요청 지연 방지)")
    args = p.parse_args()
    if args.preload:
        with inference._LOCK:
            inference._MODELS = inference.load_models()
    build_app().launch(share=args.share, server_port=args.port, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
