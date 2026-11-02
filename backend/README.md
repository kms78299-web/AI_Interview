# Backend

## 실행

```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

별도 터미널에서 프론트엔드를 실행합니다.

```powershell
cd 최종frontend
npm run dev
```

녹화된 WebM 파일은 `backend/uploads`에 저장됩니다.

## Colab Gradio 연결

로컬 연결 주소는 `backend/.env`에 저장합니다. 새 Colab 주소가 나오면
`GRADIO_API_URL`을 갱신하고 다음 명령으로 백엔드를 다시 실행합니다.
`.env`는 Git에서 제외되어 있습니다.

```powershell
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --env-file .env
```

Colab에서 `/analyze` 엔드포인트가 영상 파일 목록 하나를 입력받고,
`run_session_inference(video_paths)`의 결과를 JSON 객체 출력 하나로 반환하도록 연결합니다.
백엔드는 저장된 영상 5개를 질문 순서대로 한 번에 전달합니다.
백엔드를 실행하기 전에 Colab의 임시 `gradio.live` 주소를 환경변수로 설정합니다.
기존 `colab/gradio_dummy_api.ipynb`의 이전 숫자 점수 응답은 현재 모델 계약과 다르므로 그대로 사용할 수 없습니다.

```powershell
$env:GRADIO_API_URL="https://xxxxxxxx.gradio.live"
$env:GRADIO_TIMEOUT_SECONDS="300"
uvicorn main:app --reload
```

답변 영상 5개가 모두 저장된 세션은 다음 API로 Colab에 분석을 요청할 수 있습니다.

```text
POST /api/interviews/{session_id}/analyze
```

## STT 먼저 표시하는 두 단계 처리

최신 `model/code/serve_gradio.py` 서버를 실행해야 합니다. 드라이브의 모델 코드는 로컬에서 수정하지 않습니다.

1. 프론트가 `POST /api/interviews/{session_id}/transcripts`를 호출합니다.
2. 백엔드가 Gradio `/stt`에 영상 목록과 언어 `en`을 전달하고 질문 순서의 텍스트 5개를 받습니다.
3. 텍스트는 `uploads/<session_id>/transcripts.json`에 저장하고 질문 ID와 연결하여 반환합니다.
4. 프론트는 텍스트를 보여주면서 기존 `/analyze`를 호출합니다.
5. 백엔드는 Gradio `/analyze`에 영상 목록, 저장한 텍스트 JSON 문자열, 빈 문자열, `True`의 네 인자를 순서대로 전달합니다. STT를 다시 실행하지 않습니다.

`/transcripts` 응답은 `status=ready`, `session_id`, `is_dummy`, `provider`,
`answers=[{question_id, transcript}, ...]`입니다. 빈 STT 문자열도 보존합니다.
같은 영상의 텍스트는 재요청 시 재사용하며 파일명·크기·수정 시간과 제공자 모드가 바뀌면 새로 변환합니다.
텍스트 없이 `/analyze`만 호출하는 기존 방식은 두 번째 인자에 빈 문자열을 전달해 유지합니다.
STT나 평가 실패 시 실제 결과 대신 더미를 자동 반환하지 않습니다.

## 분석 결과 형식과 저장

분석 API는 `run_session_inference`의 다음 원본 구조를 검사한 뒤 보존합니다.

- `session`: `n_videos=5`, `smoothing`, `scores` (면접 전체 평가)
- `videos`: 입력 순서의 영상 5개. 각각 `video`, `transcript`, `scores`, 선택적으로 `single_clip_percentile`
- `explanation`: 모델이 반환한 설명 문자열

평가 항목은 `interview_score`, `answer_score`, `speaking_skills`,
`confidence_score`, `facial_expression`, `overall_performance`입니다.
각 점수의 `raw`, `percentile`, `band`와 영상별 `smoothed_raw`를 그대로 보존합니다.
백분위 기준 파일이 없으면 `percentile`, `band`의 `null`을 허용합니다.
점수를 100점으로 변환하거나 반올림하거나 백분위를 다시 평균 내지 않습니다.
점수 표시 방식은 추후 결정합니다. `percentile`은 100점 만점 점수가 아닙니다.

프론트엔드 연결용 필드도 함께 제공합니다.

| API 필드 | 원본 연결 |
| --- | --- |
| `overall_score` | `session.scores.overall_performance` |
| `average_scores` | `session.scores`의 복사본. 호환용 이름이며 백엔드에서 평균을 계산하지 않음 |
| `answers` | `videos`에 질문 번호·질문 ID·질문·카테고리·저장 파일명을 추가한 목록 |
| `combined_transcript` | 질문 순서대로 합친 `transcript` |
| `is_dummy` | 모델 응답에 없으면 `false`, 명시적 mock 모드에서는 `true` |

`videos`에도 동일한 로컬 질문 정보를 추가합니다. 원본 모델 경로 `video`는 보존하며,
실제 재생은 기존 영상 조회 API를 사용합니다.
`aggregation_method`는 `model_session_scores`입니다.
모델 API의 입력·출력 순서 보장에 따라 결과 인덱스로 로컬 질문 ID를 연결합니다.
`videos[i].video`는 서버 임시 경로이므로 로컬 파일명과 비교하거나 재생 주소로 사용하지 않습니다.
`stats`와 `llm_prompt`는 원본 그대로 저장·반환하며 `null`도 허용합니다.
LLM은 아직 호출하지 않습니다. 추후 프롬프트가 `null`인 답변은 개별 피드백 요청에서 제외해야 합니다.
기존 숫자 전용 응답, 항목 누락, 영상 개수 불일치, NaN/Infinity는 502로 처리하고 저장하지 않습니다.

성공 결과는 `uploads/<session_id>/result.json`에 저장합니다.
`manifest.json`에도 질문별 STT, 통합 STT, 분석 시각과 제공자를 기록합니다.
저장된 결과는 `GET /api/interviews/{session_id}/result`로 다시 가져옵니다.

## Colab 없이 화면 개발하기

기본 모드는 실제 Gradio 호출입니다. 더미 모드는 명시적으로 설정합니다.

```powershell
$env:GRADIO_ANALYSIS_MODE="mock"
uvicorn main:app --reload
```

이 모드는 저장된 영상 5개의 존재만 확인하고, 모델/STT를 실행하지 않은
고정된 가짜 결과를 새 JSON 구조로 반환합니다. `is_dummy=true`, `provider=mock`입니다.
Gradio 호출 실패 시 자동으로 더미 결과를 반환하지 않습니다.

실제 모델 연결로 돌아가려면 백엔드 실행 전에 다음과 같이 설정합니다.

```powershell
$env:GRADIO_ANALYSIS_MODE="gradio"
$env:GRADIO_API_URL="https://xxxxxxxx.gradio.live"
uvicorn main:app --reload
```

이번 변경은 백엔드 반환 형식만 대상으로 합니다. 실제 모델 실행 서버 연결과
프론트엔드 점수 표시·LLM 피드백 연결은 별도 작업입니다.

검증: `.venv\Scripts\python.exe -m unittest test_analysis -v`
