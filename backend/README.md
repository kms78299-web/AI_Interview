# Backend

## 현재 처리 방식 (2026-11-23)

숫자 점수는 화면에 표시하지 않는다. 모델 결과는 내부에 보존하고, Gemini가 세션 전체 상·중·하와 질문별 텍스트 코칭, 음성·영상 종합 피드백을 만든다. 모델 서버 계약은 `model/API.md`, 화면 계약은 `docs/feedback-data-contract.md`를 참고한다.

1. `POST /api/interviews/{session_id}/transcripts`
   - Gradio `/stt_segments(files, "en")`을 호출한다.
   - 영상 순서의 `{text, segments}` 5개를 검증하고 `transcripts.json`에 저장한다.
   - 화면 응답은 `answers=[{question_id, transcript, segments}]`이다. 전체 대본은 원본 `text` 그대로다.
   - 파일명·크기·수정 시각·제공자와 `segments-v1` 캐시 버전이 같으면 재사용한다. 이전 텍스트 전용 캐시는 구간 STT로 갱신한다.
2. `POST /api/interviews/{session_id}/analyze`
   - Gradio `/analyze`에 `files`, 대본 JSON, 실제 질문 원문 JSON, `True`, 영상 순서의 구간 JSON을 보낸다.
   - 처음부터 분석만 요청하고 캐시가 없으면 빈 대본·구간을 보내 모델 서버가 STT부터 수행한다.
   - 원본 `session`, `videos`, 통계·마커·구간·프롬프트와 내부 점수를 보존한다. 점수에서 제외된 짧은 답변도 전체 5개 목록에는 남는다.
   - `session.n_videos`는 점수를 낸 영상 수이며 전체 입력 수가 아니다. 입력 수는 `len(videos)==5`로 검증한다.
   - 질문 순서로 로컬 질문 ID를 붙이고 `result.json`을 저장한다. 녹화가 처리 중 바뀌면 저장하지 않는다.
3. `POST /api/interviews/{session_id}/feedback`
   - **`session.llm_prompt`를 수정하지 않고 Gemini에 한 번 보낸다.** A/B 비교 프롬프트와 답변별 `videos[i].llm_prompt`는 호출하지 않는다.
   - 형식 오류는 한 번 재시도할 수 있다. 정상 응답 한 번을 받으면 질문별 추가 호출은 하지 않는다.
   - 전체 등급과 이유, 질문 5개의 `content_band`·`coaching_feedback`, 전달 피드백·근거 참조·자료 부족 상태를 검증한다.
   - `Q1`~`Q5`를 실제 로컬 질문 ID로 바꾼다. 검증·정규화된 결과를 **`feedback_v1`**에 넣어 반환한다.
   - `session_feedback.json`에 모델명·프롬프트·녹화 서명·분석 시각·질문 ID에 따른 캐시와 원본 Gemini 응답을 보존한다.
   - Gemini 실패 시 정상 모델 분석은 유지하며 502를 반환한다. 예전 답변별 평가나 가짜 결과로 대체하지 않는다.
4. `GET /api/interviews/{session_id}/result`
   - 마지막으로 저장된 모델 분석과 성공한 `feedback_v1`을 조회한다. 예전 분석 결과에 세션 프롬프트·녹화 서명이 없다면 새 분석이 필요하다.

## 근거 영상 연결

LLM은 `(Qk, segment_id)`만 선택한다. 영상 경로나 시간은 생성하지 않는다.

- 구간 ID는 영상 안에서만 유일하다. `Q1/S1`과 `Q2/S1`은 다르다.
- 존재하지 않는 구간, 관련 질문 밖의 구간, 유효하지 않은 시간은 제외하고 `session_feedback.json`의 `dropped`에 남긴다.
- 좋은 점의 근거로 `rule_flag=issue` 구간을 쓰지 않는다. 같은 구간이 좋은 점·개선점 양쪽에 있으면 개선점 우선이다.
- 마커가 있으면 첫 마커의 `start`~`start+dur`, 없으면 원래 문장의 `start`~`end`를 사용한다.
- 재생 시작은 `max(0, start-1)`이고 종료는 `min(근거 끝, 재생 시작+5초)`다. 원래 근거의 start/end는 유지하고 playback_start/end만 5초 미리보기로 제한한다. 짧은 구간이나 영상 끝에서는 5초보다 짧을 수 있다.
- 정상 근거는 `feedback_v1.evidence_segments`에 로컬 `question_id`, `segment_id`, `start/end`, `playback_start/end`, `label`, 영상 조회 URL로 제공한다.
- 프론트는 항목의 첫 유효 근거 플레이어를 바로 표시한다. 답변별 구간 목록이나 하이라이트는 표시하지 않는다.
- 새로 녹화하거나 분석이 변경된 동안 생성된 낡은 피드백은 저장하지 않는다.

## 실행

Python 3.11 가상환경을 사용한다. 환경 파일과 녹화·분석 결과는 Git에 올리지 않는다.

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --env-file .env
```

`.env` 설정:

- `GRADIO_API_URL`: 최신 Colab Gradio 주소. 다시 실행하면 주소가 바뀔 수 있다.
- `GRADIO_ANALYSIS_MODE=gradio`: 실제 분석 모드.
- `GRADIO_TIMEOUT_SECONDS=300`: 모델 요청 HTTP 대기 설정.
- `GEMINI_API_KEY`: Gemini 키. 코드·로그에 출력하지 않는다.
- `GEMINI_MODEL`: 사용할 Gemini 모델명. 기존 기본값은 `gemini-3.1-flash-lite`다.

다른 프로젝트가 8000번 포트를 사용하면 면접 백엔드를 8012번 등 별도 포트로 실행한다. 프론트 실행 전 `INTERVIEW_API_TARGET=http://127.0.0.1:8012`를 설정한다. 환경 파일이나 실행 코드를 바꿨으면 백엔드를 재시작한다.

## 검증과 예시 모드

```powershell
.venv\Scripts\python.exe -m unittest test_analysis test_gemini_feedback test_session_feedback -v
```

위 테스트는 가짜 파일과 모의 Gradio·Gemini 응답만 사용하며 실제 녹화를 전송하지 않는다. 정상 호출·캐시 재사용·잘못된 ID·시간·자료 부족·재녹화·처리 중 분석 변경·Gemini 실패 후 재시도를 확인한다.

`GRADIO_ANALYSIS_MODE=mock`은 명시적인 화면 개발용이다. 모델과 Gemini를 실행하지 않으며 `is_dummy=true`, `feedback_status=skipped`다. 실제 서비스 실패 시 mock으로 자동 전환하지 않는다. 프론트의 `/?preview=feedback`은 별도의 가상 평가 미리보기다.

세션 프롬프트·마커의 품질과 실제 녹화 시간 정렬은 실제 영상 검증 단계에서 확인해야 한다. 이번 백엔드 수정에서는 영상 재평가를 실행하지 않았다.

## 저장된 실제 결과의 로컬 영상 미리보기

`INTERVIEW_ENABLE_PREVIEW=1`로 백엔드를 실행하면 세션의 `feedback-preview.json`을 `GET /api/interviews/{session_id}/feedback-preview`로 조회할 수 있다. 기본값은 비활성화다. 녹화 서명이 달라지면 409를 반환하고 결과를 바꾸지 않는다. 개발 프론트의 `/?preview=recorded-feedback&session={session_id}`에서 실험 결과임을 표시하고 실제 영상의 5초 구간을 재생한다. Gradio·Gemini 호출이나 서비스 result.json 덮어쓰기는 하지 않는다.
