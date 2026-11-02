# AI_Interview

영어 면접 녹화, 답변 확인, STT 및 모델 평가를 연결하는 캡스톤 프로젝트입니다.

## 폴더

- `최종frontend/`: 현재 사용하는 React·TypeScript 프론트엔드
- `backend/`: 질문·영상 저장·STT·분석 API를 제공하는 FastAPI 서버
- `model/`: 모델 추론 코드와 Gradio API 안내
- `training/`: 모델 학습 코드
- `frontend/`: 이전 프론트엔드 코드 보관
- `ai/`: 로컬에서 보관하던 이전 특징 추출·LLM 관련 코드
- `colab/`: 이전 더미 API 노트북 보관. 실제 모델 실행은 `model/`의 안내를 사용합니다.

## 로컬 실행

Python 3.11과 Vite 8을 지원하는 Node.js(20.19 이상 또는 22.12 이상)가 필요합니다.

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# .env의 GRADIO_API_URL을 실행 중인 Colab 서버 주소로 설정하세요.
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --env-file .env
```

별도 터미널에서 프론트를 실행합니다.

```powershell
cd 최종frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5174
```

`http://127.0.0.1:5174/`에서 사용합니다. 개발 서버가 `/api` 요청을 백엔드의 8000번 포트로 전달합니다.

녹화 영상 5개를 저장한 뒤 STT 텍스트를 먼저 표시하고, 같은 텍스트를 재사용하여 모델 평가를 요청합니다.
결과 화면은 백분위를 표시합니다. `band`와 LLM 프롬프트는 API에 보존하며 LLM 호출은 아직 연결하지 않았습니다.
Gradio 주소는 Colab 서버를 다시 실행하면 바뀔 수 있습니다.

환경 파일, 녹화 영상, 데이터셋·특징 배열, 설치 폴더, 빌드 결과, 카메라가 포함될 수 있는 디자인 캡처는 업로드하지 않습니다.
세부 설정과 테스트는 각 프론트·백엔드 README를 참고하세요.
