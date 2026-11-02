# Colab 모델 서버 API (Gradio)

Colab에서 `python serve_gradio.py --share`를 실행하면 나오는 `https://xxxx.gradio.live` 주소로 호출한다. 영상은 **질문 순서대로** 보내고, 결과도 같은 순서로 돌아온다. 질문 ID 연결은 백엔드가 한다(모델 쪽은 순서만 본다).

```python
from gradio_client import Client, handle_file
import json
client = Client("https://xxxx.gradio.live")
files = [handle_file(p) for p in video_paths]                              # 질문 순서대로 5개
texts = client.predict(files, "en", api_name="/stt")                       # ① STT만 -> ["1번 답변", ..., "5번 답변"]
result = client.predict(files, json.dumps(texts), "", True, api_name="/analyze")   # ② 점수 + 통계 + 프롬프트
```

| API | 입력 | 출력 |
|---|---|---|
| `/stt` | 영상 N개, 언어("en") | 텍스트 N개 (JSON 리스트, 같은 순서) |
| `/analyze` | 영상 N개, transcripts_json(위 텍스트를 `json.dumps`한 문자열, 비우면 STT부터), questions_json(질문 텍스트 리스트, 선택, ""=없음), with_stats(True/False) | 아래 응답 |

사용자는 STT 텍스트를 수정하지 않으므로 ①의 결과를 그대로 ②에 넣는다. 텍스트 개수가 영상 개수와 다르면 에러가 난다.

## `/analyze` 응답
- `session`: 영상 전체를 같은 사람으로 보고 합친 점수(6개). 영상이 1개면 단일 영상 결과 그대로.
- `videos[i]`: i번째 영상(질문 순서). `scores`(점수 6개), `transcript`, `stats`(해석용 통계), `llm_prompt`(LLM에 그대로 보낼 프롬프트).
- `explanation`: 점수 해석 문구(절대 점수가 아니라 집단 대비 상대 위치라는 설명 포함).
- 점수 항목: `percentile`(0~100 기대 백분위, 50이 평균), `percentile_low`~`percentile_high`(80% 범위), `band`(상/중/하), `raw`(모델 원 출력).
- 한 영상의 통계 추출이 실패해도(예: 얼굴이 안 잡힘) 전체는 정상 반환하고, 그 영상만 `stats: null`, `llm_prompt: null`, `stats_error`에 이유가 들어간다.
- **너무 짧은 답변은 점수를 내지 않는다.** 대본이 13단어 미만이거나 영상이 7초 미만이면 해당 영상은 `scores: null`, `skipped: true`, `warning`(이유)이 들어간다. 이 모델은 학습 데이터의 경향상 아주 짧은 답변에 점수를 높게 주기 때문이다(13단어 미만 답변의 정답 평균 백분위가 85). 이런 영상은 `session` 점수 계산에서도 빠지고(`session.n_skipped`에 개수), 전부 짧으면 `session.scores`도 `null`이다. 이 경우에도 `llm_prompt`는 점수 없는 버전으로 들어 있어 LLM이 내용을 평가할 수 있다. UI는 `scores == null`이면 점수 대신 `warning`을 보여 주면 된다.
- `videos[i].video`는 서버 임시 경로라 쓰지 말고 **순서(index)** 로 영상을 구분한다.

```json
{
  "session": {
    "n_videos": 5,
    "smoothing": 0.5,
    "scores": {
      "interview_score": {
        "raw": 0.35,
        "percentile": 64,
        "percentile_low": 26,
        "percentile_high": 94,
        "band": "상"
      },
      "answer_score": {
        "raw": 0.1,
        "percentile": 55,
        "percentile_low": 18,
        "percentile_high": 89,
        "band": "중"
      },
      "speaking_skills": {
        "raw": 0.2,
        "percentile": 56,
        "percentile_low": 19,
        "percentile_high": 91,
        "band": "중"
      },
      "confidence_score": {
        "raw": 0.15,
        "percentile": 56,
        "percentile_low": 19,
        "percentile_high": 90,
        "band": "중"
      },
      "facial_expression": {
        "raw": 0.05,
        "percentile": 52,
        "percentile_low": 14,
        "percentile_high": 88,
        "band": "중"
      },
      "overall_performance": {
        "raw": 0.3,
        "percentile": 62,
        "percentile_low": 25,
        "percentile_high": 93,
        "band": "상"
      }
    }
  },
  "videos": [
    {
      "video": "(임시 경로: 순서(index)로 구분하세요)",
      "transcript": "STT 텍스트",
      "scores": {
        "interview_score": {
          "raw": 0.35,
          "percentile": 64,
          "percentile_low": 26,
          "percentile_high": 94,
          "band": "상",
          "smoothed_raw": 0.35
        },
        "answer_score": {
          "raw": 0.1,
          "percentile": 55,
          "percentile_low": 18,
          "percentile_high": 89,
          "band": "중",
          "smoothed_raw": 0.1
        },
        "speaking_skills": {
          "raw": 0.2,
          "percentile": 56,
          "percentile_low": 19,
          "percentile_high": 91,
          "band": "중",
          "smoothed_raw": 0.2
        },
        "confidence_score": {
          "raw": 0.15,
          "percentile": 56,
          "percentile_low": 19,
          "percentile_high": 90,
          "band": "중",
          "smoothed_raw": 0.15
        },
        "facial_expression": {
          "raw": 0.05,
          "percentile": 52,
          "percentile_low": 14,
          "percentile_high": 88,
          "band": "중",
          "smoothed_raw": 0.05
        },
        "overall_performance": {
          "raw": 0.3,
          "percentile": 62,
          "percentile_low": 25,
          "percentile_high": 93,
          "band": "상",
          "smoothed_raw": 0.3
        }
      },
      "single_clip_percentile": {
        "interview_score": 50,
        "answer_score": 50,
        "speaking_skills": 50,
        "confidence_score": 50,
        "facial_expression": 50,
        "overall_performance": 50
      },
      "stats": {
        "audio": {
          "pitch_mean": 207.373,
          "pitch_std": 52.233,
          "rms_mean": 0.04,
          "rms_std": 0.03,
          "silence_ratio": 0.136,
          "speech_rate": 5.774,
          "duration_sec": 43.5,
          "speaking_rate_wpm": 120.1
        },
        "text": {
          "transcript": "Hello everyone, this is ... (STT 텍스트)",
          "filler_words": [],
          "filler_count": 0,
          "repeated_words": [
            "also",
            "interest",
            "exploring",
            "with",
            "an"
          ],
          "answer_length": 87,
          "vocab_diversity": 0.74
        },
        "video": {
          "face_detected_ratio": 1.0,
          "gaze_off_center_mean": 0.085,
          "gaze_away_ratio": 0.05,
          "gaze_x_std": 0.025,
          "gaze_y_std": 0.187,
          "head_yaw_mean_abs": 0.043,
          "head_pitch_mean": 0.305,
          "head_roll_std": 0.051,
          "head_motion": 0.097,
          "blink_rate_per_min": 19.266,
          "mouth_open_std": 0.072,
          "mouth_width_std": 0.035,
          "n_frames_used": 218.0
        }
      },
      "llm_prompt": "당신은 면접 연습 피드백을 쓰는 코치입니다. 아래는 면접 답변 영상에서 자동으로 측정한 값입니다.\n질문: Introduce yourself\n\n[답변 대본]\nHello everyone, this is ... (STT 텍스트)\n\n[측정값]\n- [audio] 말 속도(분당 단어 수): 120.1 (참가자 26번째 백분위, 100에 가까울수록 값이 큰 편)\n- [audio] 침묵 비율: 0.136 (참가자 23번째 백분위, 100에 가까울수록 값이 큰 편)\n- [audio] 평균 음높이(Hz): 207.373 (참가자 81번째 백분위, 1 ... (이하 생략)"
    },
    "... 나머지 4개 ..."
  ],
  "explanation": "점수는 100점 만점의 절대 실력이 아니라 ... (설명문)"
}
```

## LLM 호출은 백엔드가 한다
`videos[i].llm_prompt`를 LLM에 그대로 보내면 된다. 프롬프트에는 "측정값을 점수의 원인으로 단정하지 말 것" 같은 규칙과, 답변 내용이 질문에 맞는지를 먼저 평가하게 하는 규칙(질문을 이해 못 했다고 답한 경우 등)이 이미 들어 있다. 질문 텍스트를 `questions_json`으로 주면 내용 평가가 더 정확해진다. 점수는 참가자 집단 대비 상대 위치이며 정확도가 중간 수준이라, 합격/불합격 판단 같은 용도로 쓰지 않는다.
