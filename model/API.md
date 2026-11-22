# Colab 모델 서버 API (Gradio)

Colab에서 `python serve_gradio.py --share`를 실행하면 나오는 `https://xxxx.gradio.live` 주소로 호출한다. 이 주소는 서버를 다시 실행할 때마다 바뀐다.
영상은 **질문 순서대로** 보내고, 결과도 같은 순서로 돌아온다. 질문 ID는 받지 않으며, 백엔드가 요청 순서와 질문 ID를 연결한다.
팀 데이터 전달 규약(`docs/feedback-data-contract.md`)에서 민서에게 요청한 항목의 답은 [7장](#7-데이터-규약-요청-항목에-대한-답)에 모았다.

## 1. 호출 방법

```python
from gradio_client import Client, handle_file
import json

client = Client("https://xxxx.gradio.live")
files = [handle_file(p) for p in video_paths]                         # 질문 순서대로 5개

seg = client.predict(files, "en", api_name="/stt_segments")           # STT + 문장 구간
texts = [r["text"] for r in seg]
result = client.predict(files, json.dumps(texts), json.dumps(questions), True,
                        json.dumps([r["segments"] for r in seg]), api_name="/analyze")
```

| API | 입력 (순서대로) | 출력 |
|---|---|---|
| `/stt` | 영상 N개, 언어(`"en"`) | 대본 문자열 N개 |
| `/stt_segments` | 영상 N개, 언어(`"en"`) | `[{"text", "segments"}]` N개 |
| `/analyze` | 영상 N개, `transcripts_json`, `questions_json`, `with_stats`, `segments_json` | 아래 3장 |

- `transcripts_json`: STT 대본 리스트를 `json.dumps`한 문자열. 비우면 서버가 STT부터 한다. 사용자는 대본을 수정하지 않으므로 STT 결과를 그대로 넘긴다.
- `questions_json`: 질문 원문 리스트(선택, `""`이면 없음). 주면 LLM이 질문 적합성을 판단할 수 있다.
- `with_stats`: `True`면 음성·영상 통계와 LLM 프롬프트를 만든다.
- `segments_json`: 영상별 구간 리스트의 리스트(선택, `""`이면 없음). **4개 인자로 부르는 기존 방식도 그대로 동작한다**(구간 정보만 없음).
- 요청은 한 번에 하나씩 처리한다. `videos[i].video`는 서버 임시 경로라 쓰지 말고 **순서(index)** 로 구분한다.

## 2. 화면 표시와 모델 출력의 관계

팀 결정에 따라 **숫자 점수와 백분위는 화면에 보이지 않는다.** 세션 전체 상/중/하는 LLM의 판단으로 제공한다.
모델 출력(`scores`, `session.delivery`)은 **내부 참고용**이며, LLM에 참고 신호로 들어간다.

모델을 검증한 결과(학습에 안 쓴 99명, 정답과의 Spearman):

| 사용 방식 | 결과 | 판단 |
|---|---|---|
| 사람 단위 (한 사람의 답변 평균) | 약 0.73. 3등분 상/중/하 일치 약 64%, 상↔하 역전 약 4% | 참고 신호로 사용 |
| 답변 하나씩 | 약 0.53 | 단독 판단에 쓰지 않음 |
| **같은 사람의 답변끼리 비교** | 0.15~0.18. 길이 영향을 빼면 약 0.07 | **쓰지 않음.** 사실상 가장 짧은 답변을 높게 봄 |
| 같은 사람 답변 순위 학습 (쌍 순위 손실 추가) | 같은 사람 쌍 정확도 약 59%(최종 모델)에서 개선 없음. 순위 손실 비중을 높일수록 val에서 하락(57.8%→55.3%). '짧은 답변이 낫다' 규칙은 약 55% | **쓰지 않음** |
| text 구간을 빼고 점수 변화 보기 (occlusion) | 문장을 빼면 답변이 짧아져 점수가 올라감 | **쓰지 않음** |
| audio·video 구간 occlusion | 검증하지 않음 | 쓰지 않음 |

- 학습 라벨은 평가자의 쌍비교로 만든 상대 점수다. 답변 길이와 반비례한다(단어 수와 Spearman -0.49). 모델도 이 경향을 따른다.
- 그래서 답변 비교, 대표 구간 선택, 좋은/개선 구간 판단에는 모델 점수를 쓰지 않는다. 규칙 마커와 LLM이 맡는다.

## 3. `/analyze` 응답

```json
{
  "session": {
    "n_videos": 4, "n_skipped": 1, "smoothing": 0.5,
    "scores":   {"overall_performance": {"raw": 0.12, "percentile": 58, "percentile_low": 21, "percentile_high": 90, "band": "중"}, "...": "6개 점수"},
    "delivery": {"n_videos": 4, "note": "...",
                 "scores": {"overall_performance": {"band": "중", "band_length_adjusted": "중", "raw_mean": 0.12}, "...": "6개 점수"}}
  },
  "videos": [
    {"video": "(임시 경로)", "transcript": "...", "scores": {"...": "영상별 점수(내부용)"},
     "stats": {"audio": {}, "text": {}, "video": {}},
     "segments": [{"id": "S1", "start": 0.0, "end": 6.2, "text": "...", "stats": {}, "markers": [], "rule_flag": "stable"}],
     "llm_prompt": "..."},
    {"video": "(임시 경로)", "transcript": "Sorry I don't understand this question", "scores": null,
     "skipped": true, "warning": "대본이 6단어로 너무 짧음(기준 13단어 이상)", "stats": {}, "llm_prompt": "..."}
  ],
  "explanation": "..."
}
```

| 필드 | 뜻 |
|---|---|
| `session.n_videos` / `n_skipped` | 점수를 낸 영상 수 / 너무 짧아 제외한 영상 수. **`n_videos`는 5보다 작을 수 있다.** 영상 개수 확인은 `videos` 길이로 한다. |
| `session.scores`, `videos[i].scores` | 모델의 영상·세션 점수(백분위 등). 내부용. |
| `session.delivery` | 점수를 낸 영상들의 모델 출력 평균을 RecruitView 참가자 331명의 사람별 평균과 비교한 상/중/하. LLM 참고 신호용. `band_length_adjusted`는 평균 답변 길이 영향을 뺀 실험용 값이다. 점수를 낸 영상이 없으면 `null`. |
| `videos[i].skipped`, `warning` | 짧은 답변 가드(4장)로 점수를 내지 않은 영상과 그 이유 |
| `videos[i].stats` | 음성·텍스트·영상 통계(사람이 읽을 수 있는 값) |
| `videos[i].stats_error` | 통계 일부 또는 전체 추출 실패 이유. 실패해도 나머지 통계와 `llm_prompt`는 만든다. |
| `videos[i].segments` | 구간별 측정값과 마커(5장). `segments_json`을 줬거나 서버가 STT를 했을 때만 있다. |
| `videos[i].llm_prompt` | 답변별 프롬프트(호환용, 6장) |
| `session.llm_prompt` | 세션 전체 프롬프트(권장, 6장) |

영상이 1개면 세션 보정 없이 단일 영상 결과가 그대로 나온다(`smoothing: null`).

## 4. 짧은 답변 가드

대본이 13단어 미만이거나 영상이 7초 미만이면 그 영상은 점수를 내지 않는다(`scores: null`, `skipped: true`, `warning`).
학습 데이터에서 13단어 미만 답변의 정답 평균 백분위가 85일 만큼 짧은 답변이 높게 평가돼, 모델이 이런 답변에 높은 점수를 주기 때문이다.
제외된 영상도 통계, 구간, `llm_prompt`(점수 없는 버전)는 만들어서 LLM이 내용을 평가할 수 있다. 5개가 모두 짧으면 `session.scores`와 `session.delivery`는 `null`이다.

## 5. 구간별 측정값과 마커

`/stt_segments`는 Whisper 문장을 구간으로 만들고(3초 미만 문장은 앞 문장에 합침), `/analyze`는 구간마다 측정값과 규칙 마커를 붙인다.

```json
{"id": "S2", "start": 4.0, "end": 8.0, "text": "...",
 "stats": {"words": 10, "fillers": 0, "wpm": 150.0,
           "face_detected_ratio": 0.26, "gaze_away_ratio": 0.0, "head_motion": 0.08,
           "silence_ratio": 0.05, "rms_mean": 0.03, "longest_pause_sec": 0.32},
 "markers": [{"type": "face_missing", "start": 5.13, "dur": 3.0, "detail": "얼굴이 3.0초간 화면에서 잡히지 않음"}],
 "rule_flag": "issue"}
```

| `markers.type` | 기준 (고정값) |
|---|---|
| `gaze_away` | 시선이 정면에서 크게 벗어난 상태가 1초 이상 연속 |
| `face_missing` | 얼굴이 1초 이상 연속으로 안 잡힘 |
| `long_pause` | 침묵 2초 이상 (직전 문장과의 공백 포함) |
| `pace_slow` / `pace_fast` | 구간 말 속도 분당 100단어 미만 / 180단어 초과 (3초 이상 구간만) |
| `fillers` | um, uh 등 군더더기 말 2회 이상 |

- `rule_flag`: `issue`(마커 있음), `stable`(음성·영상 측정을 했고 마커 없음), `unknown`(음성이나 영상 측정 실패).
- `stats`의 영상 값(`face_detected_ratio`, `gaze_away_ratio`, `head_motion`)과 음성 값(`silence_ratio`, `rms_mean`, `longest_pause_sec`)은 측정에 실패하면 빠진다. `wpm`은 3초 이상 구간에만 있다.
- 마커와 시간은 서버가 만든다. LLM은 시간이나 구간을 새로 만들지 않는다.

## 6. LLM(Gemini) 프롬프트와 응답

### 세션 프롬프트 (권장): `session.llm_prompt`
답변 전체를 **한 번에** 평가하는 프롬프트다. 백엔드는 이것을 **수정하지 않고** Gemini에 한 번 보낸다(JSON 모드 `response_mime_type="application/json"`, `json.loads`로 파싱).
답변은 영상 순서대로 `Q1`, `Q2`… 로 표시되며, 백엔드가 순서로 실제 질문 ID와 연결한다. 출력은 데이터 전달 규약 7장·14장 형식이며, `schema_version`, `session_id`는 백엔드가 붙인다.

```json
{
  "status": "ready | partial | insufficient_data",
  "overall_band": "상 | 중 | 하 | null",
  "band_reason": "...", "summary": "...",
  "answer_feedback": [{"question_id": "Q1", "status": "evaluated | insufficient_data",
                       "content_band": "상 | 중 | 하 | null",
                       "coaching_feedback": "좋았던 점 + 재구성·구체화 방법을 이어 쓴 한 문단",
                       "relevance": {"rating": "적절 | 부분적 | 부적절 | 평가불가", "reason": "..."},
                       "specificity": "...", "accuracy": {"status": "supported | contradicted | unverified", "reason": "..."},
                       "strength": "... 또는 null", "improvement": "... 또는 null"}],
  "delivery_feedback": {"summary": "...",
                        "strengths":    [{"modality": "audio | video", "feedback": "...", "suggestion": null, "scope": "single_answer | multiple_answers",
                                          "related_question_ids": ["Q1"], "evidence": [{"question_id": "Q1", "segment_id": "S2"}]}],
                        "improvements": [{"modality": "audio | video", "feedback": "...", "suggestion": "...", "scope": "...",
                                          "related_question_ids": ["Q2"], "evidence": [{"question_id": "Q2", "segment_id": "S3"}]}]},
  "limitations": ["..."]
}
```

- `content_band`는 그 답변의 텍스트 내용만 본 등급이다(음성·영상, 모델 인상은 반영하지 않음). 화면 본문은 `coaching_feedback`이고, `relevance`, `specificity`, `accuracy`, `strength`, `improvement`는 내부 분석·호환용이다.
- 음성·영상 좋은 점과 개선점은 각각 0~2개이고, 항목마다 **대표 장면 1개를 `evidence` 첫 번째**로 넣는다(최대 2개). 답변별 좋은·개선 구간은 만들지 않는다.
- 프롬프트에 들어가는 것: 답변별 질문, 구간 ID와 시간이 붙은 대본(구간이 없으면 전체 대본), 측정값과 참가자 대비 백분위, 점수 제외·측정 실패 상태, 구간별 마커 전체,
  모델의 사람 단위 전반적 인상(`overall_performance` 하나, "답변 비교·장면 선택·등급에 쓰지 말 것" 명시), 작성 규칙(세션·질문별 등급 기준, 장면 선택 기준, 평가자 데이터에서 확인된 경향).

### 대표 장면을 초 단위로 바꾸기
LLM은 시간을 쓰지 않고 `(question_id, segment_id)`만 고른다. 시간은 서버가 만든 구간과 마커로 붙인다. 참고 구현은 `model/code/session_prompt.py`의 `resolve_evidence(feedback, analyze_result)`이며, 백엔드는 같은 규칙으로 처리하면 된다.

1. `Qk`의 `segment_id`가 `videos[k-1].segments`에 없으면 버리고 기록한다(`dropped`).
2. 좋은 점(`strengths`)의 장면 중 마커가 걸린 구간(`rule_flag == "issue"`)은 쓰지 않는다.
3. 시간: 그 구간에 마커가 있으면 **첫 마커 시간**(`start` ~ `start + dur`), 없으면 **문장 구간 전체**(`start` ~ `end`).
4. 재생 시작: `max(0, start - 1초)`. 항목마다 남은 첫 장면이 대표 장면이다.

예: `{"question_id": "Q1", "segment_id": "S4"}`(개선점) → Q1 영상 12.1~15.0초("2.9초간 말이 멈춤"), 11.1초부터 재생.

### 답변별 프롬프트 (호환용): `videos[i].llm_prompt`
이전 방식이다. 답변 하나만 보고 `content_rating`, `overall_band`, 텍스트·음성·영상별 강점과 개선점, `evidence: [{segment_id, kind, area, note}]`를 돌려준다.
세션 프롬프트를 쓰면 호출하지 않는다(중복 호출 금지).

## 7. 데이터 규약 요청 항목에 대한 답

| 요청 항목 | 답 |
|---|---|
| `/stt_segments` 최상위 형식 | 영상 순서대로 된 리스트. 각 항목은 `{"text": 전체 대본, "segments": [...]}` |
| 구간 ID 자료형·유일성 | 문자열 `"S1"`, `"S2"`… (대문자 S). **영상 안에서만 유일**하므로 `question_id + segment_id`로 찾는다. |
| 구간 시간 | 각 영상 시작이 0인 초 단위, 소수 둘째 자리. Whisper 기준(소리 기준) 시간이다. |
| 일부 영상의 구간이 없을 때 | `segments_json`에서 그 영상 자리에 `[]`를 넣으면, 그 영상만 구간 없이 분석된다. |
| 측정값, `markers`, `rule_flag` | 5장 |
| Gemini `evidence` 모양 | 6장. 세션 프롬프트의 `delivery_feedback[].evidence`는 `{question_id, segment_id}` 형식이고 첫 번째가 대표 장면이다. `question_id`는 순서 기반 `Q1`… |
| 같은 구간이 strength와 improvement로 둘 다 인용되면 | 규약 제안대로 **개선 우선**에 동의 |
| 구간 후보가 여럿일 때 우선순위 | LLM이 항목마다 대표 장면을 `evidence` 첫 번째로 고른다. 개선점은 마커가 걸린 구간을 우선한다. 모델 점수로는 고르지 않는다(2장). |
| 상대 비교·occlusion 검증 | 2장. 같은 사람 답변 비교는 순위 학습으로도 개선되지 않았다. text occlusion은 쓰지 않는다. audio·video occlusion은 검증하지 않았다. |

## 8. 주의

- 마커 기준값은 연출 녹화로 아직 검증하지 않았다. RecruitView 영상 하나를 편집해 5~8초 화면을 검게, 12~15초 소리를 끈 테스트에서는 `face_missing`이 5.13초부터 3.0초, `long_pause`가 12.1초부터 2.9초로 잡혔다. 시선 이탈은 테스트하지 않았다.
- 구간 테스트는 고정 간격의 가짜 구간으로 했다. 실제 Whisper 구간은 Colab에서 확인이 필요하다.
- Whisper가 um, uh를 대본에서 지우는 경우가 많아 `fillers` 마커는 잘 안 잡힐 수 있다.
- 브라우저 녹화는 소리와 영상의 시작 시간이 조금 어긋날 수 있다. 구간 시간은 소리 기준, 시선·얼굴은 영상 프레임 기준이다.
- 결과는 연습용 참고 정보이며 합격 여부 판단에 쓰지 않는다.
