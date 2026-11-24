import type { InterviewSession } from "./interviewApi"
import type { ResultSource } from "./FeedbackResults"

// Explicit development preview; never substituted for a failed real analysis.
export const previewSession: InterviewSession = {
  "id": "demo_session",
  "questions": [
    {
      "id": "introduction",
      "question": "Tell me about yourself.",
      "categoryId": "introduction",
      "category": "자기소개",
      "order": 1
    },
    {
      "id": "personal_03",
      "question": "What are your greatest strengths and weaknesses?",
      "categoryId": "personal_values",
      "category": "답변 내용",
      "order": 2
    },
    {
      "id": "teamwork_01",
      "question": "Describe a project where you took the lead in implementing a new strategy or process, driving positive change within your team or organization.",
      "categoryId": "teamwork_leadership",
      "category": "답변 내용",
      "order": 3
    },
    {
      "id": "personal_11",
      "question": "Where do you see yourself in the next 5 years?",
      "categoryId": "personal_values",
      "category": "답변 내용",
      "order": 4
    },
    {
      "id": "personal_12",
      "question": "Why should a company hire you?",
      "categoryId": "personal_values",
      "category": "답변 내용",
      "order": 5
    }
  ]
}
export const previewResult: ResultSource = {
  "session_id": "demo_session",
  "is_dummy": true,
  "videos": [
    {
      "question_id": "introduction",
      "transcript": "I study software engineering. I enjoy working with a team. In a recent project, I built an interview practice application with my teammates."
    },
    {
      "question_id": "personal_03",
      "transcript": "My strength is listening to other people. When our team had different ideas, I helped us agree on a plan. I am also working on explaining my own ideas more clearly."
    },
    {
      "question_id": "teamwork_01",
      "transcript": "We were working on a team project with a tight deadline. I suggested a new way to organise our tasks. We divided the work and checked our progress every day."
    },
    {
      "question_id": "personal_11",
      "transcript": "In five years, I hope to become a software engineer who builds useful products. I want to keep learning and work with people from different backgrounds."
    },
    {
      "question_id": "personal_12",
      "transcript": "I enjoy solving problems and learning from feedback. I would like to bring my teamwork skills and curiosity to this role."
    }
  ],
  "feedback_v1": {
    "schema_version": "1.0",
    "session_id": "demo_session",
    "is_dummy": true,
    "status": "ready",
    "overall_band": "중",
    "band_reason": "질문에는 대체로 답했지만 일부 경험의 행동과 결과를 구체적으로 설명하면 좋겠어요. 전달에서는 강점과 보완점이 함께 관찰됐어요.",
    "summary": "경험을 질문과 연결해 답하려는 점이 좋아요. 본인의 행동과 그 결과를 구체적으로 덧붙이면 답변을 이해하기 쉬워질 거예요.",
    "answer_feedback": [
      {
        "question_id": "introduction",
        "status": "evaluated",
        "relevance": {
          "rating": "적절",
          "reason": "전공과 관심 분야를 소개해 질문에 답했어요."
        },
        "specificity": "관심을 갖게 된 경험을 하나 덧붙이면 좋아요.",
        "accuracy": {
          "status": "unverified",
          "reason": "개인 경험의 실제 사실 여부는 대본만으로 확인하지 않았어요."
        },
        "strength": "전공과 관심 분야를 간결하게 소개했어요.",
        "improvement": "관심 분야와 관련된 프로젝트에서 본인이 한 일을 덧붙여 보세요.",
        "content_band": "중",
        "summary": "전공과 관심 분야를 간결하게 소개했어요. 다만 관심이 실제 경험으로 어떻게 이어졌는지 설명이 부족해, 본인의 특징이 충분히 드러나지 않았어요.",
        "revision_guide": "전공과 관심 분야 소개 뒤에 관련 프로젝트 하나를 덧붙여 보세요. 본인이 맡은 일과 배운 점을 한 문장씩 설명한 뒤 앞으로 하고 싶은 일로 마무리하면 좋아요.",
        "coaching_feedback": "전공과 관심 분야를 간결하게 소개했어요. 다만 관심이 실제 경험으로 어떻게 이어졌는지 설명이 부족해, 본인의 특징이 충분히 드러나지 않았어요. 전공과 관심 분야 소개 뒤에 관련 프로젝트 하나를 덧붙여 보세요. 본인이 맡은 일과 배운 점을 한 문장씩 설명한 뒤 앞으로 하고 싶은 일로 마무리하면 좋아요."
      },
      {
        "question_id": "personal_03",
        "status": "evaluated",
        "relevance": {
          "rating": "적절",
          "reason": "강점과 약점을 모두 설명했어요."
        },
        "specificity": "약점을 개선하기 위해 실제로 한 행동을 더 설명할 수 있어요.",
        "accuracy": {
          "status": "unverified",
          "reason": "제공된 대본 외에 사실 확인 자료가 없어요."
        },
        "strength": "강점을 경험과 연결했어요.",
        "improvement": "약점을 설명한 뒤 개선 행동과 변화를 덧붙여 보세요.",
        "content_band": "중",
        "summary": "강점과 약점을 모두 다뤘고 강점을 경험과 연결했어요. 약점을 개선하기 위한 행동과 변화까지 설명하면 더 설득력 있는 답변이 돼요.",
        "revision_guide": "강점은 구체적인 경험 하나로 뒷받침하고, 약점은 개선을 위해 실천한 행동과 그 이후 달라진 점을 이어서 설명해 보세요.",
        "coaching_feedback": "강점과 약점을 모두 다뤘고 강점을 경험과 연결했어요. 약점을 개선하기 위한 행동과 변화까지 설명하면 더 설득력 있는 답변이 돼요. 강점은 구체적인 경험 하나로 뒷받침하고, 약점은 개선을 위해 실천한 행동과 그 이후 달라진 점을 이어서 설명해 보세요."
      },
      {
        "question_id": "teamwork_01",
        "status": "evaluated",
        "relevance": {
          "rating": "부분적",
          "reason": "팀 프로젝트는 설명했지만 본인이 주도한 변화는 덜 분명해요."
        },
        "specificity": "본인의 결정과 변화의 결과를 구분해 설명하면 좋아요.",
        "accuracy": {
          "status": "unverified",
          "reason": "경험의 진위를 확인할 별도 자료가 없어요."
        },
        "strength": "협업 상황의 배경을 소개했어요.",
        "improvement": "상황, 본인의 행동, 결과 순서로 다시 정리해 보세요.",
        "content_band": "하",
        "summary": "팀 프로젝트의 배경은 설명했지만 본인이 주도한 변화와 결과가 분명하지 않아 질문에 충분히 답하지 못했어요.",
        "revision_guide": "어떤 문제가 있었는지, 본인이 무엇을 제안하고 실행했는지, 그 결과 무엇이 달라졌는지 순서대로 다시 구성해 보세요.",
        "coaching_feedback": "팀 프로젝트의 배경은 설명했지만 본인이 주도한 변화와 결과가 분명하지 않아 질문에 충분히 답하지 못했어요. 어떤 문제가 있었는지, 본인이 무엇을 제안하고 실행했는지, 그 결과 무엇이 달라졌는지 순서대로 다시 구성해 보세요."
      },
      {
        "question_id": "personal_11",
        "status": "evaluated",
        "relevance": {
          "rating": "적절",
          "reason": "5년 뒤의 목표를 제시했어요."
        },
        "specificity": "목표를 이루기 위한 중간 계획은 더 구체화할 수 있어요.",
        "accuracy": {
          "status": "unverified",
          "reason": "미래 계획이므로 사실 확인 대상과 구분했어요."
        },
        "strength": "희망하는 성장 방향이 분명해요.",
        "improvement": "앞으로 1년 동안 할 수 있는 구체적인 행동을 제시해 보세요.",
        "content_band": "중",
        "summary": "앞으로의 성장 방향은 분명해요. 다만 목표를 이루기 위한 구체적인 계획이 더해지면 실현 가능한 목표로 전달될 거예요.",
        "revision_guide": "5년 뒤의 목표를 말한 뒤 앞으로 1년 동안 배우거나 실천할 행동 두 가지를 덧붙여 보세요.",
        "coaching_feedback": "앞으로의 성장 방향은 분명해요. 다만 목표를 이루기 위한 구체적인 계획이 더해지면 실현 가능한 목표로 전달될 거예요. 5년 뒤의 목표를 말한 뒤 앞으로 1년 동안 배우거나 실천할 행동 두 가지를 덧붙여 보세요."
      },
      {
        "question_id": "personal_12",
        "status": "evaluated",
        "relevance": {
          "rating": "부분적",
          "reason": "자신의 장점은 설명했지만 회사에 기여하는 방식은 더 연결할 수 있어요."
        },
        "specificity": "장점을 보여주는 성과나 사례가 추가되면 좋아요.",
        "accuracy": {
          "status": "unverified",
          "reason": "개인 성과를 확인할 자료가 제공되지 않았어요."
        },
        "strength": "본인의 역량을 설명하려고 했어요.",
        "improvement": "역량, 경험, 지원 직무에 기여할 방법을 연결해 보세요.",
        "content_band": "하",
        "summary": "본인의 역량을 소개했지만 실제 사례와 지원 직무에 기여할 방법이 부족해 채용해야 하는 이유가 충분히 드러나지 않았어요.",
        "revision_guide": "직무에 필요한 역량 하나를 고르고, 이를 보여준 실제 경험과 입사 후 기여할 방법을 차례로 연결해 보세요.",
        "coaching_feedback": "본인의 역량을 소개했지만 실제 사례와 지원 직무에 기여할 방법이 부족해 채용해야 하는 이유가 충분히 드러나지 않았어요. 직무에 필요한 역량 하나를 고르고, 이를 보여준 실제 경험과 입사 후 기여할 방법을 차례로 연결해 보세요."
      }
    ],
    "delivery_feedback": {
      "summary": "자기소개에서 정면을 향해 말하는 모습이 관찰됐어요. 강점·약점 답변에서는 긴 쉼이 있는 구간을 다시 살펴볼 수 있어요. 이 장면들을 다른 답변 전체의 특징으로 일반화하지는 않았어요.",
      "strengths": [
        {
          "modality": "video",
          "feedback": "자기소개 구간에서 정면을 향해 말하는 모습이 관찰됐어요.",
          "suggestion": null,
          "scope": "single_answer",
          "related_question_ids": [
            "introduction"
          ],
          "evidence": [
            {
              "question_id": "introduction",
              "segment_id": "s1"
            }
          ]
        }
      ],
      "improvements": [
        {
          "modality": "audio",
          "feedback": "강점과 약점을 설명하는 답변 중 긴 쉼이 있는 구간이 관찰됐어요.",
          "suggestion": "핵심 단어를 먼저 정리한 뒤 짧은 문장으로 이어 말하는 연습을 해보세요.",
          "scope": "single_answer",
          "related_question_ids": [
            "personal_03"
          ],
          "evidence": [
            {
              "question_id": "personal_03",
              "segment_id": "s2"
            }
          ]
        }
      ]
    },
    "limitations": [
      "화면 개발을 위한 가상 예시이며 실제 분석·관찰 결과가 아닙니다."
    ],
    "evidence_segments": [
      {
        "question_id": "introduction",
        "segment_id": "s1",
        "start": 0.8,
        "end": 3.2,
        "playback_start": 0.0,
        "playback_end": 3.2,
        "label": "strength",
        "video_url": "/api/interviews/demo_session/answers/introduction/video"
      },
      {
        "question_id": "personal_03",
        "segment_id": "s2",
        "start": 18.0,
        "end": 21.0,
        "playback_start": 17.0,
        "playback_end": 21.0,
        "label": "improvement",
        "video_url": "/api/interviews/demo_session/answers/personal_03/video"
      }
    ]
  }
}
