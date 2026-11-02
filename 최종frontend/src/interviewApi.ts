export const interviewCategories = [
  { id: "introduction", category: "자기소개" },
  { id: "personal_values", category: "자기이해 · 직무 가치관" },
  { id: "teamwork_leadership", category: "협업 · 소통 · 리더십" },
  { id: "problem_solving", category: "문제 해결 · 성과 · 주도성" },
  { id: "adaptability_growth", category: "적응력 · 성장 · 업무 관리" },
] as const

export type InterviewQuestion = {
  id: string
  categoryId: string
  category: string
  question: string
  order: number
}

export type InterviewSession = {
  id: string
  questions: InterviewQuestion[]
}

export type SubmittedAnswer = { questionId: string; duration: number }

export const scoreMetrics = [
  { key: "interview_score", label: "면접 평가" },
  { key: "answer_score", label: "답변 평가" },
  { key: "speaking_skills", label: "말하기 능력" },
  { key: "confidence_score", label: "자신감" },
  { key: "facial_expression", label: "표정" },
  { key: "overall_performance", label: "종합 평가" },
] as const

export type ModelScore = { band: string | null; percentile: number | null }
export type ModelScores = Record<typeof scoreMetrics[number]["key"], ModelScore>
export type InterviewResult = {
  session_id: string
  is_dummy: boolean
  session: { scores: ModelScores }
  videos: { question_id: string; transcript: string; scores: ModelScores }[]
}

export type InterviewTranscripts = {
  session_id: string
  is_dummy: boolean
  answers: { question_id: string; transcript: string }[]
}

export async function transcribeInterview(session: InterviewSession, signal: AbortSignal): Promise<InterviewTranscripts> {
  const response = await fetch(`/api/interviews/${encodeURIComponent(session.id)}/transcripts`, { method: "POST", signal })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw new Error(isObject(body) && typeof body.detail === "string" ? body.detail
    : "답변 텍스트를 불러오지 못했어요. 잠시 후 다시 요청해 주세요.")
  if (!isObject(body) || body.status !== "ready" || body.session_id !== session.id
    || typeof body.is_dummy !== "boolean" || !Array.isArray(body.answers)
    || body.answers.length !== session.questions.length
    || !body.answers.every((answer: unknown, index: number) => isObject(answer)
      && answer.question_id === session.questions[index].id && typeof answer.transcript === "string")) {
    throw new Error("답변 텍스트와 제출한 질문이 맞지 않거나 누락됐어요. 다시 요청해 주세요.")
  }
  return body as InterviewTranscripts
}

function isScores(value: unknown): value is ModelScores {
  return isObject(value) && scoreMetrics.every(({ key }) => {
    const score = value[key]
    return isObject(score) && (score.band === null || typeof score.band === "string")
      && (score.percentile === null || (typeof score.percentile === "number"
        && Number.isFinite(score.percentile) && score.percentile >= 0 && score.percentile <= 100))
  })
}

export async function analyzeInterview(session: InterviewSession, signal: AbortSignal): Promise<InterviewResult> {
  const response = await fetch(`/api/interviews/${encodeURIComponent(session.id)}/analyze`, {
    method: "POST", signal,
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw new Error(isObject(body) && typeof body.detail === "string"
    ? body.detail : "답변을 분석하지 못했어요. 잠시 후 다시 시도해 주세요.")
  if (!isObject(body) || body.status !== "success" || body.session_id !== session.id
    || typeof body.is_dummy !== "boolean" || !isObject(body.session)
    || !isScores(body.session.scores) || !Array.isArray(body.videos)
    || body.videos.length !== session.questions.length
    || !body.videos.every((video: unknown, index: number) => isObject(video)
      && video.question_id === session.questions[index].id
      && typeof video.transcript === "string" && isScores(video.scores))) {
    throw new Error("분석 결과와 제출한 질문이 맞지 않거나 결과가 누락됐어요. 다시 시도해 주세요.")
  }
  return body as InterviewResult
}

export function answerVideoUrl(sessionId: string, questionId: string) {
  return `/api/interviews/${encodeURIComponent(sessionId)}/answers/${encodeURIComponent(questionId)}/video`
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export async function uploadAnswer(sessionId: string, questionId: string, blob: Blob, duration: number, signal: AbortSignal) {
  const form = new FormData()
  form.append("video", blob, "answer.webm")
  form.append("duration_seconds", String(duration))
  const response = await fetch(`/api/interviews/${encodeURIComponent(sessionId)}/answers/${encodeURIComponent(questionId)}`, {
    method: "POST", body: form, signal,
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw new Error(isObject(body) && typeof body.detail === "string" ? body.detail : "답변 영상을 저장하지 못했어요. 다시 제출해 주세요.")
  if (!isObject(body) || body.status !== "saved" || body.session_id !== sessionId || body.question_id !== questionId) {
    throw new Error("저장 완료를 확인하지 못했어요. 영상을 유지하고 있으니 다시 제출해 주세요.")
  }
}

export async function createInterview(signal: AbortSignal): Promise<InterviewSession> {
  const response = await fetch("/api/interviews", { method: "POST", signal })
  const body: unknown = await response.json().catch(() => null)

  if (!response.ok) {
    throw new Error(
      isObject(body) && typeof body.detail === "string"
        ? body.detail
        : "질문을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.",
    )
  }

  if (!isObject(body) || typeof body.session_id !== "string"
    || !/^[a-f0-9]{32}$/.test(body.session_id)
    || !Array.isArray(body.questions) || body.questions.length !== 5) {
    throw new Error("서버에서 받은 면접 정보가 올바르지 않습니다. 다시 시도해 주세요.")
  }

  const questions = body.questions.map((item: unknown, index: number): InterviewQuestion => {
    const category = interviewCategories[index]
    if (!isObject(item) || typeof item.id !== "string" || !item.id.trim()
      || typeof item.text !== "string" || !item.text.trim()
      || item.category !== category.id || item.order !== index + 1) {
      throw new Error("서버에서 받은 질문 정보가 올바르지 않습니다. 다시 시도해 주세요.")
    }
    return {
      id: item.id,
      categoryId: category.id,
      category: category.category,
      question: item.text,
      order: index + 1,
    }
  })

  if (new Set(questions.map((question) => question.id)).size !== questions.length) {
    throw new Error("질문 정보가 중복되었습니다. 다시 시도해 주세요.")
  }

  return { id: body.session_id, questions }
}
