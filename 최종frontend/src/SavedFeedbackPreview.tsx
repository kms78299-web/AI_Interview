import { useEffect, useState } from "react"
import { FeedbackResults, readFeedbackReport, type ResultSource } from "./FeedbackResults"
import type { InterviewSession } from "./interviewApi"

type Preview = { label: string; session: InterviewSession; result: ResultSource }
export default function SavedFeedbackPreview({ sessionId }: { sessionId: string }) {
  const [data, setData] = useState<Preview | null>(null)
  const [error, setError] = useState("")
  useEffect(() => {
    const request = new AbortController()
    setData(null); setError("")
    void fetch(`/api/interviews/${encodeURIComponent(sessionId)}/feedback-preview`, { signal: request.signal })
      .then(async response => {
        if (!response.ok) throw new Error("저장된 실험 결과를 불러오지 못했어요.")
        const body = await response.json() as Preview
        if (body.session?.id !== sessionId || body.result?.session_id !== sessionId
          || !Array.isArray(body.session.questions) || body.session.questions.length !== 5
          || !body.session.questions.every((q, i) => typeof q.id === "string" && typeof q.question === "string" && q.order === i + 1)
          || !Array.isArray(body.result.videos) || body.result.videos.length !== 5
          || !body.result.videos.every((v, i) => v.question_id === body.session.questions[i].id && typeof v.transcript === "string")
          || !readFeedbackReport(body.result.feedback_v1, body.session)) throw new Error("실험 결과와 녹화 질문이 맞지 않아요.")
        setData(body)
      }).catch(reason => { if (!request.signal.aborted) setError(reason instanceof Error ? reason.message : "결과를 불러오지 못했어요.") })
    return () => request.abort()
  }, [sessionId])
  if (!data) return <main className="feedback-results"><p role={error ? "alert" : "status"}>{error || "저장된 실제 평가 결과를 불러오고 있어요…"}</p></main>
  return <><div className="fr-recorded-notice" role="note">{data.label} · 실제 녹화 연결 · 서비스 결과에 반영되지 않은 실험입니다.</div><FeedbackResults result={data.result} session={data.session} /></>
}
