import { useEffect, useRef, useState } from "react"
import { answerVideoUrl, type InterviewSession } from "./interviewApi"
import "./feedbackResults.css"

type Ref = { question_id: string; segment_id: string | number }
type Clip = Ref & { start: number; end: number; playback_start: number; playback_end: number }
type Item = { modality: string; feedback: string; suggestion: string | null; evidence: Ref[] }
type Content = { question_id: string; status: string; content_band?: "상" | "중" | "하" | null; coaching_feedback?: string; summary?: string; revision_guide?: string; relevance: { rating: string; reason: string }; specificity: string; accuracy: { status: string; reason: string }; strength: string | null; improvement: string | null }
export type FeedbackReport = {
  schema_version: string; session_id: string; is_dummy?: boolean; status: string
  overall_band: "상" | "중" | "하" | null; band_reason: string; summary: string
  answer_feedback: Content[]
  delivery_feedback: { summary: string; strengths: Item[]; improvements: Item[] }
  limitations: string[]; evidence_segments: Clip[]
}
type LegacyVideo = { question_id: string; transcript: string; feedback?: { content_rating: string; content_reason: string; text: { strength: string; improvement: string } } | null }
export type ResultSource = { session_id: string; is_dummy: boolean; videos: LegacyVideo[]; feedback_v1?: unknown }
type Props = { result: ResultSource; session: InterviewSession; embedded?: boolean; feedbackLoading?: boolean; feedbackError?: string; onFeedbackRetry?: () => void }
const object = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && !Array.isArray(x)
const text = (x: unknown): x is string => typeof x === "string"
const nullableText = (x: unknown) => x === null || text(x)
const ref = (x: unknown): x is Ref => object(x) && text(x.question_id) && (text(x.segment_id) || (typeof x.segment_id === "number" && Number.isFinite(x.segment_id)))
const item = (x: unknown) => object(x) && text(x.modality) && text(x.feedback) && nullableText(x.suggestion) && Array.isArray(x.evidence) && x.evidence.every(ref)

export function readFeedbackReport(value: unknown, session: InterviewSession): FeedbackReport | null {
  if (!object(value) || value.schema_version !== "1.0" || value.session_id !== session.id
    || !["ready", "partial", "insufficient_data"].includes(String(value.status))
    || !["상", "중", "하", null].includes(value.overall_band as string | null)
    || (value.status === "ready" && value.overall_band === null)
    || (value.status === "insufficient_data" && value.overall_band !== null)
    || !text(value.summary) || !text(value.band_reason) || !Array.isArray(value.answer_feedback)
    || value.answer_feedback.length !== session.questions.length
    || !value.answer_feedback.every(a => object(a) && text(a.question_id) && text(a.status)
      && (a.content_band === undefined || ["상", "중", "하", null].includes(a.content_band as string | null))
      && (a.coaching_feedback === undefined || text(a.coaching_feedback))
      && (a.summary === undefined || text(a.summary)) && (a.revision_guide === undefined || text(a.revision_guide))
      && object(a.relevance) && text(a.relevance.rating) && text(a.relevance.reason)
      && text(a.specificity) && object(a.accuracy) && text(a.accuracy.status) && text(a.accuracy.reason)
      && nullableText(a.strength) && nullableText(a.improvement))
    || new Set(value.answer_feedback.map(a => a.question_id)).size !== session.questions.length
    || !session.questions.every(q => (value.answer_feedback as Content[]).some(a => a.question_id === q.id))
    || !object(value.delivery_feedback) || !text(value.delivery_feedback.summary)
    || !Array.isArray(value.delivery_feedback.strengths) || !value.delivery_feedback.strengths.every(item)
    || !Array.isArray(value.delivery_feedback.improvements) || !value.delivery_feedback.improvements.every(item)
    || !Array.isArray(value.limitations) || !value.limitations.every(text)
    || !Array.isArray(value.evidence_segments)) return null
  const report = value as unknown as FeedbackReport
  // Never connect unrecognised questions or invalid time ranges to a recording.
  return { ...report, evidence_segments: report.evidence_segments.filter(c => ref(c)
    && session.questions.some(q => q.id === c.question_id)
    && [c.start, c.end, c.playback_start, c.playback_end].every(n => typeof n === "number" && Number.isFinite(n))
    && c.start >= 0 && c.start < c.end && c.playback_start >= 0
    && c.playback_start <= c.start && c.playback_end > c.playback_start && c.playback_end <= c.end) }
}

export function getPreviewWindow(clip: Pick<Clip, "playback_start" | "playback_end">) {
  return { start: clip.playback_start, end: Math.min(clip.playback_end, clip.playback_start + 5) }
}

function RepresentativeVideo({ clip, session, demo, title }: { clip: Clip; session: InterviewSession; demo: boolean; title: string }) {
  const [error, setError] = useState("")
  const video = useRef<HTMLVideoElement>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const window = getPreviewWindow(clip)
  const clearStop = () => { if (timer.current !== null) { clearTimeout(timer.current); timer.current = null } }
  useEffect(() => () => { clearStop(); video.current?.pause() }, [clip.question_id, clip.segment_id, window.start, window.end])
  const stop = (element: HTMLVideoElement) => {
    clearStop(); element.pause()
    if (element.currentTime > window.end) element.currentTime = window.end
  }
  const scheduleStop = (element: HTMLVideoElement) => {
    clearStop()
    if (element.paused || element.seeking) return
    if (element.currentTime >= window.end) { stop(element); return }
    timer.current = setTimeout(() => {
      if (element.currentTime >= window.end - .04) stop(element)
      else scheduleStop(element)
    }, Math.max(20, (window.end - element.currentTime) / element.playbackRate * 1000))
  }
  const question = session.questions.find(q => q.id === clip.question_id)
  const start = (element: HTMLVideoElement) => {
    if (Number.isFinite(element.duration) && (window.start >= element.duration || window.end > element.duration + .1)) {
      setError("녹화 영상과 근거 구간의 시간이 맞지 않아요."); return
    }
    element.currentTime = window.start
    void element.play().catch(() => setError("재생 버튼을 눌러 영상을 확인해 주세요."))
  }
  return <div className="fr-evidence">
    <div className="fr-evidence-heading"><strong>대표 장면 · {(window.end - window.start).toFixed(0)}초</strong><small>{question?.order}번 답변 · {window.start.toFixed(1)}–{window.end.toFixed(1)}초</small></div>
    <div className="fr-player">
      {demo ? <div className="fr-player-demo"><span aria-hidden="true">▷</span><strong>대표 장면 미리보기</strong><p>예시 구간입니다. 실제 녹화 영상은 연결하지 않았어요.</p></div>
        : <video ref={video} key={`${clip.question_id}-${clip.segment_id}`} controls playsInline preload="metadata"
          aria-label={`${title} 대표 근거 영상`} src={answerVideoUrl(session.id, clip.question_id)}
          onLoadedMetadata={e => {
            if (Number.isFinite(e.currentTarget.duration) && window.end > e.currentTarget.duration + .1) setError("녹화 영상과 근거 구간의 시간이 맞지 않아요.")
            else e.currentTarget.currentTime = window.start
          }}
          onPlay={e => { document.querySelectorAll(".fr-player video").forEach(other => { if (other !== e.currentTarget) (other as HTMLVideoElement).pause() }); if (e.currentTarget.currentTime < window.start || e.currentTarget.currentTime >= window.end) e.currentTarget.currentTime = window.start }}
          onPlaying={e => scheduleStop(e.currentTarget)} onPause={clearStop} onEnded={clearStop} onSeeking={clearStop}
          onSeeked={e => { if (e.currentTarget.currentTime < window.start || e.currentTarget.currentTime > window.end) e.currentTarget.currentTime = window.start; else scheduleStop(e.currentTarget) }}
          onRateChange={e => scheduleStop(e.currentTarget)}
          onTimeUpdate={e => { if (e.currentTarget.currentTime >= window.end) stop(e.currentTarget) }}
          onError={() => setError("근거 영상을 불러오지 못했어요. 잠시 후 다시 열어 주세요.")} />}
      {error && <p role="alert">{error}<button type="button" onClick={() => { if (video.current) { setError(""); start(video.current) } }}>다시 재생하기</button></p>}
      <small>대표 장면 · {question?.question}</small>
      {!demo && <button type="button" className="fr-replay" onClick={() => { if (video.current) { setError(""); start(video.current) } }}>↻ {(window.end - window.start).toFixed(0)}초 다시 보기</button>}
    </div>
  </div>
}

export function FeedbackResults({ result, session, embedded = false, feedbackLoading = false, feedbackError = "", onFeedbackRetry }: Props) {
  const report = readFeedbackReport(result.feedback_v1, session)
  const demo = result.is_dummy || report?.is_dummy === true
  const pending = feedbackLoading ? "종합 피드백을 작성하고 있어요." : "세션 전체를 검토한 종합 피드백이 아직 준비되지 않았어요."
  const renderItems = (items: Item[], kind: "strength" | "improvement") => items.map((entry, index) => {
    const clip = entry.evidence.map(ref => report?.evidence_segments.find(c => c.question_id === ref.question_id && c.segment_id === ref.segment_id)).find((c): c is Clip => !!c)
    return <article className={`fr-observation fr-observation--${kind}`} key={`${kind}-${index}`}>
      <span className="fr-tag">{entry.modality === "video" ? "비디오" : entry.modality === "audio" ? "오디오" : "비디오 · 오디오"}</span>
      <p>{entry.feedback}</p>{entry.suggestion && <div className="fr-suggestion">{entry.suggestion}</div>}
      {clip && <RepresentativeVideo clip={clip} session={session} demo={demo} title={kind === "strength" ? "좋은 점" : "개선점"} />}
    </article>
  })
  const body = <>
    {!embedded && <header className="fr-header"><a href="/">English Interview</a><span>면접 피드백</span></header>}
    <div className="fr-title"><span className="fr-kicker">INTERVIEW FEEDBACK</span><h1>이번 면접을 돌아볼까요?</h1><p>전체 피드백부터 답변별 내용까지, 차근차근 살펴보세요.</p></div>
    {demo && <div className="fr-demo" role="note">화면 확인용 예시입니다. 아래 평가와 근거 구간은 실제 분석 결과가 아니에요.</div>}
    {feedbackError && <div className="fr-message" role="alert">{feedbackError}{onFeedbackRetry && <button type="button" onClick={onFeedbackRetry}>피드백 다시 요청하기</button>}</div>}
    {!report && result.feedback_v1 != null && <p className="fr-message" role="alert">종합 피드백 형식이 맞지 않아 표시하지 못했어요. 답변 텍스트는 아래에서 확인할 수 있어요.</p>}
    <section className="fr-card fr-overall" aria-labelledby="fr-overall-heading"><div className="fr-section-title"><span>01</span><div><small>OVERALL</small><h2 id="fr-overall-heading">전체 피드백</h2></div></div>
      <div className="fr-overall-content"><div className="fr-band"><small>전반적인 평가</small><strong>{report?.overall_band ?? (feedbackLoading ? "작성 중" : "평가 대기")}</strong></div><div><h3>{report?.band_reason || pending}</h3>{report?.summary && <p>{report.summary}</p>}{report && report.overall_band === null && <p>평가 자료가 부족해 상·중·하 판단을 유보했어요.</p>}</div></div>
    </section>
    <section className="fr-card" aria-labelledby="fr-delivery-heading"><div className="fr-section-title"><span>02</span><div><small>VIDEO & AUDIO</small><h2 id="fr-delivery-heading">비디오 · 오디오 피드백</h2></div></div>
      <p className="fr-section-description">{report?.delivery_feedback.summary || "다섯 답변의 말하기와 영상 표현을 함께 살펴보고, 특징이 잘 드러나는 대표 장면을 연결해요."}</p>
      <div className="fr-delivery-grid"><div><h3 className="fr-column-heading">좋은 점</h3>{report?.delivery_feedback.strengths.length ? renderItems(report.delivery_feedback.strengths, "strength") : <p className="fr-empty">{report ? "확인된 강점 피드백이 없어요." : pending}</p>}</div><div><h3 className="fr-column-heading">개선점</h3>{report?.delivery_feedback.improvements.length ? renderItems(report.delivery_feedback.improvements, "improvement") : <p className="fr-empty">{report ? "확인된 개선 피드백이 없어요." : pending}</p>}</div></div>
    </section>
    <section aria-labelledby="fr-content-heading"><div className="fr-section-title fr-section-title--outside"><span>03</span><div><small>ANSWER CONTENT</small><h2 id="fr-content-heading">답변별 텍스트 피드백</h2></div></div><p className="fr-section-description">질문에 맞게 답했는지 확인하고, 답변의 내용을 다듬어 보세요.</p>
      <div className="fr-answers">{session.questions.map(question => {
        const content = report?.answer_feedback.find(a => a.question_id === question.id)
        const video = result.videos.find(v => v.question_id === question.id)
        const legacy = video?.feedback
        const coaching = content?.coaching_feedback?.trim() || [
          content?.summary || content?.strength || legacy?.text.strength || legacy?.content_reason,
          content?.revision_guide || content?.improvement || content?.specificity || legacy?.text.improvement,
        ].filter((value, index, values) => value?.trim() && values.indexOf(value) === index).join(" ")
        return <article className="fr-card fr-answer" key={question.id}><div className="fr-question"><span>{String(question.order).padStart(2, "0")}</span><div><small>{question.category}</small><h3 lang="en">{question.question}</h3></div></div>
          <div className="fr-answer-overview"><div className="fr-answer-rating"><small>답변 내용 평가</small><span>{content?.content_band ?? (content?.status === "insufficient_data" ? "자료 부족" : "평가 대기")}</span></div><p className="fr-coaching-paragraph">{coaching || (feedbackLoading ? "답변 내용을 검토하고 있어요." : "내용 피드백이 아직 준비되지 않았어요.")}</p></div>
          {content?.accuracy.status === "contradicted" && <p className="fr-message">{content.accuracy.reason}</p>}
          <details className="fr-transcript"><summary>전체 답변 텍스트 확인 <span aria-hidden="true">⌄</span></summary><div><p lang={video?.transcript.trim() ? "en" : "ko"}>{video?.transcript.trim() || "인식된 답변 텍스트가 없어요."}</p><small>음성을 자동으로 옮긴 텍스트로 실제 발화와 다를 수 있어요.</small></div></details>
        </article>
      })}</div>
    </section>
    {!!report?.limitations.length && <aside className="fr-limitations"><h2>확인해 주세요</h2><ul>{report.limitations.map((line, i) => <li key={i}>{line}</li>)}</ul></aside>}
  </>
  return embedded ? <div className="feedback-results feedback-results--embedded">{body}</div> : <main className="feedback-results">{body}</main>
}
