import { useEffect, useLayoutEffect, useRef, useState, type ReactNode, type CSSProperties } from "react"
import { createInterview, uploadAnswer, answerVideoUrl, analyzeInterview, transcribeInterview, type InterviewTranscripts, scoreMetrics, interviewCategories, type InterviewResult, type ModelScores, type InterviewQuestion, type InterviewSession, type SubmittedAnswer } from "./interviewApi"
import { createAnswerRecorder, type Recording } from "./answerRecorder"
import { useInterviewDevices, type InterviewDevices } from "./useInterviewDevices"
import { CameraPreview, MicrophoneLevel, RecordedPreview } from "./DevicePreview"
import { createQuestionNarration, type NarrationState } from "./questionNarration"

type Screen = "home" | "guide" | "interview" | "review" | "analysis" | "results"
type RecordingState = "ready" | "recording" | "processing" | "review"
type IconName = "arrow" | "camera" | "check" | "chevron" | "clock" | "exit" | "mic" | "pause" | "play" | "refresh" | "shield" | "spark" | "volume" | "volumeOff"

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    volume: (
      <>
        <path d="M11 5 6 9H3v6h3l5 4Z" />
        <path d="M15 9a4 4 0 0 1 0 6" />
        <path d="M17.5 6.5a8 8 0 0 1 0 11" />
      </>
    ),
    volumeOff: (
      <>
        <path d="M11 5 6 9H3v6h3l5 4Z" />
        <path d="m16 10 5 5" />
        <path d="m21 10-5 5" />
      </>
    ),
    arrow: (
      <>
        <path d="M5 12h14" />
        <path d="m13 6 6 6-6 6" />
      </>
    ),
    camera: (
      <>
        <path d="M15 10 20 7v10l-5-3" />
        <rect x="3" y="5" width="12" height="14" rx="2" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    chevron: <path d="m9 18 6-6-6-6" />,
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    exit: (
      <>
        <path d="M10 5H5v14h5" />
        <path d="M13 8l4 4-4 4" />
        <path d="M8 12h9" />
      </>
    ),
    mic: (
      <>
        <rect x="9" y="3" width="6" height="12" rx="3" />
        <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
      </>
    ),
    pause: (
      <>
        <path d="M9 8v8" />
        <path d="M15 8v8" />
      </>
    ),
    play: <path d="m9 7 8 5-8 5Z" />,
    refresh: (
      <>
        <path d="M20 11a8 8 0 1 0-2 5" />
        <path d="M20 5v6h-6" />
      </>
    ),
    shield: (
      <>
        <path d="M12 3 5 6v5c0 5 3 8 7 10 4-2 7-5 7-10V6Z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
    spark: (
      <>
        <path d="m12 3 1.4 4.1L17 9l-3.6 1.9L12 15l-1.4-4.1L7 9l3.6-1.9Z" />
        <path d="m19 15 .7 2.1L22 18l-2.3.9L19 21l-.7-2.1L16 18l2.3-.9Z" />
      </>
    ),
  }

  return (
    <svg
      aria-hidden="true"
      className="icon"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      {paths[name]}
    </svg>
  )
}

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand ${compact ? "brand--compact" : ""}`}>
      <span className="brand__mark" aria-hidden="true">
        E
      </span>
      <span>English Interview</span>
    </div>
  )
}

function Button({
  children,
  className = "",
  disabled = false,
  onClick,
  variant = "primary",
}: {
  children: ReactNode
  className?: string
  disabled?: boolean
  onClick?: () => void
  variant?: "primary" | "secondary" | "text" | "danger"
}) {
  return (
    <button
      className={`button button--${variant} ${className}`}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      {children}
    </button>
  )
}

function SimulationBadge({
  children = "시뮬레이션",
}: {
  children?: ReactNode
}) {
  return (
    <span className="simulation-badge">
      <span aria-hidden="true" />
      {children}
    </span>
  )
}

function Home({ onStart }: { onStart: () => void }) {
  return (
    <main className="home page-shell">
      <header className="home__header">
        <Brand />
        <div className="home__header-note">AI 영어 면접 연습</div>
      </header>

      <section className="home__hero">
        <div className="home__copy">
          <div className="eyebrow">ENGLISH INTERVIEW COACH</div>
          <h1>
            한 번의 연습이,
            <br />
            다음 면접의 자신감으로.
          </h1>
          <p>
            실제 면접처럼 영어로 답하고, 질문별 피드백으로
            <br className="desktop-only" /> 나만의 답변을 차분하게 다듬어
            보세요.
          </p>
          <Button className="home__cta" onClick={onStart}>
            면접 시작하기
            <Icon name="arrow" />
          </Button>
          <div className="home__meta">
            <span>
              <Icon name="clock" size={16} />약 7분
            </span>
            <span>5개 질문</span>
            <span>무료 데모</span>
          </div>
        </div>

        <div className="preview-wrap" aria-label="면접 화면 미리보기">
          <div className="preview-top">
            <div>
              <span className="preview-label">QUESTION 01</span>
              <strong>Tell me about yourself.</strong>
            </div>
            <span className="preview-progress">1 / 5</span>
          </div>
          <div className="preview-grid">
            <div className="preview-video">
              <SimulationBadge>미리보기 데모</SimulationBadge>
              <div className="person-placeholder">
                <span className="person-placeholder__head" />
                <span className="person-placeholder__body" />
              </div>
              <div className="preview-video__footer">
                <span className="record-dot" />
                답변 준비
                <span>01:00</span>
              </div>
            </div>
            <div className="preview-list">
              {interviewCategories.map((item, index) => (
                <div
                  className={`preview-step ${
                    index === 0 ? "preview-step--active" : ""
                  }`}
                  key={item.category}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <small>{item.category}</small>
                    <p>{index === 0 ? "현재 질문" : "대기"}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="preview-controls">
            <span>
              <Icon name="mic" size={15} />
              마이크 준비됨
            </span>
            <span className="preview-controls__button">
              <span /> 녹화 시작
            </span>
          </div>
        </div>
      </section>

      <section className="home__features" aria-label="서비스 특징">
        <div>
          <span>01</span>
          <strong>실전형 영어 질문</strong>
          <p>면접에서 자주 묻는 5개 핵심 질문으로 연습해요.</p>
        </div>
        <div>
          <span>02</span>
          <strong>부담 없는 반복 연습</strong>
          <p>질문별 2번의 기회로 답변을 한 번 더 다듬어요.</p>
        </div>
        <div>
          <span>03</span>
          <strong>항목별 코칭 리포트</strong>
          <p>답변과 말하기 습관을 구분해 다음 연습을 안내해요.</p>
        </div>
      </section>
    </main>
  )
}

function Guide({
  onBack,
  onStart,
  loading,
  error,
  devices,
}: {
  onBack: () => void
  onStart: () => void
  loading: boolean
  error: string
  devices: InterviewDevices
}) {
  const [consented, setConsented] = useState(false)
  const cameraChecked = devices.cameraReady
  const micChecked = devices.microphoneReady

  return (
    <main className="inner-page page-shell">
      <header className="app-header">
        <Brand compact />
        <Button onClick={onBack} variant="text">
          처음으로
        </Button>
      </header>

      <div className="guide-heading">
        <span className="step-label">INTERVIEW GUIDE</span>
        <h1>면접을 시작하기 전에 확인해 주세요</h1>
        <p>답변에만 집중할 수 있도록 진행 방식과 장치를 미리 점검해요.</p>
      </div>

      <div className="guide-layout">
        <section className="card guide-rules">
          <div className="section-heading">
            <div>
              <span className="section-kicker">진행 안내</span>
              <h2>총 5개의 영어 질문이 주어집니다</h2>
            </div>
            <span className="time-pill">
              <Icon name="clock" size={16} /> 약 7분
            </span>
          </div>

          <div className="rule-summary">
            <div>
              <strong>5</strong>
              <span>전체 질문</span>
            </div>
            <div>
              <strong>
                60<small>초</small>
              </strong>
              <span>질문당 최대</span>
            </div>
            <div>
              <strong>
                2<small>회</small>
              </strong>
              <span>질문별 녹화</span>
            </div>
          </div>

          <div className="category-list">
            {interviewCategories.map((item, index) => (
              <div className="category-row" key={item.category}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <p>{item.category}</p>
                <small>영어 질문 1개</small>
              </div>
            ))}
          </div>

          <div className="notice-box">
            <Icon name="shield" size={20} />
            <p>
              질문은 서버에서 선택되며 면접 중 카테고리별로 순서대로 공개됩니다. 답변은 제출 전
              다시 확인할 수 있어요. 질문 옆 질문 읽기 버튼을 누르면 영어 질문이 음성으로 재생되고,
              읽기가 끝나면 화면 중앙의 5초 카운트다운 후 자동으로 녹화가 시작됩니다.
              카운트다운은 취소할 수 있고, ‘녹화 시작’을 눌러 직접 시작할 수도 있어요.
              녹화가 시작되면 영상·음성이 녹화되고,
              ‘이 답변 제출하기’를 누르면 서버에 저장됩니다.
            </p>
          </div>
        </section>

        <aside className="card device-card">
          <div className="section-heading">
            <div>
              <span className="section-kicker">장치 점검</span>
              <h2>카메라와 마이크</h2>
            </div>
            <span className="device-connection-label">{devices.ready ? "연결 완료" : devices.status === "connecting" ? "연결 중" : "연결 필요"}</span>
          </div>

          <div className="device-preview">
            {devices.stream ? <CameraPreview stream={devices.stream} /> : <div className="device-preview__person">
              <span />
              <span />
            </div>}
            <span className="device-preview__label">{devices.stream ? "실시간 카메라 미리보기" : "카메라 연결 전"}</span>
          </div>

          <div className="device-checks">
            <button
              className={`device-check ${
                cameraChecked ? "device-check--done" : ""
              }`}
              disabled={loading || devices.status === "connecting"}
              onClick={devices.connect}
              type="button"
            >
              <span className="device-check__icon">
                <Icon name={cameraChecked ? "check" : "camera"} size={18} />
              </span>
              <span>
                <strong>카메라 확인</strong>
                <small>
                  {cameraChecked ? "실시간 영상 연결됨" : devices.status === "connecting" ? "권한 확인 중…" : "클릭하여 두 장치 연결"}
                </small>
              </span>
              <Icon name="chevron" size={16} />
            </button>
            <button
              className={`device-check ${
                micChecked ? "device-check--done" : ""
              }`}
              disabled={loading || devices.status === "connecting"}
              onClick={devices.connect}
              type="button"
            >
              <span className="device-check__icon">
                <Icon name={micChecked ? "check" : "mic"} size={18} />
              </span>
              <span>
                <strong>마이크 확인</strong>
                <small>
                  {micChecked ? "마이크 연결됨 · 아래 음량 확인" : devices.status === "connecting" ? "권한 확인 중…" : "클릭하여 두 장치 연결"}
                </small>
              </span>
              <Icon name="chevron" size={16} />
            </button>
          </div>

          <MicrophoneLevel stream={devices.stream} />
          {devices.message && <p className={`device-message${devices.status === "error" ? " device-message--error" : ""}`} role={devices.status === "error" ? "alert" : "status"}>{devices.message}</p>}

          <label className="consent">
            <input
              checked={consented}
              disabled={loading}
              onChange={(event) => setConsented(event.target.checked)}
              type="checkbox"
            />
            <span aria-hidden="true">
              <Icon name="check" size={14} />
            </span>
            <p>
              카메라·마이크를 이용한 면접 진행에 동의합니다.
              <small>녹화한 영상·음성을 답변 제출 시 서버에 저장하는 데 동의합니다.</small>
            </p>
          </label>

          <Button
            className="device-card__start"
            disabled={!consented || !devices.ready || loading}
            onClick={onStart}
          >
            {loading ? "질문 불러오는 중…" : error ? "다시 시도하기" : "면접 시작하기"}
            <Icon name="arrow" />
          </Button>
          {loading && <p className="disabled-help" role="status">면접 질문을 준비하고 있어요.</p>}
          {error && <p className="interview-request-error" role="alert">{error}</p>}
          {(!consented || !devices.ready) && (
            <p className="disabled-help">카메라·마이크 연결과 동의 후 시작할 수 있어요.</p>
          )}
        </aside>
      </div>
    </main>
  )
}

export function Interview({
  sessionId,
  questions,
  devices,
  onExit,
  onFinish,
}: {
  sessionId: string
  questions: InterviewQuestion[]
  devices: InterviewDevices
  onExit: () => void
  onFinish: (answers: SubmittedAnswer[]) => void
}) {
  const [current, setCurrent] = useState(0)
  const [recordingState, setRecordingState] = useState<RecordingState>("ready")
  const [seconds, setSeconds] = useState(0)
  const [attempts, setAttempts] = useState(0)
  const [answer, setAnswer] = useState<Recording | null>(null)
  const [cameraRatio, setCameraRatio] = useState(() => {
    const settings = devices.stream?.getVideoTracks()[0]?.getSettings()
    return settings?.width && settings?.height ? settings.width / settings.height : 16 / 9
  })
  const updateCameraRatio = (width: number, height: number) => {
    if (width > 0 && height > 0) setCameraRatio(width / height)
  }
  const [previewUrl, setPreviewUrl] = useState("")
  const [error, setError] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const recorderRef = useRef<ReturnType<typeof createAnswerRecorder> | null>(null)
  const submittedRef = useRef<SubmittedAnswer[]>([])
  const uploadRef = useRef<AbortController | null>(null)
  const startedRef = useRef(0)
  const questionTitleRef = useRef<HTMLHeadingElement>(null)
  const [narration, setNarration] = useState<NarrationState>({ phase: "idle", seconds: 0 })
  const narratorRef = useRef<ReturnType<typeof createQuestionNarration> | null>(null)
  const startRecordingRef = useRef<() => void>(() => {})
  const speechSupported = typeof window !== "undefined" && "speechSynthesis" in window && "SpeechSynthesisUtterance" in window

  useEffect(() => {
    if (!speechSupported || !devices.ready) return
    const narrator = createQuestionNarration({
      synthesis: window.speechSynthesis,
      makeUtterance: text => new SpeechSynthesisUtterance(text),
      schedule: (callback, delay) => window.setTimeout(callback, delay),
      unschedule: timer => window.clearTimeout(timer),
      onState: setNarration,
      onReady: () => startRecordingRef.current(),
      onError: setError,
    })
    narratorRef.current = narrator
    const hide = () => { if (document.hidden) narrator.cancel() }
    const leave = () => narrator.cancel()
    document.addEventListener("visibilitychange", hide)
    window.addEventListener("pagehide", leave)
    return () => {
      narratorRef.current = null
      narrator.dispose()
      document.removeEventListener("visibilitychange", hide)
      window.removeEventListener("pagehide", leave)
    }
  }, [current, devices.ready, devices.stream, speechSupported])

  useEffect(() => {
    if (!answer) { setPreviewUrl(""); return }
    const url = URL.createObjectURL(answer.blob)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [answer])

  useEffect(() => {
    const leave = () => {
      recorderRef.current?.dispose()
      const request = uploadRef.current
      uploadRef.current = null
      request?.abort()
    }
    window.addEventListener("pagehide", leave)
    return () => { window.removeEventListener("pagehide", leave); leave() }
  }, [])

  useEffect(() => {
    if (!devices.ready && recordingState === "recording") {
      recorderRef.current?.dispose()
      recorderRef.current = null
      setRecordingState("ready")
      setAttempts((value) => Math.max(0, value - 1))
      setError("장치 연결이 끊겨 녹화를 중단했어요. 다시 연결해 주세요.")
    }
  }, [devices.ready, recordingState])

  useLayoutEffect(() => {
    const title = questionTitleRef.current
    if (!title) return
    const fitQuestion = () => {
      title.style.fontSize = ""
      let size = parseFloat(window.getComputedStyle(title).fontSize)
      while (size > 12 && (title.scrollHeight > title.clientHeight + 1 || title.scrollWidth > title.clientWidth + 1)) {
        size -= 0.5
        title.style.fontSize = size + "px"
      }
    }
    fitQuestion()
    const observer = new ResizeObserver(fitQuestion)
    observer.observe(title)
    return () => observer.disconnect()
  }, [current, questions])

  useEffect(() => {
    if (recordingState !== "recording") return
    const timer = window.setInterval(() => {
      setSeconds(Math.min(60, Math.floor((performance.now() - startedRef.current) / 1000)))
    }, 200)
    return () => window.clearInterval(timer)
  }, [recordingState])

  const startRecording = () => {
    if (!devices.ready || !devices.stream || recorderRef.current || attempts >= 2 || recordingState !== "ready" || submitting) return
    narratorRef.current?.cancel()
    setError("")
    try {
      const recorder = createAnswerRecorder(devices.stream, {
        complete: (recorded) => {
          recorderRef.current = null
          setAnswer(recorded)
          setSeconds(Math.round(recorded.duration))
          setRecordingState("review")
        },
        error: (message) => {
          recorderRef.current = null
          setError(message)
          setRecordingState("ready")
          setAttempts((value) => Math.max(0, value - 1))
        },
      })
      recorderRef.current = recorder
      recorder.start()
      startedRef.current = performance.now()
      setSeconds(0)
      setAttempts((value) => value + 1)
      setRecordingState("recording")
    } catch (cause) {
      recorderRef.current = null
      setError(cause instanceof Error ? cause.message : "녹화를 시작하지 못했어요.")
    }
  }

  startRecordingRef.current = startRecording

  const readQuestion = () => {
    if (narration.phase !== "idle") { narratorRef.current?.cancel(); return }
    if (recordingState !== "ready" || !devices.ready || attempts >= 2 || submitting) return
    setError("")
    narratorRef.current?.start(questions[current].question)
  }

  const stopRecording = () => {
    if (!recorderRef.current) return
    setRecordingState("processing")
    recorderRef.current.stop()
  }

  const rerecord = () => {
    if (attempts < 2) {
      setAnswer(null)
      setError("")
      setSeconds(0)
      setRecordingState("ready")
    }
  }

  const submit = async () => {
    if (!answer || uploadRef.current) return
    const request = new AbortController()
    uploadRef.current = request
    setSubmitting(true)
    setError("")
    const timeout = window.setTimeout(() => request.abort(), 120_000)
    try {
      await uploadAnswer(sessionId, questions[current].id, answer.blob, answer.duration, request.signal)
      if (uploadRef.current !== request) return
      const completed = { questionId: questions[current].id, duration: answer.duration }
      submittedRef.current = [...submittedRef.current.filter((entry) => entry.questionId !== completed.questionId), completed]
      if (current === questions.length - 1) { onFinish(submittedRef.current); return }
      setCurrent((value) => value + 1)
      setAnswer(null)
      setRecordingState("ready")
      setSeconds(0)
      setAttempts(0)
    } catch (cause) {
      if (uploadRef.current !== request) return
      setError(request.signal.aborted ? "저장 응답 시간이 초과됐어요. 영상은 유지되니 다시 제출해 주세요." : cause instanceof TypeError ? "서버에 연결할 수 없어요. 연결을 확인한 뒤 다시 제출해 주세요." : cause instanceof Error ? cause.message : "저장하지 못했어요. 다시 제출해 주세요.")
    } finally {
      window.clearTimeout(timeout)
      if (uploadRef.current === request) { uploadRef.current = null; setSubmitting(false) }
    }
  }

  const time = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`

  return (
    <main className="interview-page">
      <header className="interview-header">
        <Brand compact />
        <div
          className="interview-progress"
          aria-label={`${current + 1}/${questions.length} 진행`}
        >
          <div className="interview-progress__labels">
            <span>면접 진행</span>
            <strong>{current + 1} / {questions.length}</strong>
          </div>
          <div className="progress-track">
            <span style={{ width: `${((current + 1) / questions.length) * 100}%` }} />
          </div>
        </div>
        <Button onClick={onExit} disabled={submitting} variant="text">
          <Icon name="exit" size={17} />
          나가기
        </Button>
      </header>

      <div className="interview-layout">
        <section className="interview-main">
          <div className="question-panel">
            <div className="question-panel__meta">
              <span>QUESTION {String(current + 1).padStart(2, "0")}</span>
              <small>{questions[current].category}</small>
            </div>
            <h1 ref={questionTitleRef}><span>{questions[current].question}</span>{" "}
              <button type="button" className="question-read-button"
                onClick={readQuestion}
                disabled={!speechSupported || !devices.ready || recordingState !== "ready" || attempts >= 2 || submitting}
                aria-label={narration.phase === "reading" ? "질문 읽기 중지" : narration.phase === "countdown" ? "자동 녹화 취소" : "질문 듣고 5초 후 녹화 시작"}
                title={speechSupported ? "질문 듣기 · 끝나면 5초 후 자동 녹화" : "이 브라우저는 질문 읽기를 지원하지 않아요"}>
                <Icon name={narration.phase === "idle" ? "volume" : "pause"} size={17} />
              </button>
            </h1>
            <p>{speechSupported ? "질문을 들으면 5초 후 자동 녹화됩니다. 직접 녹화를 시작해도 좋아요." : "질문 읽기를 지원하지 않는 브라우저예요. 질문을 읽고 직접 녹화를 시작해 주세요."}</p>
          </div>

          <div className="video-card" style={{ "--camera-ratio": cameraRatio } as CSSProperties}>
            <div className={`video-stage${devices.stream ? " video-stage--live" : ""}`}>
              {recordingState === "review" && previewUrl
                ? <RecordedPreview key={previewUrl} url={previewUrl} duration={answer?.duration ?? 0} onDimensions={updateCameraRatio} />
                : devices.stream && <CameraPreview stream={devices.stream} onDimensions={updateCameraRatio} />}
              <SimulationBadge>{recordingState === "review" ? "녹화된 답변" : recordingState === "processing" ? "영상 준비 중" : recordingState === "recording" ? "영상·음성 녹화 중" : devices.cameraReady ? "실시간 미리보기" : "카메라 연결 필요"}</SimulationBadge>
              {!devices.stream && <div className="video-stage__guide">
                <span className="video-stage__head" />
                <span className="video-stage__body" />
              </div>}
              {recordingState !== "review" && <div className="video-stage__tip">
                <Icon name="camera" size={15} />
                시선을 화면 중앙에 맞춰 보세요
              </div>}
              {narration.phase !== "idle" && <div className="question-narration-overlay">
                <div className="question-narration-panel" role="status" aria-live="polite" aria-atomic="true">
                  {narration.phase === "reading" ? <>
                    <Icon name="volume" size={28} />
                    <strong>질문을 듣고 있어요</strong>
                    <p>질문이 끝나면 5초 후 녹화가 시작돼요.</p>
                  </> : <>
                    <strong>잠시 후 답변 녹화가 시작됩니다</strong>
                    <span className="question-countdown">{narration.seconds}</span>
                    <p>카메라를 바라보고 영어로 답변해 주세요.</p>
                  </>}
                </div>
                <button type="button" onClick={() => narratorRef.current?.cancel()}>
                  {narration.phase === "reading" ? "질문 읽기 중지" : "자동 녹화 취소"}
                </button>
              </div>}
            </div>

            <div className="recording-tools">
              <div className="record-status">
                {recordingState === "recording" ? (
                  <>
                    <span className="record-dot record-dot--live" />
                    <div>
                      <strong>녹화 중</strong>
                      <small>최대 60초</small>
                    </div>
                  </>
                ) : recordingState === "review" ? (
                  <>
                    <span className="status-icon status-icon--complete">
                      <Icon name="check" size={15} />
                    </span>
                    <div>
                      <strong>녹화 완료</strong>
                      <small>{attempts}번째 답변</small>
                    </div>
                  </>
                ) : (
                  <>
                    <span className="status-icon">
                      <Icon name="mic" size={16} />
                    </span>
                    <div>
                      <strong>{recordingState === "processing" ? "영상 준비 중…" : "답변 준비"}</strong>
                      <small>{recordingState === "processing" ? "잠시 기다려 주세요" : "준비되면 녹화를 시작하세요"}</small>
                    </div>
                  </>
                )}
              </div>

              <div className="timer" aria-label={`${seconds}초 경과`}>
                <strong>{time}</strong>
                <span>/ 01:00</span>
              </div>

              <div className="record-actions">
                {recordingState === "ready" && (
                  <Button onClick={startRecording} disabled={!devices.ready || attempts >= 2}>
                    <span className="button-record-dot" />
                    녹화 시작
                  </Button>
                )}
                {recordingState === "recording" && (
                  <Button
                    onClick={stopRecording}
                    variant="danger"
                  >
                    <Icon name="pause" size={17} />
                    녹화 마치기
                  </Button>
                )}
                {recordingState === "review" && (
                  <>
                    <Button
                      disabled={attempts >= 2 || submitting}
                      onClick={rerecord}
                      variant="secondary"
                    >
                      <Icon name="refresh" size={16} />
                      다시 녹화
                    </Button>
                    <Button onClick={submit} disabled={submitting || !answer}>
                      {submitting ? "답변 저장 중…" : "이 답변 제출하기"}
                      <Icon name="arrow" size={17} />
                    </Button>
                  </>
                )}
              </div>
            </div>
            {error && <p className="interview-feedback interview-feedback--error" role="alert">{error}</p>}
            {submitting && <p className="interview-feedback" role="status">답변 영상을 저장하고 있어요. 저장이 끝나면 자동으로 넘어갑니다.</p>}
          </div>

          <MicrophoneLevel stream={devices.stream} />
          {!devices.ready && <div className="device-reconnect">
            <p className="device-message device-message--error" role="alert">{devices.message || "카메라와 마이크를 연결해 주세요."}</p>
            <Button onClick={devices.connect} disabled={devices.status === "connecting"} variant="secondary">{devices.status === "connecting" ? "장치 연결 중…" : "카메라·마이크 다시 연결"}</Button>
          </div>}

          <p className="attempt-note">
            질문당 최대 60초 · 녹화 2회. 녹화가 끝나면 영상과 소리를 확인하고 제출해 주세요.
          </p>
        </section>

        <aside className="question-list-card">
          <div className="question-list-card__heading">
            <span className="section-kicker">INTERVIEW FLOW</span>
            <h2>질문 목록</h2>
          </div>
          <div className="interview-question-list">
            {questions.map((item, index) => {
              const status =
                index < current
                  ? "done"
                  : index === current
                    ? "current"
                    : "waiting"
              return (
                <div
                  className={`interview-step interview-step--${status}`}
                  key={item.id}
                >
                  <div className="interview-step__number">
                    {status === "done" ? (
                      <Icon name="check" size={15} />
                    ) : (
                      String(index + 1).padStart(2, "0")
                    )}
                  </div>
                  <div>
                    <span>{item.category}</span>
                    {status === "done" && <strong>{item.question}</strong>}
                  </div>
                  <small>
                    {status === "done"
                      ? "완료"
                      : status === "current"
                        ? "현재"
                        : "대기"}
                  </small>
                </div>
              )
            })}
          </div>
          <div className="question-list-card__footer">
            <Icon name="shield" size={18} />
            질문은 순서대로 공개됩니다. 제출을 마친 질문만 목록에서 다시 확인할
            수 있어요.
          </div>
        </aside>
      </div>
    </main>
  )
}

export function AnswerReview({ session, answers, onAnalyze }: { session: InterviewSession; answers: SubmittedAnswer[]; onAnalyze?: () => void }) {
  const [selected, setSelected] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [progress, setProgress] = useState(0)
  const [videoError, setVideoError] = useState("")
  const [loaded, setLoaded] = useState(false)
  const [ratio, setRatio] = useState(16 / 9)
  const videoRef = useRef<HTMLVideoElement>(null)
  const titleRef = useRef<HTMLHeadingElement>(null)
  const questions = session.questions
  const item = questions[selected]
  const duration = answers.find((answer) => answer.questionId === item.id)?.duration ?? 0
  const videoUrl = answerVideoUrl(session.id, item.id)
  const formatTime = (value: number) => `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(Math.floor(value % 60)).padStart(2, "0")}`

  useEffect(() => {
    const video = videoRef.current
    return () => { video?.pause() }
  }, [selected])

  useLayoutEffect(() => {
    const title = titleRef.current
    if (!title) return
    const fit = () => {
      title.style.fontSize = ""
      let size = parseFloat(getComputedStyle(title).fontSize)
      while (size > 12 && title.scrollHeight > title.clientHeight + 1) {
        size -= 0.5
        title.style.fontSize = `${size}px`
      }
    }
    fit()
    const observer = new ResizeObserver(fit)
    observer.observe(title)
    return () => observer.disconnect()
  }, [selected])

  const selectQuestion = (index: number) => {
    if (index === selected) return
    videoRef.current?.pause()
    setSelected(index)
    setIsPlaying(false)
    setProgress(0)
    setVideoError("")
    setLoaded(false)
  }

  const togglePlayback = () => {
    const video = videoRef.current
    if (!video) return
    if (!video.paused) video.pause()
    else {
      if (video.ended) video.currentTime = 0
      void video.play().catch(() => setVideoError("영상을 재생하지 못했어요. 다시 불러오기를 눌러 주세요."))
    }
  }

  return (
    <main className="review-page page-shell">
      <header className="app-header">
        <Brand compact />
        <SimulationBadge>답변 확인</SimulationBadge>
      </header>

      <div className="review-heading">
        <span className="step-label">ANSWER REVIEW</span>
        <h1>제출한 답변을 확인해 보세요</h1>
        <p>5개 질문과 답변 영상을 확인한 뒤 분석을 시작해 주세요.</p>
      </div>

      <div className="review-layout">
        <section className="review-viewer card">
          <div className="review-question">
            <div>
              <span>QUESTION {String(selected + 1).padStart(2, "0")}</span>
              <small>{questions[selected].category}</small>
            </div>
            <h2 ref={titleRef}>{questions[selected].question}</h2>
          </div>

          <div className="review-player">
            <div className="review-player__stage" style={{ aspectRatio: ratio }}>
              <video key={item.id} ref={videoRef} src={videoUrl} className="review-saved-video"
                playsInline preload="metadata" muted={isMuted} aria-label={`${selected + 1}번 저장된 답변 영상`}
                onLoadedMetadata={(event) => {
                  const video = event.currentTarget
                  if (video.videoWidth && video.videoHeight) setRatio(video.videoWidth / video.videoHeight)
                  setLoaded(true)
                }}
                onPlay={() => { setIsPlaying(true); setVideoError("") }} onPause={() => setIsPlaying(false)} onEnded={() => setIsPlaying(false)}
                onTimeUpdate={(event) => setProgress(event.currentTarget.currentTime)}
                onError={() => { setLoaded(false); setVideoError("저장된 영상을 불러오지 못했어요. 서버 연결을 확인한 뒤 다시 불러와 주세요.") }} />
              {!loaded && !videoError && <span className="review-video-message" role="status">영상을 불러오고 있어요…</span>}
              {loaded && !isPlaying && !videoError && (
                <button
                  aria-label={`${selected + 1}번 답변 영상 재생`}
                  className="review-player__center-control"
                  onClick={togglePlayback}
                  type="button"
                >
                  <Icon name="play" size={24} />
                </button>
              )}
            </div>

            <div className="review-player__controls">
              <button
                aria-label={isPlaying ? "영상 일시정지" : "영상 재생"}
                onClick={togglePlayback}
                type="button"
              >
                <Icon name={isPlaying ? "pause" : "play"} size={18} />
              </button>
              <input
                aria-label="영상 재생 위치"
                max={Math.max(duration, 0.1)} step="0.01"
                min="0"
                onChange={(event) => {
                  const position = Number(event.target.value)
                  if (videoRef.current) videoRef.current.currentTime = position
                  setProgress(position)
                }}
                style={
                  {
                    "--video-progress": `${duration ? Math.min(100, progress / duration * 100) : 0}%`,
                  } as React.CSSProperties
                }
                type="range"
                value={Math.min(progress, Math.max(duration, 0.1))}
              />
              <span aria-label="영상 길이">
                {formatTime(progress)} / {formatTime(duration)}
              </span>
              <button
                aria-label={isMuted ? "영상 소리 켜기" : "영상 소리 끄기"}
                onClick={() => setIsMuted((value) => !value)}
                type="button"
              >
                <Icon name={isMuted ? "volumeOff" : "volume"} size={18} />
              </button>
            </div>
          </div>

          <div className="review-demo-note">
            {videoError ? <><p role="alert">{videoError}</p><button type="button" onClick={() => {
              setVideoError(""); setLoaded(false); videoRef.current?.load()
            }}>다시 불러오기</button></> : "제출한 답변 영상을 확인할 수 있어요."}
          </div>
        </section>

        <aside className="review-list card" aria-label="제출한 답변 목록">
          <div className="review-list__heading">
            <span>제출한 답변</span>
            <strong>{answers.length} / {questions.length}</strong>
          </div>
          <div className="review-list__items">
            {questions.map((item, index) => (
              <button
                aria-current={selected === index ? "true" : undefined}
                className={`review-list-item ${
                  selected === index ? "review-list-item--selected" : ""
                }`}
                key={item.id}
                onClick={() => selectQuestion(index)}
                type="button"
              >
                <span className="review-list-item__number">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className="review-list-item__content">
                  <small>{item.category}</small>
                  <strong>{item.question}</strong>
                  <span>
                    <Icon name="check" size={12} />
                    제출 완료
                  </span>
                </span>
                <Icon name="chevron" size={16} />
              </button>
            ))}
          </div>
        </aside>
      </div>

      <div className="review-footer">
        <div>
          <span className="review-footer__check">
            <Icon name="check" size={16} />
          </span>
          <p>
            <strong>답변 5개 제출 완료</strong>
            <small>모든 질문과 영상을 확인했다면 분석을 시작해 주세요.</small>
          </p>
        </div>
        <Button disabled={!onAnalyze} onClick={onAnalyze}>
          분석 시작하기
          <Icon name="arrow" />
        </Button>
      </div>
    </main>
  )
}

export function Analysis({ error, onRetry, onBack, session, transcripts }: {
  error: string; onRetry: () => void; onBack: () => void
  session: InterviewSession; transcripts: InterviewTranscripts | null
}) {
  const [selected, setSelected] = useState(0)
  const item = session.questions[selected]
  const text = transcripts?.answers.find(answer => answer.question_id === item.id)?.transcript ?? ""
  return (
    <main className="analysis-page analysis-wait-page page-shell">
      <header className="app-header"><Brand compact /><SimulationBadge>답변 분석</SimulationBadge></header>
      <section className="analysis-wait-heading">
        <span className="step-label">INTERVIEW ANALYSIS</span>
        <h1>{error ? "분석을 완료하지 못했어요" : transcripts ? "답변을 확인하는 동안 평가를 진행할게요" : "답변을 텍스트로 변환하고 있어요"}</h1>
        <p role="status">{error ? "저장된 답변 영상은 유지돼요. 잠시 후 다시 요청해 주세요." : transcripts ? "질문별 답변 내용을 살펴보세요. 평가가 끝나면 결과 화면으로 이동해요." : "5개 답변의 음성을 차례로 정리하고 있어요. 준비되면 이곳에서 확인할 수 있어요."}</p>
      </section>
      <ol className="analysis-stage-list" aria-label="분석 진행 단계">
        <li className={transcripts ? "is-complete" : "is-current"} aria-current={!transcripts ? "step" : undefined}><Icon name={transcripts ? "check" : "mic"} size={17} /><span>답변 텍스트 변환</span></li>
        <li className={transcripts ? "is-current" : ""} aria-current={transcripts ? "step" : undefined}><Icon name="spark" size={17} /><span>면접 평가</span></li>
        <li><Icon name="check" size={17} /><span>결과 확인</span></li>
      </ol>
      {transcripts?.is_dummy && <div className="demo-notice"><Icon name="spark" /><p>화면 확인용 테스트 답변입니다. 실제 STT 결과가 아니에요.</p></div>}
      {error && <div className="analysis-error card"><p role="alert">{error}</p><div className="analysis-actions"><Button onClick={onRetry}>분석 다시 요청하기</Button><Button onClick={onBack} variant="secondary">답변 확인으로 돌아가기</Button></div></div>}
      {transcripts ? <div className="analysis-transcript-layout">
        <section className="card analysis-transcript-viewer" aria-label="질문별 답변 텍스트">
          <span className="section-kicker">QUESTION {String(selected + 1).padStart(2, "0")} · {item.category}</span>
          <h2>{item.question}</h2>
          <section className="answer-transcript"><h3>내 답변 텍스트</h3><p lang={text.trim() ? "en" : "ko"}>{text.trim() ? text : "인식된 답변 텍스트가 없어요."}</p><small>음성을 자동으로 옮긴 텍스트로, 실제 발화와 다를 수 있어요.</small></section>
          {!error && <div className="analysis-ongoing" role="status"><span className="analysis-spinner" aria-hidden="true" />답변의 영상·음성·내용을 평가하고 있어요.</div>}
        </section>
        <aside className="card analysis-transcript-list" aria-label="확인할 답변 선택">
          <h2>제출한 답변 <small>5 / 5</small></h2>
          {session.questions.map((question, index) => <button key={question.id} type="button" aria-current={selected === index ? "true" : undefined} onClick={() => setSelected(index)}><span>{String(index + 1).padStart(2, "0")}</span><div><small>{question.category}</small><strong>{question.question}</strong></div><Icon name="chevron" size={16} /></button>)}
        </aside>
      </div> : !error && <section className="card analysis-stt-wait" role="status"><span className="analysis-spinner" aria-hidden="true" /><h2>답변 내용을 정리하고 있어요</h2><p>분석에는 몇 분이 걸릴 수 있어요.<br />텍스트가 준비되면 질문별로 보여드릴게요.</p></section>}
    </main>
  )
}

function PercentileBadge({ value }: { value: number | null }) {
  return <span className="percentile-badge"><small>백분위</small><strong>{value === null ? "정보 없음" : value}</strong></span>
}

function PercentileMetrics({ scores }: { scores: ModelScores }) {
  return <dl className="band-metrics">
    {scoreMetrics.map(({ key, label }) => <div key={key}><dt>{label}</dt><dd><PercentileBadge value={scores[key].percentile} /></dd></div>)}
  </dl>
}

export function Results({ result, session }: { result: InterviewResult; session: InterviewSession }) {
  const overall = result.session.scores.overall_performance
  return (
    <main className="results-page page-shell">
      <header className="app-header"><Brand compact /><SimulationBadge>{result.is_dummy ? "테스트 결과" : "면접 결과"}</SimulationBadge></header>
      <section className="result-hero">
        <div>
          <span className="step-label">INTERVIEW REPORT</span>
          <h1>면접 결과를 확인해 보세요</h1>
          <p>전체 평가와 질문별 답변 내용을 한곳에서 살펴볼 수 있어요.</p>
        </div>
      </section>
      {result.is_dummy && <div className="demo-notice" role="note"><Icon name="spark" size={18} /><p><strong>화면 확인용 테스트 결과입니다.</strong>아래 평가와 답변 텍스트는 실제 분석 결과가 아니에요.</p></div>}
      <section className="card result-summary">
        <div className="section-heading result-summary__heading"><div><span className="section-kicker">전체 요약</span><h2>면접 전체 평가</h2></div></div>
        <div className="score-layout">
          <div className="overall-band">
            <span>종합 평가</span>
            <div className="overall-percentile"><span>백분위</span><strong>{overall.percentile === null ? "정보 없음" : overall.percentile}</strong></div>
          </div>
          <PercentileMetrics scores={result.session.scores} />
        </div>
        <p className="band-note">백분위는 기준 집단 안에서의 상대적인 위치이며 100점 만점 점수가 아니에요.</p>
      </section>
      <section className="detail-section">
        <div className="detail-section__heading"><div><span className="section-kicker">질문별 세부 결과</span><h2>답변을 하나씩 살펴보세요</h2></div><span>각 항목을 눌러 펼치기</span></div>
        <div className="result-accordions">
          {session.questions.map((question, index) => {
            const video = result.videos.find((item) => item.question_id === question.id)!
            return <details className="result-detail card" key={question.id} open={index === 0}>
              <summary>
                <span className="result-detail__number">{String(index + 1).padStart(2, "0")}</span>
                <span className="result-detail__title"><small>{question.category}</small><strong>{question.question}</strong></span>
                <span className="question-band"><small>종합 평가</small><PercentileBadge value={video.scores.overall_performance.percentile} /></span>
                <span className="result-detail__chevron"><Icon name="chevron" size={17} /></span>
              </summary>
              <div className="result-detail__body">
                <section className="answer-transcript"><h3>내 답변 텍스트</h3>
                  <p className={video.transcript.trim() ? "" : "answer-transcript--empty"} lang={video.transcript.trim() ? "en" : "ko"}>{video.transcript.trim() ? video.transcript : "인식된 답변 텍스트가 없어요."}</p>
                  <small>음성을 자동으로 옮긴 텍스트로, 실제 발화와 다를 수 있어요.</small>
                </section>
                <section className="answer-band-section"><h3>항목별 백분위</h3><PercentileMetrics scores={video.scores} /></section>
              </div>
            </details>
          })}
        </div>
      </section>
    </main>
  )
}

export default function App() {
  const devices = useInterviewDevices()
  const devicesReadyRef = useRef(devices.ready)
  devicesReadyRef.current = devices.ready
  const [screen, setScreen] = useState<Screen>("home")
  const [session, setSession] = useState<InterviewSession | null>(null)
  const [submittedAnswers, setSubmittedAnswers] = useState<SubmittedAnswer[]>([])
  const [questionsLoading, setQuestionsLoading] = useState(false)
  const [questionsError, setQuestionsError] = useState("")
  const [analysisResult, setAnalysisResult] = useState<InterviewResult | null>(null)
  const [analysisError, setAnalysisError] = useState("")
  const [analysisTexts, setAnalysisTexts] = useState<InterviewTranscripts | null>(null)
  const requestRef = useRef<AbortController | null>(null)

  useEffect(() => () => requestRef.current?.abort(), [])

  const goHome = () => {
    devices.disconnect()
    const request = requestRef.current
    requestRef.current = null
    request?.abort()
    setQuestionsLoading(false)
    setQuestionsError("")
    setSession(null)
    setSubmittedAnswers([])
    setAnalysisTexts(null)
    setAnalysisResult(null)
    setAnalysisError("")
    setScreen("home")
  }

  const startInterview = async () => {
    if (requestRef.current || !devices.ready) return
    const request = new AbortController()
    requestRef.current = request
    setQuestionsLoading(true)
    setQuestionsError("")
    setSession(null)
    setSubmittedAnswers([])
    setAnalysisTexts(null)
    setAnalysisResult(null)
    setAnalysisError("")
    const timeout = window.setTimeout(() => request.abort(), 15000)

    try {
      const created = await createInterview(request.signal)
      if (requestRef.current !== request) return
      if (!devicesReadyRef.current) throw new Error("장치 연결이 끊겼어요. 카메라·마이크를 다시 연결한 뒤 시작해 주세요.")
      setSession(created)
      setScreen("interview")
    } catch (error) {
      if (requestRef.current !== request) return
      setQuestionsError(
        request.signal.aborted
          ? "응답 시간이 초과됐어요. 잠시 후 다시 시도해 주세요."
          : error instanceof TypeError
            ? "질문 서버에 연결할 수 없어요. 백엔드가 실행 중인지 확인해 주세요."
            : error instanceof Error ? error.message : "질문을 불러오지 못했습니다.",
      )
    } finally {
      window.clearTimeout(timeout)
      if (requestRef.current === request) {
        requestRef.current = null
        setQuestionsLoading(false)
      }
    }
  }

  const startAnalysis = async () => {
    if (!session || requestRef.current) return
    const request = new AbortController()
    requestRef.current = request
    setAnalysisError("")
    setAnalysisResult(null)
    setScreen("analysis")
    let timeout = window.setTimeout(() => request.abort(), 360000)
    try {
      const texts = await transcribeInterview(session, request.signal)
      if (requestRef.current !== request) return
      setAnalysisTexts(texts)
      window.clearTimeout(timeout)
      timeout = window.setTimeout(() => request.abort(), 360000)
      const result = await analyzeInterview(session, request.signal)
      if (requestRef.current !== request) return
      setAnalysisResult(result)
      setScreen("results")
    } catch (error) {
      if (requestRef.current !== request) return
      setAnalysisError(request.signal.aborted ? "분석 응답 시간이 초과됐어요. 잠시 후 다시 요청해 주세요."
        : error instanceof TypeError ? "분석 서버에 연결할 수 없어요. 잠시 후 다시 요청해 주세요."
        : error instanceof Error ? error.message : "답변을 분석하지 못했어요.")
    } finally {
      window.clearTimeout(timeout)
      if (requestRef.current === request) requestRef.current = null
    }
  }

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" })
  }, [screen])

  return (
    <>
      {screen === "home" && <Home onStart={() => setScreen("guide")} />}
      {screen === "guide" && (
        <Guide
          onBack={goHome}
          onStart={startInterview}
          loading={questionsLoading}
          error={questionsError}
          devices={devices}
        />
      )}
      {screen === "interview" && session && (
        <Interview
          key={session.id}
          sessionId={session.id}
          questions={session.questions}
          devices={devices}
          onExit={goHome}
          onFinish={(answers) => { devices.disconnect(); setSubmittedAnswers(answers); setScreen("review") }}
        />
      )}
      {screen === "review" && session && <AnswerReview session={session} answers={submittedAnswers} onAnalyze={startAnalysis} />}
      {screen === "analysis" && session && (
        <Analysis session={session} transcripts={analysisTexts} error={analysisError} onRetry={startAnalysis} onBack={() => setScreen("review")} />
      )}
      {screen === "results" && session && analysisResult && <Results result={analysisResult} session={session} />}
    </>
  )
}
