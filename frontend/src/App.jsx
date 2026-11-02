import { useEffect, useRef, useState } from 'react'
import ResultView from './ResultView'
import { createDemoResult } from './resultData'

const categoryLabels = {
  introduction: '필수 자기소개',
  personal_values: '자기이해·직무 가치관',
  teamwork_leadership: '협업·소통·리더십',
  problem_solving: '문제 해결·성과·주도성',
  adaptability_growth: '적응력·성장·업무 관리',
}

function App() {
  const [screen, setScreen] = useState('home')
  const [cameraStatus, setCameraStatus] = useState('idle')
  const [cameraMessage, setCameraMessage] = useState('')
  const [recordingStatus, setRecordingStatus] = useState('idle')
  const [recordedUrl, setRecordedUrl] = useState(null)
  const [uploadStatus, setUploadStatus] = useState('idle')
  const [uploadMessage, setUploadMessage] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [questions, setQuestions] = useState([])
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [interviewStatus, setInterviewStatus] = useState('idle')
  const [interviewMessage, setInterviewMessage] = useState('')
  const [recordingSecondsLeft, setRecordingSecondsLeft] = useState(60)
  const [recordedDuration, setRecordedDuration] = useState(0)
  const [recordingAttempts, setRecordingAttempts] = useState({})
  const [savedAnswers, setSavedAnswers] = useState({})
  const [analysisStatus, setAnalysisStatus] = useState('idle')
  const [analysisMessage, setAnalysisMessage] = useState('')
  const [analysisResult, setAnalysisResult] = useState(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const requestIdRef = useRef(0)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const recordingIdRef = useRef(0)
  const recordedUrlRef = useRef(null)
  const recordedBlobRef = useRef(null)
  const frameStreamRef = useRef(null)
  const animationFrameRef = useRef(null)
  const cameraAttemptedRef = useRef(false)
  const recordingStartedAtRef = useRef(0)
  const recordingTimeoutRef = useRef(null)
  const recordingIntervalRef = useRef(null)

  function clearRecordingTimers() {
    if (recordingTimeoutRef.current !== null) clearTimeout(recordingTimeoutRef.current)
    if (recordingIntervalRef.current !== null) clearInterval(recordingIntervalRef.current)
    recordingTimeoutRef.current = null
    recordingIntervalRef.current = null
  }

  function stopFrameCapture() {
    if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current)
    animationFrameRef.current = null
    frameStreamRef.current?.getTracks().forEach((track) => track.stop())
    frameStreamRef.current = null
  }

  function clearRecording() {
    if (recordedUrlRef.current) URL.revokeObjectURL(recordedUrlRef.current)
    recordedUrlRef.current = null
    recordedBlobRef.current = null
    setRecordedUrl(null)
    setUploadStatus('idle')
    setUploadMessage('')
    setRecordedDuration(0)
    setRecordingSecondsLeft(60)
  }

  function stopPreview() {
    requestIdRef.current += 1
    recordingIdRef.current += 1
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
    recorderRef.current = null
    clearRecordingTimers()
    stopFrameCapture()
    chunksRef.current = []
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraStatus('idle')
    setRecordingStatus('idle')
    setCameraMessage('')
  }

  function startRecording() {
    if (!streamRef.current || cameraStatus !== 'ready') return
    if (typeof MediaRecorder === 'undefined') {
      setCameraMessage('이 브라우저에서는 영상 녹화를 지원하지 않습니다.')
      return
    }

    try {
      clearRecording()
      chunksRef.current = []
      const recordingId = ++recordingIdRef.current
      const preview = videoRef.current
      if (!preview.videoWidth || !preview.videoHeight) {
        throw new Error('카메라 영상이 아직 준비되지 않았습니다.')
      }

      // 영상 프레임만 좌우로 뒤집어 새 스트림을 만듭니다. 마이크 소리는 원본을 사용합니다.
      const canvas = document.createElement('canvas')
      canvas.width = preview.videoWidth
      canvas.height = preview.videoHeight
      const context = canvas.getContext('2d')
      if (!context || !canvas.captureStream) throw new Error('영상 변환을 지원하지 않습니다.')
      context.translate(canvas.width, 0)
      context.scale(-1, 1)

      function drawFrame() {
        context.drawImage(preview, 0, 0, canvas.width, canvas.height)
        animationFrameRef.current = requestAnimationFrame(drawFrame)
      }
      drawFrame()

      const frameStream = canvas.captureStream(30)
      frameStreamRef.current = frameStream
      const recordingStream = new MediaStream([
        ...frameStream.getVideoTracks(),
        ...streamRef.current.getAudioTracks(),
      ])
      const webmMimeType = [
        'video/webm;codecs=vp9,opus',
        'video/webm;codecs=vp8,opus',
        'video/webm',
      ].find((type) => MediaRecorder.isTypeSupported(type))

      if (!webmMimeType) throw new Error('WebM 녹화를 지원하지 않습니다.')

      const recorder = new MediaRecorder(recordingStream, { mimeType: webmMimeType })
      recorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (recordingId === recordingIdRef.current && event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }
      recorder.onstop = () => {
        if (recordingId !== recordingIdRef.current) return
        stopFrameCapture()
        clearRecordingTimers()

        const duration = Math.min((Date.now() - recordingStartedAtRef.current) / 1000, 60)
        setRecordedDuration(duration)

        const blob = new Blob(chunksRef.current, { type: recorder.mimeType })
        chunksRef.current = []
        recorderRef.current = null
        setRecordingStatus('idle')

        if (blob.size === 0) {
          setCameraMessage('녹화된 영상이 없습니다. 다시 시도해 주세요.')
          return
        }

        const url = URL.createObjectURL(blob)
        recordedBlobRef.current = blob
        recordedUrlRef.current = url
        setRecordedUrl(url)
        setRecordingAttempts((attempts) => ({
          ...attempts,
          [currentQuestion.id]: (attempts[currentQuestion.id] || 0) + 1,
        }))
        setCameraMessage('녹화가 끝났습니다. 아래에서 답변 영상을 재생해 보세요.')
      }
      recorder.onerror = () => {
        if (recordingId !== recordingIdRef.current) return
        recordingIdRef.current += 1
        clearRecordingTimers()
        stopFrameCapture()
        recorderRef.current = null
        chunksRef.current = []
        setRecordingStatus('idle')
        setCameraMessage('녹화 중 오류가 발생했습니다. 다시 시도해 주세요.')
      }

      recorder.start()
      recordingStartedAtRef.current = Date.now()
      setRecordingSecondsLeft(60)
      recordingIntervalRef.current = setInterval(() => {
        const elapsedSeconds = (Date.now() - recordingStartedAtRef.current) / 1000
        setRecordingSecondsLeft(Math.max(0, Math.ceil(60 - elapsedSeconds)))
      }, 250)
      recordingTimeoutRef.current = setTimeout(() => {
        if (recorderRef.current?.state === 'recording') {
          setRecordingStatus('processing')
          setCameraMessage('60초가 지나 녹화를 자동으로 종료합니다.')
          recorderRef.current.stop()
        }
      }, 60000)
      setRecordingStatus('recording')
      setCameraMessage('녹화 중입니다. 답변을 마치면 녹화 종료를 눌러 주세요.')
    } catch {
      stopFrameCapture()
      recorderRef.current = null
      setRecordingStatus('idle')
      setCameraMessage('녹화를 시작하지 못했습니다. 장치 연결을 확인해 주세요.')
    }
  }

  function stopRecording() {
    if (recorderRef.current?.state !== 'recording') return
    setRecordingStatus('processing')
    setCameraMessage('녹화를 마무리하고 있습니다.')
    clearRecordingTimers()
    recorderRef.current.stop()
  }

  async function uploadRecording() {
    const blob = recordedBlobRef.current
    if (!blob || uploadStatus === 'uploading') return

    const formData = new FormData()
    formData.append('video', blob, 'recording.webm')
    formData.append('duration_seconds', recordedDuration.toFixed(2))

    setUploadStatus('uploading')
    setUploadMessage('영상을 서버로 전송하고 있습니다.')

    try {
      const response = await fetch(`/api/interviews/${sessionId}/answers/${currentQuestion.id}`, {
        method: 'POST',
        body: formData,
      })
      const body = await response.json().catch(() => null)

      if (!response.ok) {
        throw new Error(body?.detail || '서버가 영상을 저장하지 못했습니다.')
      }

      setUploadStatus('success')
      setSavedAnswers((answers) => ({ ...answers, [currentQuestion.id]: body }))
      setUploadMessage(`답변 저장 완료: ${body.filename} (${(body.size_bytes / 1024 / 1024).toFixed(2)} MB)`)
    } catch (error) {
      setUploadStatus('error')
      setUploadMessage(error instanceof Error ? error.message : '영상 업로드 중 오류가 발생했습니다.')
    }
  }

  async function startPreview() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraMessage('이 브라우저에서는 카메라와 마이크를 사용할 수 없습니다.')
      return
    }

    const requestId = ++requestIdRef.current
    setCameraStatus('loading')
    setCameraMessage('카메라와 마이크 권한을 요청하고 있습니다.')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })

      // 권한 창이 열린 동안 시작 화면으로 돌아갔다면 장치를 바로 해제합니다.
      if (requestId !== requestIdRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }

      streamRef.current = stream
      videoRef.current.srcObject = stream
      await videoRef.current.play()

      if (requestId === requestIdRef.current) {
        setCameraStatus('ready')
        setCameraMessage('카메라와 마이크가 연결되었습니다. 아직 녹화는 시작되지 않았습니다.')
      }
    } catch (error) {
      if (requestId !== requestIdRef.current) return

      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      if (videoRef.current) videoRef.current.srcObject = null
      setCameraStatus('idle')
      setCameraMessage(
        error.name === 'NotAllowedError'
          ? '이 브라우저에서 카메라·마이크 접근이 허용되지 않았습니다. Chrome이나 Edge에서 이 주소를 열고 권한을 허용해 주세요.'
          : error.name === 'NotFoundError'
            ? '카메라 또는 마이크를 찾을 수 없습니다.'
            : '장치를 연결하지 못했습니다. 연결 상태를 확인하고 다시 시도해 주세요.',
      )
    }
  }

  useEffect(() => {
    return () => {
      requestIdRef.current += 1
      recordingIdRef.current += 1
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
      clearRecordingTimers()
      stopFrameCapture()
      streamRef.current?.getTracks().forEach((track) => track.stop())
      if (recordedUrlRef.current) URL.revokeObjectURL(recordedUrlRef.current)
    }
  }, [])

  useEffect(() => {
    if (
      screen !== 'question'
      || questions.length === 0
      || cameraStatus !== 'idle'
      || streamRef.current
      || cameraAttemptedRef.current
    ) return undefined

    cameraAttemptedRef.current = true
    const frameId = requestAnimationFrame(() => startPreview())
    return () => cancelAnimationFrame(frameId)
  }, [screen, questions.length, cameraStatus])

  function goHome() {
    stopPreview()
    clearRecording()
    cameraAttemptedRef.current = false
    setAnalysisStatus('idle')
    setAnalysisMessage('')
    setAnalysisResult(null)
    setScreen('home')
  }

  async function startInterview() {
    setInterviewStatus('loading')
    setInterviewMessage('면접 질문을 선택하고 있습니다.')

    try {
      const response = await fetch('/api/interviews', { method: 'POST' })
      const body = await response.json().catch(() => null)
      if (!response.ok) throw new Error(body?.detail || '질문을 불러오지 못했습니다.')

      setSessionId(body.session_id)
      setQuestions(body.questions)
      setCurrentQuestionIndex(0)
      setRecordingAttempts({})
      setSavedAnswers({})
      setAnalysisStatus('idle')
      setAnalysisMessage('')
      setAnalysisResult(null)
      cameraAttemptedRef.current = false
      setInterviewStatus('ready')
      setInterviewMessage('')
      setScreen('question')
    } catch (error) {
      setInterviewStatus('error')
      setInterviewMessage(error instanceof Error ? error.message : '질문을 불러오지 못했습니다.')
    }
  }

  function showNextQuestion() {
    if (recordingStatus !== 'idle' || uploadStatus !== 'success') return
    clearRecording()

    if (currentQuestionIndex === questions.length - 1) {
      stopPreview()
      setScreen('complete')
      return
    }
    setCurrentQuestionIndex((index) => index + 1)
  }

  async function analyzeInterview() {
    if (!sessionId || analysisStatus === 'loading') return

    setAnalysisStatus('loading')
    setAnalysisMessage('답변 5개의 분석 결과를 요청하고 있습니다. 잠시 기다려 주세요.')

    try {
      const response = await fetch(`/api/interviews/${sessionId}/analyze`, {
        method: 'POST',
      })
      const body = await response.json().catch(() => null)

      if (!response.ok) {
        throw new Error(body?.detail || '분석 서버에서 결과를 받지 못했습니다.')
      }

      setAnalysisResult(body)
      setAnalysisStatus('success')
      setAnalysisMessage('')
      setScreen('result')
    } catch (error) {
      setAnalysisStatus('error')
      setAnalysisMessage(error instanceof Error ? error.message : '분석 중 오류가 발생했습니다.')
    }
  }

  const currentQuestion = questions[currentQuestionIndex]
  const currentAttempts = currentQuestion ? (recordingAttempts[currentQuestion.id] || 0) : 0

  return (
    <main className="page">
      {screen === 'home' ? (
        <section className="content" aria-labelledby="service-title">
          <h1 id="service-title">AI 모의면접</h1>
          <p>AI 기반 모의면접 분석 서비스</p>
          <button type="button" onClick={startInterview} disabled={interviewStatus === 'loading'}>
            {interviewStatus === 'loading' ? '질문 선택 중...' : '면접 시작하기'}
          </button>
          {interviewMessage && <p className="upload-message error" role="alert">{interviewMessage}</p>}
          <div className="demo-preview-action">
            <button
              className="secondary-button"
              type="button"
              disabled={interviewStatus === 'loading'}
              onClick={() => {
                setAnalysisResult(createDemoResult())
                setScreen('result')
              }}
            >
              데모 결과 미리보기
            </button>
          </div>
        </section>
      ) : screen === 'question' ? (
        <section className="content" aria-labelledby="question-title">
          <h1 id="question-title">면접 질문</h1>
          <p className="question-number">질문 {currentQuestionIndex + 1} / {questions.length}</p>
          <p className="question-category">{categoryLabels[currentQuestion?.category]}</p>
          <p className="question-text">{currentQuestion?.text}</p>
          <video
            ref={videoRef}
            className="preview"
            autoPlay
            muted
            playsInline
            aria-label="웹캠 미리보기"
          />
          <p className="camera-message" role="status">
            {cameraMessage || '웹캠을 연결하고 있습니다.'}
          </p>
          {cameraStatus === 'idle' && cameraMessage && (
            <button
              type="button"
              onClick={() => {
                cameraAttemptedRef.current = true
                startPreview()
              }}
            >
              카메라·마이크 다시 연결
            </button>
          )}
          {cameraStatus === 'ready' && (
            <div className="actions recording-actions">
              {recordingStatus === 'recording' ? (
                <>
                  <span className="recording-timer">남은 시간 {recordingSecondsLeft}초</span>
                  <button type="button" onClick={stopRecording}>답변 완료</button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={startRecording}
                  disabled={
                    recordingStatus === 'processing'
                    || uploadStatus === 'uploading'
                    || (recordedUrl && currentAttempts >= 2)
                  }
                >
                  {recordedUrl
                    ? currentAttempts >= 2
                      ? '다시 녹화 사용 완료'
                      : '다시 녹화 (1회 가능)'
                    : '녹화 시작'}
                </button>
              )}
            </div>
          )}
          {recordedUrl && (
            <div className="recording-result">
              <h2>질문 {currentQuestionIndex + 1} 답변</h2>
              <video className="recorded-video" src={recordedUrl} controls playsInline />
              <p className="recording-duration">녹화 시간: {recordedDuration.toFixed(1)}초</p>
              <button
                className="result-button"
                type="button"
                onClick={uploadRecording}
                disabled={uploadStatus === 'uploading' || uploadStatus === 'success'}
              >
                {uploadStatus === 'uploading' ? '저장 중...' : uploadStatus === 'success' ? '저장 완료' : '이 답변 저장'}
              </button>
              {uploadMessage && (
                <p className={`upload-message ${uploadStatus}`} role="status">{uploadMessage}</p>
              )}
            </div>
          )}
          <p className="implementation-note">답변은 최대 60초까지 녹화되며, 저장이 완료되면 다음 질문으로 이동할 수 있습니다.</p>
          <button
            className="secondary-button next-question-button"
            type="button"
            onClick={showNextQuestion}
            disabled={recordingStatus !== 'idle' || uploadStatus !== 'success'}
          >
            {currentQuestionIndex === questions.length - 1 ? '질문 확인 완료' : '다음 질문'}
          </button>
        </section>
      ) : screen === 'complete' ? (
        <section className="content" aria-labelledby="complete-title">
          <h1 id="complete-title">질문 확인 완료</h1>
          <p>총 5개의 질문과 저장된 답변 영상을 순서대로 확인할 수 있습니다.</p>
          <ol className="selected-question-list answer-review-list">
            {questions.map((question) => (
              <li key={question.id}>
                <span>{categoryLabels[question.category]}</span>
                <strong>{question.text}</strong>
                {savedAnswers[question.id] && (
                  <video
                    className="recorded-video review-video"
                    src={`/api/interviews/${sessionId}/answers/${question.id}/video`}
                    controls
                    playsInline
                    preload="metadata"
                  />
                )}
              </li>
            ))}
          </ol>
          <p className="session-id">세션 ID: {sessionId}</p>
          <button
            type="button"
            onClick={analyzeInterview}
            disabled={analysisStatus === 'loading'}
          >
            {analysisStatus === 'loading' ? '분석 중...' : '분석하기'}
          </button>
          {analysisMessage && (
            <p className={`upload-message ${analysisStatus}`} role="status">{analysisMessage}</p>
          )}
          <button className="secondary-button home-button" type="button" onClick={goHome}>처음으로</button>
        </section>
      ) : (
        <ResultView
          result={analysisResult}
          questions={questions}
          onReview={analysisResult?.provider === 'ui_preview' ? undefined : () => setScreen('complete')}
          onHome={goHome}
        />
      )}
    </main>
  )
}

export default App
