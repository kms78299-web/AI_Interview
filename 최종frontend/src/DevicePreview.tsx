import { useEffect, useRef, useState } from "react"

export function CameraPreview({ stream, onDimensions }: { stream: MediaStream | null; onDimensions?: (width: number, height: number) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playError, setPlayError] = useState(false)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    let active = true
    setPlayError(false)
    video.srcObject = stream
    if (stream) video.play().catch(() => { if (active) setPlayError(true) })
    return () => {
      active = false
      video.pause()
      video.srcObject = null
    }
  }, [stream])

  return <>
    <video ref={videoRef} className="live-camera-preview" autoPlay muted playsInline onLoadedMetadata={(event) => onDimensions?.(event.currentTarget.videoWidth, event.currentTarget.videoHeight)} aria-label="실시간 카메라 미리보기" />
    {playError && <button className="preview-play-button" type="button" onClick={() => {
      videoRef.current?.play().then(() => setPlayError(false)).catch(() => setPlayError(true))
    }}>카메라 미리보기 재생</button>}
  </>
}

export function RecordedPreview({ url, duration, onDimensions }: { url: string; duration: number; onDimensions: (width: number, height: number) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(false)
  const [position, setPosition] = useState(0)
  const [error, setError] = useState("")
  const time = (value: number) => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}`

  return <>
    <video ref={videoRef} className="recorded-answer-preview" src={url} playsInline preload="auto" muted={muted}
      aria-label="녹화된 답변 미리보기"
      onLoadedMetadata={(event) => onDimensions(event.currentTarget.videoWidth, event.currentTarget.videoHeight)}
      onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
      onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
      onError={() => setError("영상을 재생하지 못했어요. 다시 녹화해 주세요.")} />
    <div className="recorded-playback-controls">
      <button type="button" title={playing ? "일시정지" : "재생"} aria-label={playing ? "녹화 영상 일시정지" : "녹화 영상 재생"} onClick={() => {
        const video = videoRef.current
        if (!video) return
        if (playing) video.pause()
        else { if (video.ended) video.currentTime = 0; void video.play().catch(() => setError("재생 버튼을 다시 눌러 주세요.")) }
      }}><svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
        {playing ? <><rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" /></> : <path d="M7 4.5v15l12-7.5Z" />}
      </svg></button>
      <span>{time(position)} / {time(duration)}</span>
      <input type="range" aria-label="녹화 영상 재생 위치" min={0} max={Math.max(duration, 0.1)} step={0.01}
        value={Math.min(position, duration)} onChange={(event) => {
          if (videoRef.current) videoRef.current.currentTime = Number(event.target.value)
          setPosition(Number(event.target.value))
        }} />
      <button type="button" title={muted ? "소리 켜기" : "소리 끄기"} aria-label={muted ? "녹화 영상 소리 켜기" : "녹화 영상 소리 끄기"} onClick={() => setMuted(!muted)}>
        <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M11 4 6 8H3v8h3l5 4Z" />
          {muted ? <path d="m16 9 6 6m0-6-6 6" /> : <><path d="M15 8a6 6 0 0 1 0 8" /><path d="M18 5a10 10 0 0 1 0 14" /></>}
        </svg>
      </button>
    </div>
    {error && <p className="recorded-playback-error" role="alert">{error}</p>}
  </>
}

export function MicrophoneLevel({ stream }: { stream: MediaStream | null }) {
  const [level, setLevel] = useState(0)
  const [error, setError] = useState("")

  useEffect(() => {
    setLevel(0)
    setError("")
    if (!stream || !stream.getAudioTracks().some((track) => track.readyState === "live")) return
    let context: AudioContext | undefined
    let source: MediaStreamAudioSourceNode | undefined
    let analyser: AnalyserNode | undefined
    let frame = 0
    let active = true
    try {
      context = new AudioContext()
      source = context.createMediaStreamSource(stream)
      analyser = context.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser) // No connection to speakers: avoids echo/feedback.
      const samples = new Uint8Array(analyser.fftSize)
      let updatedAt = 0
      const read = (now: number) => {
        if (!active || !analyser) return
        if (now - updatedAt >= 100) {
          analyser.getByteTimeDomainData(samples)
          const rms = Math.sqrt(samples.reduce((sum, sample) => sum + ((sample - 128) / 128) ** 2, 0) / samples.length)
          setLevel(Math.min(100, Math.round(rms * 400)))
          updatedAt = now
        }
        frame = requestAnimationFrame(read)
      }
      void context.resume().catch(() => { if (active) setError("음량 표시를 시작하지 못했어요. 장치를 다시 연결해 주세요.") })
      frame = requestAnimationFrame(read)
    } catch {
      setError("이 브라우저에서는 음량 표시를 지원하지 않아요.")
    }
    return () => {
      active = false
      cancelAnimationFrame(frame)
      source?.disconnect()
      analyser?.disconnect()
      void context?.close().catch(() => {})
    }
  }, [stream])

  return <div className="microphone-level">
    <div><span>마이크 입력</span><small>{error || (stream ? "말하면 막대가 움직여요" : "장치를 연결해 주세요")}</small></div>
    <div className="microphone-level__track" role="meter" aria-label="마이크 입력 크기" aria-valuemin={0} aria-valuemax={100} aria-valuenow={level}>
      <span style={{ width: `${level}%` }} />
    </div>
  </div>
}
