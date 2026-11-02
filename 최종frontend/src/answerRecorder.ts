export type Recording = { blob: Blob; duration: number }

// Owns only the recording; the shared camera/microphone stream stays connected.
export function createAnswerRecorder(
  stream: MediaStream,
  callbacks: { complete: (answer: Recording) => void; error: (message: string) => void },
) {
  if (typeof MediaRecorder === "undefined") throw new Error("이 브라우저는 영상 녹화를 지원하지 않아요. Chrome 또는 Edge에서 열어 주세요.")
  const mimeType = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"]
    .find((type) => MediaRecorder.isTypeSupported(type))
  if (!mimeType) throw new Error("이 브라우저는 WebM 녹화를 지원하지 않아요. Chrome 또는 Edge에서 열어 주세요.")
  if (!["video", "audio"].every((kind) => stream.getTracks().some((track) => track.kind === kind && track.readyState === "live" && !track.muted))) {
    throw new Error("카메라·마이크를 다시 연결한 뒤 녹화를 시작해 주세요.")
  }
  const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 2_500_000 })
  const chunks: Blob[] = []
  let active = true
  let startedAt = 0
  let stoppedAt = 0
  let timer: ReturnType<typeof setTimeout>
  const cleanup = () => {
    clearTimeout(timer)
    stream.getTracks().forEach((track) => track.removeEventListener("ended", lostDevice))
  }
  const fail = (message: string) => {
    if (!active) return
    active = false
    cleanup()
    if (recorder.state !== "inactive") recorder.stop()
    callbacks.error(message)
  }
  const lostDevice = () => fail("녹화 중 장치 연결이 끊겼어요. 다시 연결한 뒤 녹화해 주세요.")
  const stop = () => {
    if (!active || recorder.state === "inactive") return
    stoppedAt = performance.now()
    clearTimeout(timer)
    recorder.stop()
  }
  recorder.ondataavailable = (event) => { if (active && event.data.size) chunks.push(event.data) }
  recorder.onerror = () => fail("녹화하지 못했어요. 장치를 확인한 뒤 다시 시도해 주세요.")
  recorder.onstop = () => {
    cleanup()
    if (!active) return
    active = false
    const blob = new Blob(chunks, { type: "video/webm" })
    if (!blob.size) return callbacks.error("녹화된 영상이 없어요. 다시 녹화해 주세요.")
    callbacks.complete({ blob, duration: Math.min(60, Math.max(0, ((stoppedAt || performance.now()) - startedAt) / 1000)) })
  }
  return {
    start() {
      try {
        recorder.start(1000)
        startedAt = performance.now()
        stream.getTracks().forEach((track) => track.addEventListener("ended", lostDevice))
        timer = setTimeout(stop, 60_000)
      } catch (error) { active = false; cleanup(); throw error }
    },
    stop,
    dispose() {
      active = false
      cleanup()
      if (recorder.state !== "inactive") recorder.stop()
    },
  }
}
