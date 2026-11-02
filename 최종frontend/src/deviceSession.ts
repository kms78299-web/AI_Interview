export type DeviceState = {
  status: "idle" | "connecting" | "ready" | "error"
  stream: MediaStream | null
  cameraReady: boolean
  microphoneReady: boolean
  message: string
}

export const emptyDeviceState: DeviceState = {
  status: "idle", stream: null, cameraReady: false, microphoneReady: false, message: "",
}

export function deviceErrorMessage(error: unknown): string {
  const name = error instanceof Error ? error.name : ""
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "카메라·마이크 권한이 허용되지 않았어요. 브라우저의 사이트 권한에서 두 장치를 허용한 뒤 다시 연결해 주세요."
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "카메라 또는 마이크를 찾을 수 없어요. 장치 연결을 확인한 뒤 다시 시도해 주세요."
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "장치를 사용할 수 없어요. 다른 앱에서 카메라·마이크를 사용 중인지 확인해 주세요."
  }
  return error instanceof Error ? error.message : "장치를 연결하지 못했어요. 다시 시도해 주세요."
}

export async function requestBrowserDevices(): Promise<MediaStream> {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new Error("이 브라우저에서는 장치를 연결할 수 없어요. Chrome이나 Edge에서 localhost 또는 HTTPS 주소로 열어 주세요.")
  }
  return navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30, max: 30 },
    },
    audio: { echoCancellation: true, noiseSuppression: true },
  })
}

// The same stream survives the guide → interview transition. Late permission
// responses are disposed after leaving, rather than switching the camera on again.
export function createDeviceSession(
  requestStream: () => Promise<MediaStream>,
  onChange: (state: DeviceState) => void,
) {
  let stream: MediaStream | null = null
  let generation = 0
  let pending = false
  let removeListeners = () => {}

  function release() {
    generation += 1
    pending = false
    removeListeners()
    removeListeners = () => {}
    stream?.getTracks().forEach((track) => track.stop())
    stream = null
  }

  function publishReadiness() {
    if (!stream) return
    const cameraReady = stream.getVideoTracks().some((track) => track.readyState === "live" && track.enabled && !track.muted)
    const microphoneReady = stream.getAudioTracks().some((track) => track.readyState === "live" && track.enabled && !track.muted)
    onChange({
      status: "ready", stream, cameraReady, microphoneReady,
      message: cameraReady && microphoneReady
        ? "카메라와 마이크가 연결됐어요. 아직 녹화나 서버 전송은 하지 않습니다."
        : "장치 입력이 일시 중지됐어요. 카메라·마이크 상태를 확인해 주세요.",
    })
  }

  return {
    async connect() {
      if (pending) return
      if (stream?.getTracks().every((track) => track.readyState === "live" && !track.muted)) {
        publishReadiness()
        return
      }
      release()
      const request = generation
      pending = true
      onChange({ ...emptyDeviceState, status: "connecting", message: "카메라와 마이크 권한을 확인하고 있어요. 브라우저에서 허용을 선택해 주세요." })
      try {
        const received = await requestStream()
        if (request !== generation) {
          received.getTracks().forEach((track) => track.stop())
          return
        }
        if (!received.getVideoTracks().some((track) => track.readyState === "live")
          || !received.getAudioTracks().some((track) => track.readyState === "live")) {
          received.getTracks().forEach((track) => track.stop())
          throw new Error("카메라와 마이크가 모두 필요해요. 두 장치의 연결 상태를 확인해 주세요.")
        }
        stream = received
        const tracks = received.getTracks()
        const ended = () => {
          release()
          onChange({ ...emptyDeviceState, status: "error", message: "장치 연결이 끊겼어요. 카메라·마이크를 다시 연결해 주세요." })
        }
        tracks.forEach((track) => {
          track.addEventListener("ended", ended)
          track.addEventListener("mute", publishReadiness)
          track.addEventListener("unmute", publishReadiness)
        })
        removeListeners = () => tracks.forEach((track) => {
          track.removeEventListener("ended", ended)
          track.removeEventListener("mute", publishReadiness)
          track.removeEventListener("unmute", publishReadiness)
        })
        publishReadiness()
      } catch (error) {
        if (request === generation) onChange({ ...emptyDeviceState, status: "error", message: deviceErrorMessage(error) })
      } finally {
        if (request === generation) pending = false
      }
    },
    stop(notify = true) {
      release()
      if (notify) onChange({ ...emptyDeviceState })
    },
  }
}
