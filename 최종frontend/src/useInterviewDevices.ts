import { useEffect, useRef, useState } from "react"
import { createDeviceSession, emptyDeviceState, requestBrowserDevices } from "./deviceSession"

export function useInterviewDevices() {
  const [state, setState] = useState(emptyDeviceState)
  const session = useRef<ReturnType<typeof createDeviceSession> | null>(null)
  if (!session.current) session.current = createDeviceSession(requestBrowserDevices, setState)

  useEffect(() => {
    const onPageHide = () => session.current?.stop()
    window.addEventListener("pagehide", onPageHide)
    return () => {
      window.removeEventListener("pagehide", onPageHide)
      session.current?.stop(false)
    }
  }, [])

  return {
    ...state,
    ready: state.cameraReady && state.microphoneReady,
    connect: () => session.current!.connect(),
    disconnect: () => session.current!.stop(),
  }
}

export type InterviewDevices = ReturnType<typeof useInterviewDevices>
