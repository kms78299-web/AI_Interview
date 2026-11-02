export type NarrationState = { phase: "idle" | "reading" | "countdown"; seconds: number }

type NarrationOptions = {
  synthesis: Pick<SpeechSynthesis, "getVoices" | "speak" | "cancel">
  makeUtterance: (text: string) => SpeechSynthesisUtterance
  schedule: (callback: () => void, delay: number) => number
  unschedule: (timer: number) => void
  onState: (state: NarrationState) => void
  onReady: () => void
  onError: (message: string) => void
}

// Own the utterance and timers together so stale speech events cannot start recording.
export function createQuestionNarration(options: NarrationOptions) {
  let generation = 0
  let utterance: SpeechSynthesisUtterance | null = null
  let timer: number | undefined
  let disposed = false
  const clearTimer = () => {
    if (timer !== undefined) options.unschedule(timer)
    timer = undefined
  }
  const cancel = () => {
    generation++
    clearTimer()
    if (utterance) {
      utterance.onend = null
      utterance.onerror = null
      utterance = null
      options.synthesis.cancel()
    }
    options.onState({ phase: "idle", seconds: 0 })
  }
  const start = (text: string) => {
    if (disposed || !text.trim()) return
    cancel()
    const active = generation
    try {
      const speech = options.makeUtterance(text)
      utterance = speech
      speech.lang = "en-US"
      speech.rate = 0.9
      const voices = options.synthesis.getVoices()
      const voice = voices.find(item => item.lang.replace("_", "-").toLowerCase() === "en-us")
        ?? voices.find(item => /^en(?:[-_]|$)/i.test(item.lang))
      if (voice) speech.voice = voice
      const fail = () => {
        if (active !== generation || disposed) return
        cancel()
        options.onError("질문을 읽지 못했어요. 다시 눌러보거나 ‘녹화 시작’으로 직접 답변해 주세요.")
      }
      speech.onerror = fail
      speech.onend = () => {
        if (active !== generation || disposed) return
        clearTimer()
        utterance = null
        speech.onend = null
        speech.onerror = null
        const countdown = (seconds: number) => {
          if (active !== generation || disposed) return
          if (seconds === 0) {
            timer = undefined
            options.onState({ phase: "idle", seconds: 0 })
            options.onReady()
            return
          }
          options.onState({ phase: "countdown", seconds })
          timer = options.schedule(() => countdown(seconds - 1), 1000)
        }
        countdown(5)
      }
      options.onState({ phase: "reading", seconds: 0 })
      // A browser that never delivers an end/error event must not leave the UI stuck.
      timer = options.schedule(fail, 120000)
      options.synthesis.speak(speech)
    } catch {
      cancel()
      options.onError("이 브라우저에서 질문 읽기를 시작하지 못했어요. ‘녹화 시작’으로 직접 답변해 주세요.")
    }
  }
  return { start, cancel, dispose: () => { cancel(); disposed = true } }
}
