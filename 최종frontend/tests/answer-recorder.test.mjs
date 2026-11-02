import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

const { outputText } = ts.transpileModule(await readFile(new URL('../src/answerRecorder.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
})
const { createAnswerRecorder } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)

function setup(t) {
  class FakeRecorder {
    static latest
    static isTypeSupported(type) { return type === 'video/webm;codecs=vp8,opus' }
    state = 'inactive'
    constructor(stream, options) { this.stream = stream; this.options = options; FakeRecorder.latest = this }
    start() { this.state = 'recording' }
    stop() {
      this.state = 'inactive'
      queueMicrotask(() => {
        this.ondataavailable?.({ data: new Blob(['final-video-and-audio']) })
        this.onstop?.()
      })
    }
  }
  const original = Object.getOwnPropertyDescriptor(globalThis, 'MediaRecorder')
  Object.defineProperty(globalThis, 'MediaRecorder', { value: FakeRecorder, configurable: true, writable: true })
  t.after(() => {
    if (original) Object.defineProperty(globalThis, 'MediaRecorder', original)
    else delete globalThis.MediaRecorder
  })
  const tracks = ['video', 'audio'].map((kind) => Object.assign(new EventTarget(), { kind, readyState: 'live', muted: false }))
  const stream = { getTracks: () => tracks }
  const complete = t.mock.fn()
  const error = t.mock.fn()
  return { FakeRecorder, tracks, complete, error, recorder: createAnswerRecorder(stream, { complete, error }) }
}

test('최종 dataavailable을 받은 뒤 영상 생성, 공유 장치는 유지', async (t) => {
  const { recorder, FakeRecorder, complete, error, tracks } = setup(t)
  recorder.start()
  assert.equal(FakeRecorder.latest.options.mimeType, 'video/webm;codecs=vp8,opus')
  recorder.stop()
  assert.equal(complete.mock.callCount(), 0)
  await new Promise(queueMicrotask)
  const answer = complete.mock.calls[0].arguments[0]
  assert.equal(await answer.blob.text(), 'final-video-and-audio')
  assert.equal(answer.blob.type, 'video/webm')
  assert.ok(answer.duration >= 0 && answer.duration <= 60)
  assert.equal(error.mock.callCount(), 0)
  assert.ok(tracks.every((track) => track.readyState === 'live'))
})

test('60초에 자동 종료한다', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const { recorder, FakeRecorder, complete } = setup(t)
  recorder.start()
  t.mock.timers.tick(60_000)
  await new Promise(queueMicrotask)
  assert.equal(FakeRecorder.latest.state, 'inactive')
  assert.equal(complete.mock.callCount(), 1)
})

test('화면 이탈 시 늦게 도착하는 녹화 결과를 버린다', async (t) => {
  const { recorder, complete, error } = setup(t)
  recorder.start()
  recorder.dispose()
  await new Promise(queueMicrotask)
  assert.equal(complete.mock.callCount(), 0)
  assert.equal(error.mock.callCount(), 0)
})

test('장치 분리 시 불완전한 영상을 제출용으로 만들지 않는다', async (t) => {
  const { recorder, tracks, complete, error } = setup(t)
  recorder.start()
  tracks[0].dispatchEvent(new Event('ended'))
  await new Promise(queueMicrotask)
  assert.equal(complete.mock.callCount(), 0)
  assert.match(error.mock.calls[0].arguments[0], /연결이 끊겼어요/)
})
