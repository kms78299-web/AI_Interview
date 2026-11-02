import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

const source = await readFile(new URL('../src/deviceSession.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
})
const { createDeviceSession } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)

class Track extends EventTarget {
  readyState = 'live'
  enabled = true
  muted = false
  stops = 0
  constructor(kind) { super(); this.kind = kind }
  stop() { this.readyState = 'ended'; this.stops += 1 }
}

function streamWith(kinds = ['video', 'audio']) {
  const tracks = kinds.map((kind) => new Track(kind))
  return {
    getTracks: () => tracks,
    getVideoTracks: () => tracks.filter((track) => track.kind === 'video'),
    getAudioTracks: () => tracks.filter((track) => track.kind === 'audio'),
  }
}

test('연결된 동일 스트림을 재사용하고 나갈 때 두 장치를 해제한다', async () => {
  const stream = streamWith(), states = []
  let requests = 0
  const devices = createDeviceSession(async () => { requests++; return stream }, (state) => states.push(state))
  await devices.connect()
  assert.equal(states.at(-1).cameraReady, true)
  assert.equal(states.at(-1).microphoneReady, true)
  assert.equal(states.at(-1).stream, stream)
  await devices.connect()
  assert.equal(requests, 1)
  devices.stop()
  assert.equal(states.at(-1).stream, null)
  assert.ok(stream.getTracks().every((track) => track.stops === 1))
})

test('권한 응답 전에 나가면 나중에 받은 장치도 즉시 해제한다', async () => {
  const stream = streamWith(), states = []
  let resolve
  const devices = createDeviceSession(() => new Promise((done) => { resolve = done }), (state) => states.push(state))
  const pending = devices.connect()
  devices.stop()
  resolve(stream)
  await pending
  assert.equal(states.at(-1).status, 'idle')
  assert.ok(stream.getTracks().every((track) => track.readyState === 'ended'))
})

test('연결 버튼을 중복으로 눌러도 권한 요청을 한 번만 보낸다', async () => {
  let resolve, requests = 0
  const devices = createDeviceSession(() => { requests++; return new Promise((done) => { resolve = done }) }, () => {})
  const pending = devices.connect()
  await devices.connect()
  assert.equal(requests, 1)
  resolve(streamWith())
  await pending
  devices.stop()
})

test('한 장치만 들어온 경우 시작을 허용하지 않고 나머지 장치도 해제한다', async () => {
  const stream = streamWith(['video']), states = []
  const devices = createDeviceSession(async () => stream, (state) => states.push(state))
  await devices.connect()
  assert.equal(states.at(-1).status, 'error')
  assert.equal(states.at(-1).cameraReady, false)
  assert.equal(stream.getTracks()[0].readyState, 'ended')
})

test('장치가 분리되면 모든 트랙을 해제하고 재연결 안내를 표시한다', async () => {
  const stream = streamWith(), states = []
  const devices = createDeviceSession(async () => stream, (state) => states.push(state))
  await devices.connect()
  stream.getVideoTracks()[0].dispatchEvent(new Event('ended'))
  assert.equal(states.at(-1).status, 'error')
  assert.equal(states.at(-1).stream, null)
  assert.ok(stream.getTracks().every((track) => track.readyState === 'ended'))
})

test('권한 거부 후 다시 연결할 수 있다', async () => {
  const states = [], stream = streamWith()
  let requests = 0
  const devices = createDeviceSession(async () => {
    if (++requests === 1) throw new DOMException('Denied', 'NotAllowedError')
    return stream
  }, (state) => states.push(state))
  await devices.connect()
  assert.equal(states.at(-1).status, 'error')
  assert.match(states.at(-1).message, /권한/)
  await devices.connect()
  assert.equal(states.at(-1).status, 'ready')
  devices.stop()
})

test('마이크 입력 중단과 복구를 연결 상태에 반영한다', async () => {
  const states = [], stream = streamWith()
  const devices = createDeviceSession(async () => stream, (state) => states.push(state))
  await devices.connect()
  const mic = stream.getAudioTracks()[0]
  mic.muted = true
  mic.dispatchEvent(new Event('mute'))
  assert.equal(states.at(-1).microphoneReady, false)
  mic.muted = false
  mic.dispatchEvent(new Event('unmute'))
  assert.equal(states.at(-1).microphoneReady, true)
  devices.stop()
})
