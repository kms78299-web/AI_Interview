import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

const source = await readFile(new URL('../src/questionNarration.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } })
const { createQuestionNarration } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)
function setup() {
  const timers = new Map(), states = [], utterances = []
  let next = 0, ready = 0, cancelled = 0
  const errors = []
  const voice = { lang: 'en-US', name: 'English' }
  const synthesis = { getVoices: () => [{ lang:'ko-KR' }, voice], speak: u => utterances.push(u), cancel: () => cancelled++ }
  const narrator = createQuestionNarration({ synthesis, makeUtterance: text => ({ text }),
    schedule: (cb, delay) => { timers.set(++next, { cb, delay }); return next }, unschedule: id => timers.delete(id),
    onState: state => states.push(state), onReady: () => ready++, onError: e => errors.push(e) })
  return { narrator, states, utterances, errors, timers, synthesis, voice,
    get ready() { return ready }, get cancelled() { return cancelled },
    tick() { const [id, task] = timers.entries().next().value; timers.delete(id); task.cb(); return task.delay } }
}

test('읽기를 마친 뒤 5초가 지나야 녹화하며 영어 음성을 선택한다', () => {
  const f = setup(); f.narrator.start('Tell me about yourself.')
  const u = f.utterances[0]
  assert.equal(u.lang, 'en-US'); assert.equal(u.voice, f.voice)
  assert.equal(f.ready, 0); assert.equal(f.states.at(-1).phase, 'reading')
  u.onend()
  assert.deepEqual(f.states.at(-1), { phase: 'countdown', seconds: 5 })
  for (let i = 0; i < 4; i++) { assert.equal(f.tick(), 1000); assert.equal(f.ready, 0) }
  assert.equal(f.states.at(-1).seconds, 1)
  assert.equal(f.tick(), 1000); assert.equal(f.ready, 1); assert.equal(f.timers.size, 0)
})

test('읽기 중지 후 늦게 도착한 종료 이벤트는 자동 녹화를 시작하지 않는다', () => {
  const f = setup(); f.narrator.start('Question one')
  const lateEnd = f.utterances[0].onend
  f.narrator.cancel(); lateEnd()
  assert.equal(f.cancelled, 1); assert.equal(f.timers.size, 0); assert.equal(f.ready, 0)
  assert.equal(f.states.at(-1).phase, 'idle')
})

test('카운트다운 취소·장치 해제·화면 종료에 사용하는 정리는 타이머를 제거한다', () => {
  const f = setup(); f.narrator.start('Question'); f.utterances[0].onend()
  f.tick(); f.narrator.cancel()
  assert.equal(f.ready, 0); assert.equal(f.timers.size, 0)
  f.narrator.start('Question two'); const lateEnd = f.utterances[1].onend
  f.narrator.dispose(); lateEnd(); f.narrator.start('Should not speak')
  assert.equal(f.utterances.length, 2); assert.equal(f.ready, 0); assert.equal(f.timers.size, 0)
})

test('중복 재생 요청은 기존 음성과 카운트다운을 무효화한다', () => {
  const f = setup(); f.narrator.start('First'); const lateEnd = f.utterances[0].onend
  f.narrator.start('Second'); lateEnd()
  assert.equal(f.states.at(-1).phase, 'reading'); assert.equal(f.ready, 0)
  f.utterances[1].onend()
  for (let i=0; i<5; i++) f.tick()
  assert.equal(f.ready, 1)
})

test('음성 오류·응답 없음·지원 실패 시 자동 녹화하지 않는다', () => {
  for (const mode of ['error', 'timeout', 'exception']) {
    const f = setup()
    if (mode === 'exception') f.synthesis.speak = () => { throw new Error('unsupported') }
    f.narrator.start('Question')
    if (mode === 'error') f.utterances[0].onerror()
    if (mode === 'timeout') assert.equal(f.tick(), 120000)
    assert.equal(f.errors.length, 1); assert.equal(f.ready, 0); assert.equal(f.timers.size, 0)
  }
})

test('음성 목록이 아직 없어도 언어를 지정해 기본 영어 음성을 요청한다', () => {
  const f = setup(); f.synthesis.getVoices = () => []
  f.narrator.start('Question')
  assert.equal(f.utterances[0].lang, 'en-US'); assert.equal(f.errors.length, 0)
})
