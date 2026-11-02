import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

const source = await readFile(new URL('../src/interviewApi.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } })
const { analyzeInterview, transcribeInterview, scoreMetrics } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)
const session = { id: 'a'.repeat(32), questions: Array.from({ length: 5 }, (_, i) => ({ id: `q${i + 1}` })) }
const scores = () => Object.fromEntries(scoreMetrics.map(({ key }) => [key, { raw: -0.12, percentile: 56, percentile_low: 20, percentile_high: 85, band: '중' }]))
const fixture = () => ({ status: 'success', session_id: session.id, is_dummy: false,
  session: { scores: scores() }, videos: session.questions.map(q => ({ question_id: q.id, transcript: `Answer for ${q.id}`, scores: scores() })) })

test('분석 API를 한 번 호출하고 모델의 band·텍스트·보정 필드를 그대로 받는다', async t => {
  const body = fixture()
  const signal = new AbortController().signal
  const fetch = t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, `/api/interviews/${session.id}/analyze`)
    assert.equal(options.method, 'POST')
    assert.equal(options.signal, signal)
    return Response.json(body)
  })
  assert.deepEqual(await analyzeInterview(session, signal), body)
  assert.equal(fetch.mock.callCount(), 1)
})

test('평가가 없는 null과 빈 STT는 임의 결과로 대체하지 않는다', async t => {
  const body = fixture()
  body.session.scores.overall_performance.band = null
  body.session.scores.overall_performance.percentile = null
  body.videos[0].transcript = ''
  t.mock.method(globalThis, 'fetch', async () => Response.json(body))
  const result = await analyzeInterview(session, new AbortController().signal)
  assert.equal(result.session.scores.overall_performance.band, null)
  assert.equal(result.session.scores.overall_performance.percentile, null)
  assert.equal(result.videos[0].transcript, '')
})

test('다른 세션·질문 순서·누락된 평가와 텍스트를 결과 화면에 연결하지 않는다', async t => {
  let body
  t.mock.method(globalThis, 'fetch', async () => Response.json(body))
  for (const mutate of [
    b => { b.session_id = 'b'.repeat(32) },
    b => { b.videos.reverse() },
    b => { b.videos.pop() },
    b => { delete b.session.scores.answer_score },
    b => { b.videos[0].scores.answer_score.band = 80 },
    b => { b.session.scores.overall_performance.percentile = 101 },
    b => { b.session.scores.overall_performance.percentile = '56' },
    b => { b.videos[0].transcript = null },
  ]) {
    body = fixture(); mutate(body)
    await assert.rejects(analyzeInterview(session, new AbortController().signal), /맞지 않거나/)
  }
})

test('분석 서버 오류와 취소는 예시 결과로 대체하지 않는다', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: '모델 서버 연결 필요' }, { status: 503 }))
  await assert.rejects(analyzeInterview(session, new AbortController().signal), /모델 서버 연결 필요/)
  globalThis.fetch.mock.mockImplementation(async () => new Response('Bad Gateway', { status: 502 }))
  await assert.rejects(analyzeInterview(session, new AbortController().signal), /분석하지 못했어요/)
  globalThis.fetch.mock.mockImplementation(async (_, { signal }) => { signal.throwIfAborted() })
  const controller = new AbortController(); controller.abort()
  await assert.rejects(analyzeInterview(session, controller.signal), { name: 'AbortError' })
})

test('STT 중간 결과를 질문 순서대로 받고 빈 답변도 보존한다', async t => {
  const body = { status: 'ready', session_id: session.id, is_dummy: false,
    answers: session.questions.map((q,i) => ({ question_id:q.id, transcript:i ? `Answer ${i}` : '' })) }
  const signal = new AbortController().signal
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, `/api/interviews/${session.id}/transcripts`)
    assert.equal(options.method, 'POST'); assert.equal(options.signal, signal)
    return Response.json(body)
  })
  assert.deepEqual(await transcribeInterview(session, signal), body)
})

test('STT의 질문 불일치와 서버 오류를 정상 텍스트로 처리하지 않는다', async t => {
  let body = { status:'ready', session_id:session.id, is_dummy:false, answers:[] }
  t.mock.method(globalThis, 'fetch', async () => Response.json(body))
  await assert.rejects(transcribeInterview(session, new AbortController().signal), /누락/)
  body.answers = session.questions.map(q => ({question_id:q.id,transcript:'Answer'})).reverse()
  await assert.rejects(transcribeInterview(session, new AbortController().signal), /맞지 않거나/)
  globalThis.fetch.mock.mockImplementation(async () => Response.json({detail:'STT 서버 연결 필요'}, {status:503}))
  await assert.rejects(transcribeInterview(session, new AbortController().signal), /STT 서버 연결 필요/)
})
