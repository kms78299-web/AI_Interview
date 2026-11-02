import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import ts from 'typescript'

const source = await readFile(new URL('../src/interviewApi.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
})
const { createInterview, uploadAnswer } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)
const bank = JSON.parse(await readFile(new URL('../../backend/questions.json', import.meta.url), 'utf8'))
const body = {
  session_id: '0123456789abcdef0123456789abcdef',
  questions: [bank.required, ...Object.values(bank.categories).map((items) => items[0])]
    .map((question, index) => ({ ...question, order: index + 1 })),
}

test('답변 영상과 녹화 시간을 해당 세션·질문에 제출한다', async (t) => {
  const signal = new AbortController().signal
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, `/api/interviews/${body.session_id}/answers/intro`)
    assert.equal(options.method, 'POST')
    assert.equal(options.signal, signal)
    assert.equal(options.body.get('duration_seconds'), '4.25')
    assert.equal(options.body.get('video').type, 'video/webm')
    assert.equal(await options.body.get('video').text(), 'recorded-answer')
    return Response.json({ status: 'saved', session_id: body.session_id, question_id: 'intro' })
  })
  await uploadAnswer(body.session_id, 'intro', new Blob(['recorded-answer'], { type: 'video/webm' }), 4.25, signal)
})

test('저장 실패 또는 다른 질문의 저장 응답을 성공으로 처리하지 않는다', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: '영상은 100MB까지 업로드할 수 있습니다.' }, { status: 413 }))
  const send = () => uploadAnswer(body.session_id, 'intro', new Blob(['video']), 2, new AbortController().signal)
  await assert.rejects(send(), /100MB/)
  globalThis.fetch.mock.mockImplementation(async () => Response.json({ status: 'saved', session_id: body.session_id, question_id: 'other' }))
  await assert.rejects(send(), /저장 완료를 확인하지 못했어요/)
})

test('서버 질문과 ID를 보존하고 카테고리 이름을 화면에 맞춘다', async (t) => {
  const signal = new AbortController().signal
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/interviews')
    assert.equal(options.method, 'POST')
    assert.equal(options.signal, signal)
    return Response.json(body)
  })
  const session = await createInterview(signal)
  assert.equal(session.id, body.session_id)
  assert.equal(session.questions.length, 5)
  assert.equal(session.questions[1].category, '자기이해 · 직무 가치관')
  assert.deepEqual(session.questions.map((q) => q.id), body.questions.map((q) => q.id))
  assert.deepEqual(session.questions.map((q) => q.question), body.questions.map((q) => q.text))
})

test('서버 실패 시 예시 질문으로 넘어가지 않고 오류를 전달한다', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: '질문 서버 점검 중' }, { status: 503 }))
  await assert.rejects(createInterview(new AbortController().signal), /질문 서버 점검 중/)
})

test('프록시의 JSON이 아닌 오류도 사용자에게 설명할 수 있다', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('Bad Gateway', { status: 502 }))
  await assert.rejects(createInterview(new AbortController().signal), /질문을 불러오지 못했습니다/)
})

test('누락된 질문이나 잘못된 순서의 응답을 면접에 사용하지 않는다', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ ...body, questions: body.questions.slice(1) }))
  await assert.rejects(createInterview(new AbortController().signal), /면접 정보가 올바르지/)
  globalThis.fetch.mock.mockImplementation(async () => Response.json({ ...body, questions: [...body.questions].reverse() }))
  await assert.rejects(createInterview(new AbortController().signal), /질문 정보가 올바르지/)
})

test('질문 ID 중복과 요청 취소를 정상 결과로 처리하지 않는다', async (t) => {
  const duplicate = structuredClone(body)
  duplicate.questions[1].id = duplicate.questions[0].id
  t.mock.method(globalThis, 'fetch', async () => Response.json(duplicate))
  await assert.rejects(createInterview(new AbortController().signal), /질문 정보가 중복/)
  globalThis.fetch.mock.mockImplementation(async (_url, { signal }) => {
    signal.throwIfAborted()
    return Response.json(body)
  })
  const controller = new AbortController()
  controller.abort()
  await assert.rejects(createInterview(controller.signal), { name: 'AbortError' })
})
