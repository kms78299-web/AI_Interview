import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'
import test from 'node:test'
import ts from 'typescript'
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement } from 'react'

const require = createRequire(import.meta.url)
const dataModule = code => `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`
const compile = source => ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX } }).outputText
const preview = await import(dataModule(compile(await readFile(new URL('../src/feedbackPreview.ts', import.meta.url), 'utf8'))))
let source = compile(await readFile(new URL('../src/FeedbackResults.tsx', import.meta.url), 'utf8'))
source = source.replace(/import "\.\/feedbackResults\.css";?/, '')
source = source.replaceAll('"./interviewApi"', JSON.stringify(dataModule('export function answerVideoUrl(s,q){return `/api/interviews/${s}/answers/${q}/video`}')))
for (const name of ['react/jsx-runtime', 'react']) source = source.replaceAll(`"${name}"`, JSON.stringify(pathToFileURL(require.resolve(name)).href))
const { readFeedbackReport, FeedbackResults, getPreviewWindow } = await import(dataModule(source))

test('대표 영상은 시작 시점부터 최대 5초로 제한하고 더 짧은 구간은 유지한다', () => {
  assert.deepEqual(getPreviewWindow({ playback_start: 8, playback_end: 20 }), { start: 8, end: 13 })
  assert.deepEqual(getPreviewWindow({ playback_start: 0, playback_end: 8 }), { start: 0, end: 5 })
  assert.deepEqual(getPreviewWindow({ playback_start: 8, playback_end: 9.5 }), { start: 8, end: 9.5 })
})

test('새 화면 순서·대본 토글 5개·항상 보이는 대표 장면과 점수 제거', () => {
  const html = renderToStaticMarkup(createElement(FeedbackResults, { result: preview.previewResult, session: preview.previewSession }))
  assert.ok(html.indexOf('id="fr-overall-heading"') < html.indexOf('id="fr-delivery-heading"'))
  assert.ok(html.indexOf('id="fr-delivery-heading"') < html.indexOf('id="fr-content-heading"'))
  assert.equal((html.match(/<details/g) || []).length, 5)
  assert.equal((html.match(/class="fr-player"/g) || []).length, 2)
  assert.doesNotMatch(html, /백분위|대표 장면 보기|대표 장면 닫기/)
})

test('다른 세션·질문 누락·등급 불일치 결과는 표시하지 않는다', () => {
  const original = preview.previewResult.feedback_v1
  for (const mutate of [r => r.session_id = 'other', r => r.answer_feedback.pop(), r => r.overall_band = '최고', r => r.status = 'insufficient_data']) {
    const report = structuredClone(original); mutate(report)
    assert.equal(readFeedbackReport(report, preview.previewSession), null)
  }
})

test('잘못된 시간·다른 질문의 근거는 녹화에 연결하지 않는다', () => {
  const report = structuredClone(preview.previewResult.feedback_v1)
  report.evidence_segments[0].playback_start = -1
  report.evidence_segments[1].question_id = 'unrelated'
  assert.deepEqual(readFeedbackReport(report, preview.previewSession).evidence_segments, [])
})

test('종합 응답이 없을 때 기존 모델 등급으로 종합 등급을 만들지 않는다', () => {
  const result = { ...preview.previewResult, feedback_v1: undefined, session: { scores: { overall_performance: { band: '상', percentile: 99 } } } }
  const html = renderToStaticMarkup(createElement(FeedbackResults, { result, session: preview.previewSession }))
  assert.match(html, /평가 대기/)
  assert.doesNotMatch(html, />상<|백분위|99/)
})
