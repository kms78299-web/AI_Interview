const scoreLabels = {
  interview_score: '면접 점수',
  answer_score: '답변 점수',
  speaking_skills: '말하기 능력',
  confidence_score: '자신감',
  facial_expression: '표정',
  overall_performance: '종합 평가',
}

function hasPercentile(score) {
  return typeof score?.percentile === 'number'
    && Number.isFinite(score.percentile)
    && score.percentile >= 0
    && score.percentile <= 100
}

function Percentile({ score, average = false }) {
  if (!hasPercentile(score)) return <span className="score-unavailable">계산되지 않음</span>
  return (
    <>
      <strong>{new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 1 }).format(score.percentile)}</strong>
      <small> {average ? '평균 백분위' : '백분위'}</small>
    </>
  )
}

function Band({ score }) {
  if (!hasPercentile(score) || !score?.band) return null
  return <span className="score-band">{score.band}</span>
}

export default function ResultView({ result, questions, onReview, onHome }) {
  const isDummy = result?.is_dummy === true
  const overall = result?.overall_score

  return (
    <section className="content result-content" aria-labelledby="result-title">
      <span className={`demo-label${isDummy ? '' : ' actual-label'}`}>
        {isDummy ? '데모 결과 · 가짜 데이터' : '모델 분석 결과'}
      </span>
      <h1 id="result-title">면접 분석 결과</h1>
      <p className="demo-note">
        {isDummy
          ? '화면 확인용 예시입니다. 영상 분석과 음성 인식은 실행되지 않았습니다.'
          : '답변 영상 5개의 분석 결과입니다.'}
      </p>

      <article className="score-card overall-card" aria-label="전체 종합평가">
        <span className="score-label">종합 평가</span>
        <Percentile score={overall} average />
        <Band score={overall} />
      </article>
      <p className="aggregation-note">
        전체 수치는 질문별 백분위의 평균입니다. 100점 만점 점수나 면접 세션 전체의 백분위를 뜻하지 않습니다.
      </p>

      <div className="score-grid" aria-label="6개 항목의 평균 백분위">
        {Object.entries(scoreLabels).map(([key, label]) => {
          const score = result?.average_scores?.[key]
          return (
            <article className="score-card" key={key}>
              <span className="score-label">{label}</span>
              <Percentile score={score} average />
              <Band score={score} />
            </article>
          )
        })}
      </div>

      {result?.explanation && (
        <section className="feedback" aria-labelledby="score-explanation-title">
          <h2 id="score-explanation-title">점수 안내</h2>
          <p>{result.explanation}</p>
        </section>
      )}

      <section className="feedback" aria-labelledby="answer-score-title">
        <h2 id="answer-score-title">질문별 세부 결과</h2>
        <div className="answer-score-list">
          {result?.answers?.map((answer) => {
            const question = questions.find((item) => item.id === answer.question_id)
              || questions[answer.question_index - 1]
            return (
              <details className="answer-score-card" key={answer.question_index}>
                <summary>
                  <span>질문 {answer.question_index} · {answer.question || question?.text || answer.filename}</span>
                  <span className="answer-summary-score">
                    <Percentile score={answer.scores?.overall_performance} />
                    <Band score={answer.scores?.overall_performance} />
                  </span>
                </summary>
                <dl className="compact-score-grid">
                  {Object.entries(scoreLabels).map(([key, label]) => (
                    <div key={key}>
                      <dt>{label}</dt>
                      <dd>
                        <Percentile score={answer.scores?.[key]} />
                        <Band score={answer.scores?.[key]} />
                      </dd>
                    </div>
                  ))}
                </dl>
                <h3 className="transcript-title">답변 텍스트</h3>
                <p className="transcript-text">
                  {answer.transcript?.trim() || '인식된 답변 텍스트가 없습니다.'}
                </p>
              </details>
            )
          })}
        </div>
      </section>

      {result?.combined_transcript && (
        <details className="feedback transcript-details">
          <summary>전체 답변 텍스트 보기</summary>
          <p className="transcript-text">{result.combined_transcript}</p>
        </details>
      )}

      <div className="result-actions">
        {onReview && (
          <button className="secondary-button" type="button" onClick={onReview}>답변 다시 보기</button>
        )}
        <button type="button" onClick={onHome}>처음으로</button>
      </div>
    </section>
  )
}
