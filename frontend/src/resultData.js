// 시연용 임의 점수입니다. 녹화 영상이나 MLP가 계산한 값이 아닙니다.
export const demoScores = {
  interview_score: 78,
  answer_score: 82,
  speaking_skills: 74,
  confidence_score: 70,
  facial_expression: 80,
  overall_performance: 77,
}

export const scoreFields = [
  { key: 'interview_score', label: '면접 점수' },
  { key: 'answer_score', label: '답변 내용' },
  { key: 'speaking_skills', label: '말하기 능력' },
  { key: 'confidence_score', label: '자신감' },
  { key: 'facial_expression', label: '표정' },
  { key: 'overall_performance', label: '전체 수행도' },
]

// UI preview only: same response shape as the backend, with explicit dummy labels.
export function createDemoResult() {
  const prompts = [
    'Tell me about yourself.',
    'What are your strengths?',
    'Tell me about a time you worked in a team.',
    'Describe a challenge you overcame.',
    'How do you adapt to change?',
  ]
  const answers = prompts.map((question, offset) => ({
    question_index: offset + 1,
    question,
    filename: `demo_answer_${offset + 1}.webm`,
    scores: Object.fromEntries(scoreFields.map(({ key }, position) => [key, {
      raw: Number(((offset - 2) * 0.1 + position * 0.02).toFixed(4)),
      percentile: 45 + offset * 5 + position,
      band: '중',
    }])),
    transcript: `[더미] 질문 ${offset + 1}의 화면 확인용 답변입니다. 실제 음성 인식 결과가 아닙니다.`,
  }))
  const averages = Object.fromEntries(scoreFields.map(({ key }) => [key, {
    raw: Number((answers.reduce((sum, answer) => sum + answer.scores[key].raw, 0) / 5).toFixed(4)),
    percentile: answers.reduce((sum, answer) => sum + answer.scores[key].percentile, 0) / 5,
    band: '중',
  }]))
  return {
    status: 'success',
    is_dummy: true,
    provider: 'ui_preview',
    video_count: 5,
    overall_score: averages.overall_performance,
    average_scores: averages,
    answers,
    combined_transcript: answers.map((answer) => `질문 ${answer.question_index}: ${answer.transcript}`).join('\n\n'),
    explanation: '화면 개발용 가짜 결과입니다. 실제 모델 분석은 실행되지 않았습니다.',
  }
}
