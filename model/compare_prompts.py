"""
LLM 프롬프트 비교 실험(docs/llm-prompt-comparison.md)용: 미리 찍은 영상을 Colab Gradio /analyze에 보내고
조건 A/B/C 프롬프트를 파일로 저장한다. Gemini 호출은 하지 않는다.

python model/compare_prompts.py --url https://xxxx.gradio.live --videos q1.mp4 ... --questions questions.json --out runs/s01
questions.json: 영상 순서대로 질문 문자열 리스트 (생략 가능)
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Colab gradio.live 주소")
    ap.add_argument("--videos", nargs="+", required=True, help="질문 순서대로")
    ap.add_argument("--questions", default="", help="질문 리스트 JSON 파일 (선택)")
    ap.add_argument("--out", required=True, help="결과 저장 폴더")
    args = ap.parse_args()

    from gradio_client import Client, handle_file

    questions = ""
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            q = json.load(f)
        if len(q) != len(args.videos):
            raise SystemExit(f"질문 {len(q)}개와 영상 {len(args.videos)}개의 개수가 다름")
        questions = json.dumps(q, ensure_ascii=False)

    # transcripts_json을 비워야 STT가 문장 구간과 마커까지 만든다
    result = Client(args.url).predict([handle_file(v) for v in args.videos], "", questions, True, "",
                                      api_name="/analyze")
    if isinstance(result, str):
        result = json.loads(result)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s = result["session"]
    compare = s.get("llm_prompt_compare")
    if not compare:
        raise SystemExit("응답에 llm_prompt_compare가 없음: Colab의 model 코드가 최신인지 확인")
    prompts = {"A_no_model": compare["no_model"], "B_model_output": compare["model_output"], "C_full": s["llm_prompt"]}
    for name, text in prompts.items():
        with open(os.path.join(args.out, f"prompt_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    print(f"저장: {args.out}/result.json, prompt_A_no_model.txt, prompt_B_model_output.txt, prompt_C_full.txt")


if __name__ == "__main__":
    main()
