"""
LLM 프롬프트 검증 2단계: llm_eval_prepare.py가 저장한 사람별 A/B/C 프롬프트를 Gemini에 보내고 원문 응답을 저장한다.

  A = session.llm_prompt_compare.no_model      (모델 정보 없음)
  B = session.llm_prompt_compare.model_output  (모델 결과 + 학습 정보)
  C = session.llm_prompt                       (B + 성능·한계·사용 제한, 서비스용)

반복마다 조건 순서를 바꾼다(ABC, BCA, CAB). JSON 파싱 실패는 한 번만 다시 요청하고, 429/5xx는 기다렸다 재시도한다.
응답을 고치거나 버리지 않고 그대로 저장한다. 이미 성공한 (사람, 조건, 반복)은 건너뛴다(이어 돌리기 가능).
영상이 필요 없어서 아무 노드(로그인 노드 포함)에서 돌려도 된다. API 키는 환경변수 GEMINI_API_KEY로만 읽는다.

  export GEMINI_API_KEY=...
  python code/llm_eval_gemini.py --split val --limit 2      # 소량 확인
  python code/llm_eval_gemini.py --split val                # -> llm_eval/val/gemini/{user}_{A|B|C}_{rep}.json
"""

import argparse
import glob
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ORDERS = ["ABC", "BCA", "CAB"]


def prompt_of(result, cond):
    s = result["session"]
    return {"A": s["llm_prompt_compare"]["no_model"], "B": s["llm_prompt_compare"]["model_output"], "C": s["llm_prompt"]}[cond]


def call(prompt, model, temperature, key, timeout=180, max_tries=6):
    """한 번의 생성 요청. 429/5xx/네트워크 오류는 지수 대기 후 재시도. -> (응답 텍스트 또는 None, 시도 기록)"""
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}}
    log = []
    for k in range(max_tries):
        t0 = time.time()
        try:
            r = requests.post(API.format(model=model), json=body, timeout=timeout,
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"})
            entry = {"http": r.status_code, "sec": round(time.time() - t0, 2)}
            if r.status_code == 200:
                data = r.json()
                parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                entry["finish_reason"] = (data.get("candidates") or [{}])[0].get("finishReason")
                entry["usage"] = data.get("usageMetadata")
                log.append(entry)
                return text, log
            entry["error"] = r.text[:300]
            log.append(entry)
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                return None, log  # 400/403 등은 재시도해도 같다
        except requests.RequestException as e:
            log.append({"http": None, "sec": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {e}"[:300]})
        time.sleep(min(60, 2 ** k * 2) + random.random())
    return None, log


def run_one(job, args, key):
    user, cond, rep, prompt, path = job
    attempts, parsed, text, parse_error = [], None, None, None
    for _ in range(2):  # JSON 파싱 실패에 한해 한 번 더
        text, log = call(prompt, args.model, args.temperature, key)
        attempts.append(log)
        if text is None:
            parse_error = "API 실패"
            break
        try:
            parsed = json.loads(text)
            parse_error = None
            break
        except json.JSONDecodeError as e:
            parse_error = f"JSON 파싱 실패: {e}"
    rec = {"user_no": user, "condition": cond, "repeat": rep, "model": args.model, "temperature": args.temperature,
           "ok": parsed is not None, "parse_error": parse_error, "raw_text": text, "parsed": parsed, "attempts": attempts,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite", help="팀원 v1 실험과 같은 모델")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--conditions", default="ABC")
    ap.add_argument("--workers", type=int, default=4, help="동시 요청 수 (429가 많으면 줄인다)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N명만 (확인용)")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "llm_eval"))
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("환경변수 GEMINI_API_KEY가 없음")

    base = os.path.join(args.out_dir, args.split)
    files = sorted(glob.glob(os.path.join(base, "analysis", "*.json")), key=lambda p: int(os.path.basename(p)[:-5]))
    if args.limit:
        files = files[:args.limit]
    out = os.path.join(base, "gemini")
    os.makedirs(out, exist_ok=True)

    jobs = []
    for rep in range(1, args.repeats + 1):  # 반복 단위로 묶고, 반복마다 조건 순서를 돌린다
        order = [c for c in ORDERS[(rep - 1) % 3] if c in args.conditions]
        for f in files:
            d = json.load(open(f, encoding="utf-8"))
            for cond in order:
                path = os.path.join(out, f"{d['user_no']}_{cond}_{rep}.json")
                if os.path.exists(path) and json.load(open(path, encoding="utf-8")).get("ok"):
                    continue
                jobs.append((d["user_no"], cond, rep, prompt_of(d["result"], cond), path))
    print(f"{args.split}: 사람 {len(files)}명, 남은 요청 {len(jobs)}개 (모델 {args.model}, temperature {args.temperature}, "
          f"동시 {args.workers})", flush=True)

    done, fails, lock, t0 = 0, 0, threading.Lock(), time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, j, args, key) for j in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                done += 1
                fails += not rec["ok"]
                if done % 10 == 0 or not rec["ok"] or done == len(jobs):
                    print(f"  {done}/{len(jobs)} (실패 {fails}) {time.time() - t0:.0f}초"
                          + ("" if rec["ok"] else f" - {rec['user_no']} {rec['condition']}{rec['repeat']}: {rec['parse_error']}"),
                          flush=True)
    print("완료. 실패한 요청은 같은 명령을 다시 실행하면 그것만 다시 보낸다.")


if __name__ == "__main__":
    main()
