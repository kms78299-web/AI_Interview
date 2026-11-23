"""
LLM 프롬프트 검증 1단계: RecruitView 사람 1명 = 세션 1개로, 서비스와 같은 analyze_session(Whisper STT, 모델, 측정값, 구간 마커)을 돌려
사람별 결과와 A/B/C 프롬프트를 저장한다. Gemini는 부르지 않는다 (2단계 llm_eval_gemini.py).

val은 프롬프트 비교·수정용, test는 최종 프롬프트로 한 번만 쓴다.
영상이 ariel-v7에만 있으므로 scripts/run_llm_eval_prepare.sh로 돌린다. 이미 저장된 사람은 건너뛴다(이어 돌리기 가능).

출력: llm_eval/{split}/analysis/{user_no}.json = {"user_no", "ids", "questions", "result"(analyze_session 결과)}
"""

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_CODE = "/ceph_data/kms78299/model/code"
# model/code와 이 폴더에 같은 이름의 모듈(config.py 등)이 있어서, 서비스 코드(model/code)를 먼저 찾게 한다
sys.path = [MODEL_CODE] + [p for p in sys.path if os.path.abspath(p or ".") != HERE]


def sessions(metadata_path, split_ids):
    rows = [json.loads(line) for line in open(metadata_path, encoding="utf-8")]
    rows = [r for r in rows if r["id"] in split_ids]
    by_user = {}
    for r in sorted(rows, key=lambda r: r["id"]):  # 같은 사람 안에서는 id 순서(녹화 순서)
        by_user.setdefault(r["user_no"], []).append(r)
    return by_user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--metadata_path", default=os.path.join(ROOT, "metadata.jsonl"))
    ap.add_argument("--video_root", default="/local_datasets/RecruitView")
    ap.add_argument("--checkpoint_dir", default="/ceph_data/kms78299/model/checkpoints/final_audio6_R1_ens5")
    ap.add_argument("--whisper_model", default="base", help="서비스(Colab)와 같게")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "llm_eval"))
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N명만 (확인용)")
    args = ap.parse_args()

    from inference import analyze_session, load_models

    split_ids = set(json.load(open(os.path.join(ROOT, "splits/split_seed42.json")))[args.split])
    by_user = sessions(args.metadata_path, split_ids)
    users = sorted(by_user, key=lambda u: int(u))
    if args.limit:
        users = users[:args.limit]
    out = os.path.join(args.out_dir, args.split, "analysis")
    os.makedirs(out, exist_ok=True)
    print(f"{args.split}: {len(users)}명, 답변 {sum(len(by_user[u]) for u in users)}개 -> {out}", flush=True)

    models = load_models(args.checkpoint_dir, args.whisper_model)
    failed = []
    for n, user in enumerate(users, 1):
        path = os.path.join(out, f"{user}.json")
        if os.path.exists(path):
            continue
        rows = by_user[user]
        videos = [os.path.join(args.video_root, r["file_name"]) for r in rows]
        questions = [r["question"] for r in rows]
        t0 = time.time()
        try:
            result = analyze_session(videos, questions=questions, models=models, checkpoint_dir=args.checkpoint_dir)
        except Exception as e:
            failed.append({"user_no": user, "error": f"{type(e).__name__}: {e}"})
            print(f"[{n}/{len(users)}] {user} 실패: {e}", flush=True)
            traceback.print_exc()
            continue
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"user_no": user, "ids": [r["id"] for r in rows], "questions": questions, "result": result},
                      f, ensure_ascii=False, default=float)
        os.replace(path + ".tmp", path)
        print(f"[{n}/{len(users)}] {user}: 답변 {len(rows)}개, {time.time() - t0:.0f}초", flush=True)

    if failed:
        with open(os.path.join(args.out_dir, args.split, "prepare_failed.json"), "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
    print(f"완료. 실패 {len(failed)}명", flush=True)


if __name__ == "__main__":
    main()
