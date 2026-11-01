"""
run_inference가 학습 파이프라인과 같은 결과를 내는지 확인 (Colab 또는 로컬에서 실행).

1단계(정확 비교): 학습 때와 같은 대본을 넘겨서(STT 생략) 영상+음성 경로와 모델/앙상블/scaler가 맞는지 확인.
2단계(실사용): STT(Whisper)를 켜서 실제 사용과 같은 경로로 돌리고, 1단계와 얼마나 달라지는지 확인.

사용 (Colab):
    !python /content/drive/MyDrive/model_parity/parity_check.py \
        --model_root /content/drive/MyDrive/model --parity_dir /content/drive/MyDrive/model_parity
"""

import argparse
import json
import os
import sys

import numpy as np

TOL_RAW = 0.02  # 1단계 통과 기준: 타깃별 raw 점수 차이의 최댓값 (GPU/하드웨어 차이 정도는 허용)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_root", required=True, help="model/ 폴더 (code/, checkpoints/ 가 있는 곳)")
    p.add_argument("--parity_dir", required=True, help="model_parity/ 폴더 (parity_reference.json, videos/)")
    p.add_argument("--whisper_model", default="base")
    p.add_argument("--skip_stt", action="store_true", help="2단계(Whisper) 생략")
    args = p.parse_args()

    sys.path.insert(0, os.path.join(args.model_root, "code"))
    from inference import run_inference

    ref = json.load(open(os.path.join(args.parity_dir, "parity_reference.json"), encoding="utf-8"))
    names = None
    fails = 0

    print("=" * 80)
    print("1단계: 학습 때와 같은 대본 사용 (STT 생략) — 기대값과 정확히 비교")
    for e in ref:
        res = run_inference(os.path.join(args.parity_dir, e["video"]), whisper_model=args.whisper_model,
                            transcript=e["transcript"])
        names = list(res["scores"])
        got = np.array([res["scores"][n]["raw"] for n in names])
        exp = np.array([e["expected_raw"][n] for n in names])
        diff = float(np.abs(got - exp).max())
        pct_got = [res["scores"][n]["rank_percentile"] for n in names]
        pct_exp = [e["expected_percentile"][n] for n in names]
        ok = diff <= TOL_RAW
        fails += not ok
        print(f"  id={e['id']}: raw 최대 차이 {diff:.4f} -> {'PASS' if ok else 'FAIL'} | "
              f"백분위 예상 {pct_exp} / 실제 {pct_got}")

    if not args.skip_stt:
        print("=" * 80)
        print("2단계: Whisper STT 포함 (실제 사용 경로) — 1단계와 얼마나 달라지는지 (통과/실패 판정 없음)")
        for e in ref:
            res = run_inference(os.path.join(args.parity_dir, e["video"]), whisper_model=args.whisper_model)
            got = np.array([res["scores"][n]["raw"] for n in names])
            exp = np.array([e["expected_raw"][n] for n in names])
            print(f"  id={e['id']}: raw 최대 차이 {float(np.abs(got - exp).max()):.4f}")
            print(f"     학습 대본: {e['transcript'][:90]}")
            print(f"     Whisper : {res['transcript'][:90]}")

    print("=" * 80)
    print(f"결과: 1단계 {len(ref) - fails}/{len(ref)} PASS" + ("  <- 모두 통과하면 영상/음성 경로와 모델 로딩이 학습과 일치" if not fails else
          "  <- FAIL이 있으면 전처리 버전(transformers 등) 불일치를 먼저 의심"))


if __name__ == "__main__":
    main()
