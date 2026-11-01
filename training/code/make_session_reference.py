"""
세션(같은 사람의 영상 여러 개) 점수의 0~100 변환 기준과 성능 기록을 만든다.

배경: 같은 사람 영상의 예측을 사람 평균 쪽으로 당기면(a*자기 + (1-a)*사람 평균) 순위 정확도가 올랐다(session 실험,
a=0.5를 val로 선택). 당긴 예측은 분산이 줄어서 단일 클립 기준 분포로 백분위를 계산하면 한쪽으로 쏠리므로,
"사람 평균으로 당긴 val 예측"의 분포를 별도 기준으로 저장한다 (percentile_reference_session.json).

패키지(model/checkpoints/...)의 모델 5개를 캐시된 특징으로 직접 돌려서 계산하므로, 학습 코드 경로와 독립적인 재현 검증도 겸한다
(단일 클립 앙상블 val/test가 metrics_report.json과 같아야 함).

사용: python make_session_reference.py [--model_dir ...] [--smoothing 0.5]
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from config import DEFAULT_FEATURES_DIR, DEFAULT_META_PATH, DEFAULT_SPLITS_DIR, INTERVIEW_LABEL_NAMES, SPLIT_SEED
from train_mini_crmf import MiniCRMF


def macro(p, y):
    return float(np.mean([spearmanr(p[:, i], y[:, i]).statistic for i in range(y.shape[1])]))


def smooth(p, users, a):
    return a * p + (1 - a) * pd.DataFrame(p).groupby(users).transform("mean").to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/ceph_data/kms78299/model/checkpoints/final_audio6_R1_ens5")
    ap.add_argument("--smoothing", type=float, default=0.5)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    files = [m["file"] for m in json.load(open(os.path.join(args.model_dir, "ensemble.json")))["members"]]
    scalers = pickle.load(open(os.path.join(args.model_dir, "scaler.pkl"), "rb"))
    models = []
    for f in files:
        m = MiniCRMF(cfg["modality_dims"], cfg["d_model"], cfg["fusion"], INTERVIEW_LABEL_NAMES, dropout=cfg["dropout"],
                     n_transformer_layers=cfg["transformer_layers"], n_heads=cfg["n_heads"])
        m.load_state_dict(torch.load(os.path.join(args.model_dir, f), map_location="cpu"))
        models.append(m.eval())

    split = json.load(open(os.path.join(DEFAULT_SPLITS_DIR, f"split_seed{SPLIT_SEED}.json")))
    meta = pd.read_csv(DEFAULT_META_PATH, dtype={"id": str, "user_no": str}).set_index("id")
    T = INTERVIEW_LABEL_NAMES
    out = {}
    for sp in ("val", "test"):
        ids = split[sp]
        feats = {
            "text": np.stack([np.load(f"{DEFAULT_FEATURES_DIR}/embed/bert-base-uncased/{i}.npy") for i in ids]),
            "video": np.stack([np.load(f"{DEFAULT_FEATURES_DIR}/embed/videomae/{i}.npy") for i in ids]),
            "audio": np.stack([np.load(f"{DEFAULT_FEATURES_DIR}/embed_layers/audio/wav2vec2/{i}.npy")[0, cfg["audio_layer"]]
                               for i in ids]),
        }
        batch = {m: torch.from_numpy(scalers[m].transform(v).astype(np.float32)) for m, v in feats.items()}
        with torch.no_grad():
            pred = np.mean([m(batch).numpy() for m in models], axis=0)
        out[sp] = {"pred": pred, "y": meta.loc[ids, T].to_numpy(), "users": meta.loc[ids, "user_no"].to_numpy()}

    report = {"smoothing": args.smoothing, "note": "a=0.5는 session 실험에서 val로 선택. test는 보고용 1회."}
    for sp in ("val", "test"):
        d = out[sp]
        report[f"{sp}_single_clip_macro"] = macro(d["pred"], d["y"])
        report[f"{sp}_session_smoothed_macro"] = macro(smooth(d["pred"], d["users"], args.smoothing), d["y"])
    print("[CHECK] 패키지 모델로 재계산한 단일 클립 앙상블: "
          f"val {report['val_single_clip_macro']:.4f} / test {report['test_single_clip_macro']:.4f} "
          "(metrics_report.json의 앙상블 값 val 0.4911 / test 0.5316과 같아야 함)")
    print(f"[RESULT] 사람 평균으로 당긴 후(a={args.smoothing}): val {report['val_session_smoothed_macro']:.4f} / "
          f"test {report['test_session_smoothed_macro']:.4f}")

    val_s = smooth(out["val"]["pred"], out["val"]["users"], args.smoothing)
    with open(os.path.join(args.model_dir, "percentile_reference_session.json"), "w") as f:
        json.dump({"source": "validation ensemble predictions pulled toward same-person mean", "smoothing": args.smoothing,
                   "n_samples": len(val_s),
                   "predictions_sorted": {n: sorted(float(v) for v in val_s[:, i]) for i, n in enumerate(T)}}, f)
    with open(os.path.join(args.model_dir, "session_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] 저장: {args.model_dir}/percentile_reference_session.json, session_report.json")


if __name__ == "__main__":
    main()
