"""
점수(0~100 백분위) 보정 파일 생성: 모델 예측 -> "실제 점수 백분위"의 기대값과 80% 구간.

문제: 예측 순위를 그대로 백분위로 보여 주면 극단이 과장된다. 모델 예측과 실제 점수의 상관이 약 0.5라서, 예측 상위 10%인 사람들의
실제 점수는 평균 상위 24% 정도였다(test). 이 평균으로의 회귀(regression to the mean)를 반영해서 보정한다.

방법 (타깃별, 정규점수 선형 보정):
  u = Phi^-1(실제 라벨의 백분위)      # 데이터셋 전체(2011개) 라벨 분포 기준, 정규점수로 변환
  v = (예측 raw - mu) / sd             # 예측을 표준화
  u = a + b*v + e,  e ~ N(0, sigma^2)  # 최소제곱으로 a, b, sigma 추정
  기대 백분위 = 100 * Phi( (a + b*v) / sqrt(1 + sigma^2) ),   80% 구간 = 100 * Phi( a + b*v +- 1.2816*sigma )
두 변수를 정규점수로 바꾸면 이중정규(bivariate normal)에 가까워서 위 식이 닫힌 형태로 나온다. 모수가 3개(a, b, sigma)뿐이라 표본이 적어도
계단이 생기지 않고, 항상 단조 증가이며 순위(Spearman)는 바뀌지 않는다.

보정 데이터: 최종 앙상블이 학습에 쓰지 않은 val(296) + test(304) = 600개 클립(사람 약 100명)의 예측과 라벨.
(최종 파일은 이 600개 전체로 맞추고, 보정이 실제로 나아지는지는 사람 단위 5-fold로 따로 평가해서 기록한다.)
  - single : 영상 1개 예측 그대로
  - session: 같은 사람 영상끼리 평균 쪽으로 당긴 예측(a=0.5, 같은 split 안의 같은 사람 기준)
상/중/하 경계는 보정 백분위가 1/3씩 되도록 보정 데이터의 33%/67% 분위수로 정한다.

사용: python make_calibration.py [--model_dir ...]
"""

import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

from config import DEFAULT_FEATURES_DIR, DEFAULT_META_PATH, DEFAULT_SPLITS_DIR, INTERVIEW_LABEL_NAMES, SPLIT_SEED
from train_mini_crmf import MiniCRMF

Z80 = float(norm.ppf(0.9))
T = INTERVIEW_LABEL_NAMES


def load_many(paths):
    with ThreadPoolExecutor(32) as ex:  # ceph의 작은 파일 읽기는 지연이 커서 병렬로
        return list(ex.map(np.load, paths))


def fit_params(pred, u):
    mu, sd = float(pred.mean()), float(pred.std())
    v = (pred - mu) / sd
    (a, b), *_ = np.linalg.lstsq(np.c_[np.ones_like(v), v], u, rcond=None)
    sigma = float((u - (a + b * v)).std(ddof=2))
    return {"mu": mu, "sd": sd, "a": float(a), "b": float(b), "sigma": sigma}


def apply_params(p, raw):
    m = p["a"] + p["b"] * (raw - p["mu"]) / p["sd"]
    s = p["sigma"]
    return 100 * norm.cdf(m / np.sqrt(1 + s ** 2)), 100 * norm.cdf(m - Z80 * s), 100 * norm.cdf(m + Z80 * s)


def smooth(p, users, a=0.5):
    return a * p + (1 - a) * pd.DataFrame(p).groupby(users).transform("mean").to_numpy()


def evaluate_cv(pred, true_p, u, users):
    """사람 단위 5-fold: 보정 vs 기존 방식(기준 예측 분포에서의 순위) — 실제 백분위와의 평균 절대 오차, 80% 구간 포함률."""
    mae_rank, mae_cal, cover, width = [], [], [], []
    for tr, te in GroupKFold(n_splits=5).split(pred, groups=users):
        for k in range(6):
            params = fit_params(pred[tr, k], u[tr, k])
            est, lo, hi = apply_params(params, pred[te, k])
            ref = np.sort(pred[tr, k])
            rank_est = np.searchsorted(ref, pred[te, k], side="right") / len(ref) * 100
            t = true_p[te, k] * 100
            mae_rank.append(np.abs(rank_est - t).mean())
            mae_cal.append(np.abs(est - t).mean())
            cover.append(np.mean((t >= lo) & (t <= hi)))
            width.append(np.mean(hi - lo))
    return {"mae_rank_percentile": float(np.mean(mae_rank)), "mae_calibrated": float(np.mean(mae_cal)),
            "coverage_80": float(np.mean(cover)), "mean_width_80": float(np.mean(width))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/ceph_data/kms78299/model/checkpoints/final_audio6_R1_ens5")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    files = [m["file"] for m in json.load(open(os.path.join(args.model_dir, "ensemble.json")))["members"]]
    scalers = pickle.load(open(os.path.join(args.model_dir, "scaler.pkl"), "rb"))
    models = []
    for f in files:
        m = MiniCRMF(cfg["modality_dims"], cfg["d_model"], cfg["fusion"], T, dropout=cfg["dropout"],
                     n_transformer_layers=cfg["transformer_layers"], n_heads=cfg["n_heads"])
        m.load_state_dict(torch.load(os.path.join(args.model_dir, f), map_location="cpu"))
        models.append(m.eval())

    split = json.load(open(os.path.join(DEFAULT_SPLITS_DIR, f"split_seed{SPLIT_SEED}.json")))
    meta = pd.read_csv(DEFAULT_META_PATH, dtype={"id": str, "user_no": str}).set_index("id")
    pop = np.sort(meta[T].to_numpy(), axis=0)  # 데이터셋 전체 라벨 분포(기준 집단)
    F = DEFAULT_FEATURES_DIR

    per_split = {}
    for sp in ("val", "test"):
        ids = split[sp]
        feats = {"text": np.stack(load_many([f"{F}/embed/bert-base-uncased/{i}.npy" for i in ids])),
                 "video": np.stack(load_many([f"{F}/embed/videomae/{i}.npy" for i in ids])),
                 "audio": np.stack([a[0, cfg["audio_layer"]] for a in load_many(
                     [f"{F}/embed_layers/audio/wav2vec2/{i}.npy" for i in ids])])}
        batch = {m: torch.from_numpy(scalers[m].transform(v).astype(np.float32)) for m, v in feats.items()}
        with torch.no_grad():
            pred = np.mean([m(batch).numpy() for m in models], axis=0)
        y = meta.loc[ids, T].to_numpy()
        per_split[sp] = {"pred": pred, "y": y, "users": meta.loc[ids, "user_no"].to_numpy()}
        print(f"[INFO] {sp}: {len(ids)}개 예측 완료")

    users = np.concatenate([per_split[s]["users"] for s in ("val", "test")])
    y = np.concatenate([per_split[s]["y"] for s in ("val", "test")])
    true_p = np.stack([(np.searchsorted(pop[:, k], y[:, k], "left") + np.searchsorted(pop[:, k], y[:, k], "right"))
                       / 2 / len(pop) for k in range(6)], axis=1)
    u = norm.ppf(np.clip(true_p, 1e-3, 1 - 1e-3))

    out = {"method": "정규점수 선형 보정 (평균으로의 회귀): 기대 백분위 = 100*Phi((a+b*v)/sqrt(1+sigma^2)), 80% 구간 = 100*Phi(a+b*v +- 1.2816*sigma)",
           "z80": Z80, "population": f"RecruitView 전체 라벨 {len(pop)}개", "n_calibration": int(len(y)),
           "calibration_data": "최종 앙상블이 학습에 쓰지 않은 val+test 클립의 예측과 라벨"}
    for mode in ("single", "session"):
        pred = np.concatenate([per_split[s]["pred"] if mode == "single" else smooth(per_split[s]["pred"], per_split[s]["users"])
                               for s in ("val", "test")])
        targets = {}
        for k, name in enumerate(T):
            params = fit_params(pred[:, k], u[:, k])
            est = apply_params(params, pred[:, k])[0]
            params["band_cuts"] = [float(c) for c in np.quantile(est, [1 / 3, 2 / 3])]
            targets[name] = params
        cv = evaluate_cv(pred, true_p, u, users)
        out[mode] = {"targets": targets, "cv": cv}
        if mode == "session":
            out[mode]["smoothing"] = 0.5
        print(f"\n[{mode}] 사람 단위 5-fold 평가 (실제 라벨 백분위 대비)")
        print(f"  평균 절대 오차: 기존(예측 순위 그대로) {cv['mae_rank_percentile']:.1f} -> 보정 {cv['mae_calibrated']:.1f}")
        print(f"  80% 구간: 실제 포함률 {cv['coverage_80'] * 100:.0f}%, 평균 폭 {cv['mean_width_80']:.0f}점")
        print("  타깃별 b(기울기, 1에 가까울수록 예측이 믿을 만함):",
              {n: round(t['b'], 2) for n, t in targets.items()})

    with open(os.path.join(args.model_dir, "calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[INFO] 저장: {args.model_dir}/calibration.json")


if __name__ == "__main__":
    main()
