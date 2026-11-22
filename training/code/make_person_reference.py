"""
세션(한 사람의 답변 여러 개) 단위 '전달 인상' 기준표(person_reference.json) 생성.

모델은 같은 사람의 답변끼리 비교하는 데는 약하지만(사람 안 Spearman 약 0.17), 답변을 평균낸 사람 단위로는 잘 맞는다(약 0.75).
그래서 사람별 평균 예측을 RecruitView 참가자 331명의 분포와 비교해 상/중/하를 낸다.
라벨과 모델이 모두 '평소 짧게 말하는 사람'을 높게 보므로, 사람 평균 길이(log 단어 수)로 회귀한 잔차 기준(길이 보정)도 함께 저장한다.
검증은 train 사람으로 기준을 만들고 val+test 사람(학습에 안 쓴 99명)에 적용해 정답 상/중/하와 비교한다.

사용: python make_person_reference.py [--model_dir ...]
"""

import argparse
import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from config import DEFAULT_FEATURES_DIR, DEFAULT_METADATA_PATH, DEFAULT_SPLITS_DIR, INTERVIEW_LABEL_NAMES as T, SPLIT_SEED
from data_utils import clean_transcript, load_metadata
from train_mini_crmf import MiniCRMF

WORD_RE = re.compile(r"[A-Za-z가-힣']+")


def log_words(text):
    return float(np.log1p(len(WORD_RE.findall(text or ""))))


def predict(model_dir, ids):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    scalers = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    models = []
    for m in json.load(open(os.path.join(model_dir, "ensemble.json")))["members"]:
        net = MiniCRMF(cfg["modality_dims"], cfg["d_model"], cfg["fusion"], T, dropout=cfg["dropout"],
                       n_transformer_layers=cfg["transformer_layers"], n_heads=cfg["n_heads"])
        net.load_state_dict(torch.load(os.path.join(model_dir, m["file"]), map_location="cpu"))
        models.append(net.eval())
    F = DEFAULT_FEATURES_DIR
    with ThreadPoolExecutor(48) as ex:
        load = lambda paths: np.stack(list(ex.map(np.load, paths)))  # noqa: E731
        feats = {"text": load([f"{F}/embed/bert-base-uncased/{i}.npy" for i in ids]),
                 "video": load([f"{F}/embed/videomae/{i}.npy" for i in ids]),
                 "audio": load([f"{F}/embed_layers/audio/wav2vec2/{i}.npy" for i in ids])[:, 0, cfg["audio_layer"]]}
    batch = {k: torch.from_numpy(scalers[k].transform(v).astype(np.float32)) for k, v in feats.items()}
    with torch.no_grad():
        return np.mean([m(batch).numpy() for m in models], axis=0)


def fit_reference(person):
    """사람 단위 표(raw 평균, 평균 log 단어 수) -> 타깃별 길이 회귀 계수와 상/중/하 경계."""
    ref = {}
    for t in T:
        b, a = np.polyfit(person["len"], person[t], 1)
        adj = person[t] - (a + b * person["len"])
        ref[t] = {"cuts": np.quantile(person[t], [1 / 3, 2 / 3]).tolist(),
                  "len_coef": [float(a), float(b)],
                  "cuts_adjusted": np.quantile(adj, [1 / 3, 2 / 3]).tolist()}
    return ref


def band(value, cuts):
    return "하" if value <= cuts[0] else ("중" if value <= cuts[1] else "상")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/ceph_data/kms78299/model/checkpoints/final_audio6_R1_ens5")
    args = ap.parse_args()

    items = {it["id"]: it for it in load_metadata(DEFAULT_METADATA_PATH)}
    split = json.load(open(os.path.join(DEFAULT_SPLITS_DIR, f"split_seed{SPLIT_SEED}.json")))
    ids = split["train"] + split["val"] + split["test"]
    pred = predict(args.model_dir, ids)
    df = pd.DataFrame(pred, columns=T)
    df["user"] = [items[i]["user_no"] for i in ids]
    df["len"] = [log_words(clean_transcript(items[i].get("transcript"))) for i in ids]
    df["heldout"] = [i not in set(split["train"]) for i in ids]
    for t in T:
        df["y_" + t] = [items[i][t] for i in ids]
    person = df.groupby("user").mean(numeric_only=True)
    person["heldout"] = df.groupby("user")["heldout"].first()

    # 검증: train 사람으로 기준 -> 학습에 안 쓴 사람에 적용
    tr, te = person[~person["heldout"]], person[person["heldout"]]
    ref_tr = fit_reference(tr)
    print(f"[검증] 기준: train {len(tr)}명 / 평가: 학습에 안 쓴 {len(te)}명 (정답 상/중/하는 전체 331명의 사람 평균 라벨 3등분)")
    for t in T:
        y_cuts = np.quantile(person["y_" + t], [1 / 3, 2 / 3])
        y_band = [band(v, y_cuts) for v in te["y_" + t]]
        r = ref_tr[t]
        raw_band = [band(v, r["cuts"]) for v in te[t]]
        adj = te[t] - (r["len_coef"][0] + r["len_coef"][1] * te["len"])
        adj_band = [band(v, r["cuts_adjusted"]) for v in adj]
        agree = lambda b: np.mean([x == y for x, y in zip(b, y_band)])  # noqa: E731
        flip = lambda b: np.mean([{x, y} == {"상", "하"} for x, y in zip(b, y_band)])  # noqa: E731
        print(f"  {t:20s} 원래: 일치 {agree(raw_band):.0%}, 상↔하 {flip(raw_band):.0%}, Spearman {spearmanr(te[t], te['y_' + t]).statistic:.2f}"
              f" | 길이 보정: 일치 {agree(adj_band):.0%}, 상↔하 {flip(adj_band):.0%}, Spearman {spearmanr(adj, te['y_' + t]).statistic:.2f}")
    short = te["len"] < np.log1p(40)
    r = ref_tr["overall_performance"]
    adj = te["overall_performance"] - (r["len_coef"][0] + r["len_coef"][1] * te["len"])
    print(f"  평균 40단어 미만 사람 {short.sum()}명의 overall '상' 비율: 원래 {np.mean([band(v, r['cuts']) == '상' for v in te.loc[short, 'overall_performance']]):.0%}"
          f" / 길이 보정 {np.mean([band(v, r['cuts_adjusted']) == '상' for v in adj[short]]):.0%}"
          f" / 정답 {np.mean([band(v, np.quantile(person['y_overall_performance'], [1/3, 2/3])) == '상' for v in te.loc[short, 'y_overall_performance']]):.0%}")

    # 서비스용 기준: 331명 전체
    out = {"description": "사람별(답변 평균) 모델 raw의 상/중/하 경계. adjusted = raw 평균 - (a + b * 평균 log1p(단어 수))",
           "n_persons": int(len(person)), "targets": fit_reference(person)}
    path = os.path.join(args.model_dir, "person_reference.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"[INFO] 저장: {path}")


if __name__ == "__main__":
    main()
