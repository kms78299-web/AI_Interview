"""
전달력 전용 모델 실험: text 없이 audio+video만으로 speaking/confidence/facial을 학습하고, 최종 모델과 길이 의존을 비교한다.

--label_mode raw          : 원래 라벨로 학습
--label_mode length_resid : 라벨에서 답변 길이(log 단어 수)로 설명되는 부분을 뺀 잔차로 학습 (회귀 계수는 train으로만 추정)
구조와 규제는 최종 모델(train_final_ensemble.py, R1)과 같고 seed 5개 앙상블이다. val로 early stopping, test는 마지막에 한 번 본다.

비교 지표 (학습에 안 쓴 val/test, 앙상블 기준):
  spearman_raw      원래 라벨과의 Spearman
  spearman_resid    길이 잔차 라벨과의 Spearman
  corr_length       예측과 log 단어 수의 Spearman (0에 가까울수록 길이에 덜 기댐)
  within_person     같은 사람 답변끼리(사람 평균을 뺀 값) 라벨과의 Spearman
  person_level      사람 평균끼리 라벨과의 Spearman

    python train_delivery.py --smoke                        # 동작 확인용 (seed 1개, 2 epoch, 결과 해석 금지)
    python train_delivery.py --label_mode raw
    python train_delivery.py --label_mode length_resid
"""

import argparse
import copy
import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from config import (DEFAULT_CHECKPOINTS_DIR, DEFAULT_FEATURES_DIR, DEFAULT_META_PATH, DEFAULT_METADATA_PATH,
                    DEFAULT_SPLITS_DIR, INTERVIEW_LABEL_NAMES, SPLIT_SEED)
from data_utils import clean_transcript, load_metadata
from train_mini_crmf import (EmbedDataset, MiniCRMF, build_or_load_split, check_no_user_overlap, set_seed,
                             soft_winsorize)

SPLITS = ("train", "val", "test")
AUDIO_LAYER = 6
FINAL_MODEL_DIR = "/ceph_data/kms78299/model/checkpoints/final_audio6_R1_ens5"
WORD_RE = re.compile(r"[A-Za-z']+")


class HP:
    """최종 모델과 같은 설정 (규제 R1)."""
    d_model = 256
    dropout = 0.3
    weight_decay = 5e-2
    lr = 1e-3
    transformer_layers = 1
    n_heads = 4
    batch_size = 32
    max_epochs = 30
    patience = 5


def parse_args():
    p = argparse.ArgumentParser(description="전달력 전용(audio+video) 모델 학습과 길이 의존 비교")
    p.add_argument("--modalities", nargs="+", default=["audio", "video"], choices=["text", "audio", "video"])
    p.add_argument("--targets", nargs="+", default=["speaking_skills", "confidence_score", "facial_expression"],
                   choices=INTERVIEW_LABEL_NAMES)
    p.add_argument("--label_mode", default="raw", choices=["raw", "length_resid"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--exp_name", default=None, help="기본값: delivery_{modalities}_{label_mode}")
    p.add_argument("--final_model_dir", default=FINAL_MODEL_DIR, help="비교할 최종 모델 (없으면 비교 생략)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def load_features(ids):
    F = DEFAULT_FEATURES_DIR
    with ThreadPoolExecutor(32) as ex:  # ceph의 작은 파일 읽기는 지연이 커서 병렬로
        load = lambda paths: np.stack(list(ex.map(np.load, paths)))  # noqa: E731
        return {"text": load([f"{F}/embed/bert-base-uncased/{i}.npy" for i in ids]),
                "audio": load([f"{F}/embed_layers/audio/wav2vec2/{i}.npy" for i in ids])[:, 0, AUDIO_LAYER],
                "video": load([f"{F}/embed/videomae/{i}.npy" for i in ids])}


def predict(model, feats, device, batch_size=256):
    model.eval()
    n = next(iter(feats.values())).shape[0]
    out = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            out.append(model({m: torch.from_numpy(v[s:s + batch_size]).to(device) for m, v in feats.items()}).cpu().numpy())
    return np.concatenate(out)


def train_seed(seed, hp, dims, targets, train_x, train_y, val_x, val_y, device):
    """최종 모델과 같은 학습 루프: Huber, AdamW, OneCycleLR, val macro Spearman으로 early stopping 후 최고 epoch 복원."""
    set_seed(seed)
    model = MiniCRMF(dims, hp.d_model, "prefusion", targets, dropout=hp.dropout,
                     n_transformer_layers=hp.transformer_layers, n_heads=hp.n_heads).to(device)
    ds = EmbedDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=hp.batch_size, shuffle=True, drop_last=len(ds) > hp.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp.lr, total_steps=max(1, len(loader)) * hp.max_epochs,
                                                pct_start=0.15)
    huber = nn.HuberLoss(delta=1.0)
    best, best_state, wait = -np.inf, None, 0
    for epoch in range(hp.max_epochs):
        model.train()
        for xb, yb in loader:
            loss = huber(model({k: v.to(device) for k, v in xb.items()}), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        score = float(np.mean([spearmanr(p, y).statistic for p, y in zip(predict(model, val_x, device).T, val_y.T)]))
        print(f"  [seed={seed}] epoch={epoch:02d} val_spearman_macro={score:.4f}")
        if score > best:
            best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= hp.patience:
                break
    model.load_state_dict(best_state)
    return model, best


def evaluate(pred, labels_raw, labels_resid, log_words, users):
    """타깃별 지표와 macro 평균."""
    out = {}
    for k in range(pred.shape[1]):
        df = pd.DataFrame({"p": pred[:, k], "y": labels_raw[:, k], "u": users})
        w = df[df.groupby("u")["p"].transform("size") >= 3].copy()
        for c in ("p", "y"):
            w[c] = w[c] - w.groupby("u")[c].transform("mean")
        person = df.groupby("u")[["p", "y"]].mean()
        out[k] = {"spearman_raw": spearmanr(pred[:, k], labels_raw[:, k]).statistic,
                  "spearman_resid": spearmanr(pred[:, k], labels_resid[:, k]).statistic,
                  "corr_length": spearmanr(pred[:, k], log_words).statistic,
                  "within_person": spearmanr(w["p"], w["y"]).statistic,
                  "person_level": spearmanr(person["p"], person["y"]).statistic}
    macro = {m: float(np.mean([out[k][m] for k in out])) for m in out[0]}
    return {k: {m: float(v) for m, v in d.items()} for k, d in out.items()}, macro


def final_model_predict(model_dir, raw_feats, target_idx, device):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    scalers = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    x = {m: scalers[m].transform(raw_feats[m]).astype(np.float32) for m in ("text", "audio", "video")}
    preds = []
    for member in json.load(open(os.path.join(model_dir, "ensemble.json")))["members"]:
        net = MiniCRMF(cfg["modality_dims"], cfg["d_model"], cfg["fusion"], INTERVIEW_LABEL_NAMES, dropout=cfg["dropout"],
                       n_transformer_layers=cfg["transformer_layers"], n_heads=cfg["n_heads"]).to(device)
        net.load_state_dict(torch.load(os.path.join(model_dir, member["file"]), map_location=device))
        preds.append(predict(net, x, device)[:, target_idx])
    return np.mean(preds, axis=0)


def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    hp = HP()
    seeds = args.seeds[:1] if args.smoke else args.seeds
    if args.smoke:
        hp.max_epochs = 2
    name = args.exp_name or f"delivery_{'-'.join(args.modalities)}_{args.label_mode}"
    out_dir = os.path.join(DEFAULT_CHECKPOINTS_DIR, name + ("_smoke" if args.smoke else ""))
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(DEFAULT_META_PATH, dtype={"id": str, "user_no": str})
    split = build_or_load_split(meta, DEFAULT_SPLITS_DIR, seed=SPLIT_SEED)
    check_no_user_overlap(meta, split)
    meta = meta.set_index("id")
    transcripts = {it["id"]: clean_transcript(it.get("transcript")) for it in load_metadata(DEFAULT_METADATA_PATH)}

    data = {}
    for s in SPLITS:
        ids = split[s]
        data[s] = {"ids": ids, "raw_feats": load_features(ids), "users": meta.loc[ids, "user_no"].to_numpy(),
                   "y": meta.loc[ids, args.targets].to_numpy(dtype=np.float64),
                   "len": np.array([np.log1p(len(WORD_RE.findall(transcripts.get(i, "")))) for i in ids])}
        print(f"[INFO] {s}: {len(ids)}개")

    # 길이 잔차 라벨: train으로 y = a + b * log 단어 수를 맞추고, 모든 split에 같은 계수로 적용
    coefs = [np.polyfit(data["train"]["len"], data["train"]["y"][:, k], 1).tolist() for k in range(len(args.targets))]
    for s in SPLITS:
        d = data[s]
        d["y_resid"] = np.stack([d["y"][:, k] - np.polyval(coefs[k], d["len"]) for k in range(len(args.targets))], axis=1)

    scalers = {m: StandardScaler().fit(data["train"]["raw_feats"][m]) for m in args.modalities}
    for s in SPLITS:
        data[s]["x"] = {m: scalers[m].transform(data[s]["raw_feats"][m]).astype(np.float32) for m in args.modalities}
    y_key = "y" if args.label_mode == "raw" else "y_resid"
    y_tr = data["train"][y_key]
    mean, std = y_tr.mean(axis=0), y_tr.std(axis=0)
    train_target = soft_winsorize((y_tr - mean) / np.where(std < 1e-6, 1.0, std), threshold=1.5).astype(np.float32)

    dims = {m: 768 for m in args.modalities}
    preds = {"val": [], "test": []}
    for seed in seeds:
        print(f"[INFO] === seed={seed} ({'+'.join(args.modalities)}, {args.label_mode}) ===")
        model, best = train_seed(seed, hp, dims, args.targets, data["train"]["x"], train_target,
                                 data["val"]["x"], data["val"][y_key], device)
        torch.save(model.state_dict(), os.path.join(out_dir, f"mlp_seed{seed}.pt"))
        for s in ("val", "test"):  # test는 학습/선택에 쓰지 않고 여기서 한 번만 예측
            preds[s].append(predict(model, data[s]["x"], device))
        del model

    report = {"exp_name": name, "modalities": args.modalities, "targets": args.targets, "label_mode": args.label_mode,
              "seeds": seeds, "length_coefs": dict(zip(args.targets, coefs)), "results": {}}
    target_idx = [INTERVIEW_LABEL_NAMES.index(t) for t in args.targets]
    runs = {name: {s: np.mean(preds[s], axis=0) for s in ("val", "test")}}
    if args.final_model_dir and os.path.exists(os.path.join(args.final_model_dir, "ensemble.json")):
        runs["final_model(text+audio+video)"] = {s: final_model_predict(args.final_model_dir, data[s]["raw_feats"],
                                                                         target_idx, device) for s in ("val", "test")}
    for run, p in runs.items():
        report["results"][run] = {}
        for s in ("val", "test"):
            d = data[s]
            per, macro = evaluate(p[s], d["y"], d["y_resid"], d["len"], d["users"])
            report["results"][run][s] = {"macro": macro, "per_target": {args.targets[k]: v for k, v in per.items()}}

    json.dump(report, open(os.path.join(out_dir, "metrics_report.json"), "w"), indent=2)
    json.dump({"modalities": args.modalities, "modality_dims": dims, "targets": args.targets, "label_mode": args.label_mode,
               "audio_layer": AUDIO_LAYER, "fusion": "prefusion", "d_model": hp.d_model, "dropout": hp.dropout,
               "transformer_layers": hp.transformer_layers, "n_heads": hp.n_heads, "seeds": seeds,
               "label_mean": mean.tolist(), "label_std": std.tolist(), "length_coefs": coefs},
              open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    pickle.dump(scalers, open(os.path.join(out_dir, "scaler.pkl"), "wb"))

    cols = ["spearman_raw", "spearman_resid", "corr_length", "within_person", "person_level"]
    print("=" * 110)
    print(f"[RESULT] {name}  (타깃 {len(args.targets)}개 macro, 앙상블 {len(seeds)}개)")
    print(f"  {'모델':34s} {'split':5s} " + " ".join(f"{c:>15s}" for c in cols))
    for run in runs:
        for s in ("val", "test"):
            m = report["results"][run][s]["macro"]
            print(f"  {run:34s} {s:5s} " + " ".join(f"{m[c]:15.3f}" for c in cols))
    print("  corr_length는 0에 가까울수록 길이에 덜 기댐. 모드 비교와 선택은 val로 하고, test는 참고용으로 한 번만 본다.")
    print(f"[INFO] 저장: {out_dir}")
    print("=" * 110)


if __name__ == "__main__":
    main()
