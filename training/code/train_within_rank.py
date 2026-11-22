"""
같은 사람 답변 순위 학습 실험: 회귀 손실 + lam * 같은 사람 답변 쌍의 순위 손실(-log sigmoid(s_i - s_j)).

기존 모델은 사람끼리는 잘 맞지만(사람 단위 Spearman 약 0.73) 같은 사람의 답변끼리는 거의 못 가린다(0.15~0.18).
같은 사람의 답변 쌍으로 "어느 답변이 더 높게 평가됐나"를 직접 학습시키면 이 비교가 나아지는지 확인한다.
lam=0은 회귀만 쓰는 기준선이다. 쌍은 같은 사람, 타깃별 라벨 차이가 --min_diff보다 큰 경우만 쓴다.
구조와 규제는 최종 모델과 같고 seed 5개 앙상블이다. lam은 val로 고르고, test는 마지막에 한 번 본다.

비교 대상: 무작위(50%), "더 짧은 답변이 낫다" 규칙, 최종 모델.
주요 지표 (같은 사람 답변 쌍에서 더 높게 평가된 쪽을 맞힌 비율):
  pair_acc_all / pair_acc_d05 / pair_acc_d10   라벨 차이 0 / 0.5 / 1.0 초과 쌍
  pair_acc_len_matched                         라벨 차이 0.5 초과이면서 길이가 비슷한 쌍(log 단어 수 차이 < 0.15)
  person_level                                 사람 평균끼리 Spearman (떨어지지 않는지 확인)
  ci95                                         사람 단위 bootstrap 95% 구간 (pair_acc_d05, 그리고 '짧은 쪽' 규칙과의 차이)

    python train_within_rank.py --smoke
    python train_within_rank.py                              # text+audio+video, lam 0 / 0.5 / 1 / 2
    python train_within_rank.py --modalities audio video
"""

import argparse
import copy
import itertools
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from config import (DEFAULT_CHECKPOINTS_DIR, DEFAULT_META_PATH, DEFAULT_METADATA_PATH, DEFAULT_SPLITS_DIR,
                    INTERVIEW_LABEL_NAMES, SPLIT_SEED)
from data_utils import clean_transcript, load_metadata
from train_delivery import FINAL_MODEL_DIR, HP, WORD_RE, final_model_predict, load_features, predict
from train_mini_crmf import (EmbedDataset, MiniCRMF, build_or_load_split, check_no_user_overlap, set_seed,
                             soft_winsorize)

SPLITS = ("train", "val", "test")
LEN_MATCH = 0.15  # log1p(단어 수) 차이가 이보다 작으면 길이가 비슷한 쌍


def parse_args():
    p = argparse.ArgumentParser(description="같은 사람 답변 순위 학습 실험")
    p.add_argument("--modalities", nargs="+", default=["text", "audio", "video"], choices=["text", "audio", "video"])
    p.add_argument("--targets", nargs="+", default=INTERVIEW_LABEL_NAMES, choices=INTERVIEW_LABEL_NAMES)
    p.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--min_diff", type=float, default=0.5, help="학습 쌍의 최소 라벨 차이")
    p.add_argument("--pairs_per_step", type=int, default=128, help="스텝마다 타깃별로 뽑는 쌍 수")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--exp_name", default=None, help="기본값: within_rank_{modalities}")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def build_pairs(users, y, min_diff):
    """타깃별 같은 사람 답변 쌍 배열 (N_k, 2): 첫 열이 라벨이 더 높은 답변."""
    groups = defaultdict(list)
    for idx, u in enumerate(users):
        groups[u].append(idx)
    pairs = []
    for k in range(y.shape[1]):
        pk = []
        for idxs in groups.values():
            for i, j in itertools.combinations(idxs, 2):
                d = y[i, k] - y[j, k]
                if abs(d) > min_diff:
                    pk.append((i, j) if d > 0 else (j, i))
        pairs.append(np.array(pk, dtype=np.int64).reshape(-1, 2))
    return pairs


def pair_table(users, y, log_len):
    """평가용: 같은 사람의 모든 답변 쌍과 라벨 차이, 길이 차이, 사람."""
    rows = []
    groups = defaultdict(list)
    for idx, u in enumerate(users):
        groups[u].append(idx)
    for u, idxs in groups.items():
        for i, j in itertools.combinations(idxs, 2):
            rows.append((u, i, j))
    u, i, j = (np.array(c) for c in zip(*rows))
    return {"user": u, "i": i, "j": j, "dy": y[i] - y[j], "dlen": log_len[i] - log_len[j]}


def _correct(score_diff, label_diff):
    """쌍마다 맞으면 1, 틀리면 0, 예측이 같으면 0.5."""
    return np.where(score_diff == 0, 0.5, (np.sign(score_diff) == np.sign(label_diff)).astype(float))


def pair_metrics(pred, pt, log_len, n_boot, rng):
    """타깃별 쌍 정확도와 길이 규칙 기준선, 사람 단위 bootstrap 구간 (타깃 macro)."""
    T = pred.shape[1]
    out = {"model": defaultdict(list), "length_rule": defaultdict(list)}
    boot_inputs = []
    for k in range(T):
        dy = pt["dy"][:, k]
        dp = pred[pt["i"], k] - pred[pt["j"], k]
        dl = -(log_len[pt["i"]] - log_len[pt["j"]])  # 짧은 쪽을 높게 보는 규칙
        masks = {"pair_acc_all": dy != 0, "pair_acc_d05": np.abs(dy) > 0.5, "pair_acc_d10": np.abs(dy) > 1.0,
                 "pair_acc_len_matched": (np.abs(dy) > 0.5) & (np.abs(pt["dlen"]) < LEN_MATCH)}
        for name, m in masks.items():
            out["model"][name].append(_correct(dp[m], dy[m]).mean() if m.any() else np.nan)
            out["length_rule"][name].append(_correct(dl[m], dy[m]).mean() if m.any() else np.nan)
        m = masks["pair_acc_d05"]
        boot_inputs.append((pt["user"][m], _correct(dp[m], dy[m]), _correct(dl[m], dy[m])))
    res = {who: {name: float(np.nanmean(v)) for name, v in d.items()} for who, d in out.items()}
    res["n_pairs_d05"] = int(sum(len(b[0]) for b in boot_inputs))

    users = np.unique(pt["user"])
    per_user = []  # 타깃별 (사람별 맞은 수, 쌍 수) — bootstrap을 빠르게 하려고 미리 집계
    for u_arr, c_model, c_rule in boot_inputs:
        df = pd.DataFrame({"u": u_arr, "m": c_model, "r": c_rule}).groupby("u").agg(m=("m", "sum"), r=("r", "sum"), n=("m", "size"))
        df = df.reindex(users, fill_value=0)
        per_user.append(df.to_numpy())
    per_user = np.stack(per_user)  # (T, U, 3)
    accs, diffs = [], []
    for _ in range(n_boot):
        s = per_user[:, rng.integers(0, len(users), len(users))].sum(axis=1)  # (T, 3)
        acc_m, acc_r = s[:, 0] / np.maximum(s[:, 2], 1), s[:, 1] / np.maximum(s[:, 2], 1)
        accs.append(acc_m.mean())
        diffs.append((acc_m - acc_r).mean())
    res["ci95"] = {"pair_acc_d05": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
                   "model_minus_length_rule": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    return res


def spearman_metrics(pred, y, users):
    overall, person = [], []
    for k in range(pred.shape[1]):
        overall.append(spearmanr(pred[:, k], y[:, k]).statistic)
        df = pd.DataFrame({"p": pred[:, k], "y": y[:, k], "u": users}).groupby("u").mean()
        person.append(spearmanr(df["p"], df["y"]).statistic)
    return {"spearman": float(np.mean(overall)), "person_level": float(np.mean(person))}


def val_pair_acc(model, x, pt, device):
    pred = predict(model, x, device)
    accs = []
    for k in range(pred.shape[1]):
        m = np.abs(pt["dy"][:, k]) > 0.5
        accs.append(_correct(pred[pt["i"][m], k] - pred[pt["j"][m], k], pt["dy"][m, k]).mean())
    return float(np.mean(accs))


def train_seed(seed, hp, dims, targets, x_tr, y_target, pairs, lam, pairs_per_step, val_x, val_pt, device):
    """회귀(Huber) + lam * 순위 손실. val 같은 사람 쌍 정확도(라벨 차이 0.5 초과)로 early stopping."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = MiniCRMF(dims, hp.d_model, "prefusion", targets, dropout=hp.dropout,
                     n_transformer_layers=hp.transformer_layers, n_heads=hp.n_heads).to(device)
    ds = EmbedDataset(x_tr, y_target)
    loader = torch.utils.data.DataLoader(ds, batch_size=hp.batch_size, shuffle=True, drop_last=len(ds) > hp.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp.lr, total_steps=max(1, len(loader)) * hp.max_epochs,
                                                pct_start=0.15)
    huber = nn.HuberLoss(delta=1.0)
    x_all = {m: torch.from_numpy(v).to(device) for m, v in x_tr.items()}
    best, best_state, wait = -np.inf, None, 0
    for epoch in range(hp.max_epochs):
        model.train()
        for xb, yb in loader:
            loss = huber(model({k: v.to(device) for k, v in xb.items()}), yb.to(device))
            if lam > 0:
                picked = [p[rng.integers(0, len(p), min(pairs_per_step, len(p)))] for p in pairs]
                uniq, inv = np.unique(np.concatenate([q.ravel() for q in picked]), return_inverse=True)
                out = model({m: v[torch.from_numpy(uniq).to(device)] for m, v in x_all.items()})
                rank_losses, offset = [], 0
                for k, q in enumerate(picked):
                    pos = torch.from_numpy(inv[offset:offset + q.size].reshape(-1, 2)).to(device)
                    offset += q.size
                    rank_losses.append(F.softplus(-(out[pos[:, 0], k] - out[pos[:, 1], k])).mean())
                loss = loss + lam * torch.stack(rank_losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        score = val_pair_acc(model, val_x, val_pt, device)
        print(f"  [lam={lam} seed={seed}] epoch={epoch:02d} val_pair_acc_d05={score:.4f}")
        if score > best:
            best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= hp.patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    hp = HP()
    seeds, lams, n_boot = (args.seeds[:1], args.lams[:2], 50) if args.smoke else (args.seeds, args.lams, args.n_boot)
    if args.smoke:
        hp.max_epochs = 2
    name = args.exp_name or f"within_rank_{'-'.join(args.modalities)}"
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
        d = {"raw_feats": load_features(ids), "users": meta.loc[ids, "user_no"].to_numpy(),
             "y": meta.loc[ids, args.targets].to_numpy(dtype=np.float64),
             "len": np.array([np.log1p(len(WORD_RE.findall(transcripts.get(i, "")))) for i in ids])}
        d["pairs_eval"] = pair_table(d["users"], d["y"], d["len"])
        data[s] = d
        print(f"[INFO] {s}: 답변 {len(ids)}개, 같은 사람 쌍 {len(d['pairs_eval']['i'])}개")

    scalers = {m: StandardScaler().fit(data["train"]["raw_feats"][m]) for m in args.modalities}
    for s in SPLITS:
        data[s]["x"] = {m: scalers[m].transform(data[s]["raw_feats"][m]).astype(np.float32) for m in args.modalities}
    y_tr = data["train"]["y"]
    mean, std = y_tr.mean(axis=0), y_tr.std(axis=0)
    y_target = soft_winsorize((y_tr - mean) / np.where(std < 1e-6, 1.0, std), threshold=1.5).astype(np.float32)
    train_pairs = build_pairs(data["train"]["users"], y_tr, args.min_diff)
    print(f"[INFO] 학습 쌍 (타깃별, 라벨 차이 > {args.min_diff}): {[len(p) for p in train_pairs]}")

    dims = {m: 768 for m in args.modalities}
    runs = {}
    for lam in lams:
        preds = {"val": [], "test": []}
        lam_dir = os.path.join(out_dir, f"lam{lam:g}")
        os.makedirs(lam_dir, exist_ok=True)
        for seed in seeds:
            print(f"[INFO] === lam={lam} seed={seed} ===")
            model = train_seed(seed, hp, dims, args.targets, data["train"]["x"], y_target, train_pairs, lam,
                               args.pairs_per_step, data["val"]["x"], data["val"]["pairs_eval"], device)
            torch.save(model.state_dict(), os.path.join(lam_dir, f"mlp_seed{seed}.pt"))
            for s in ("val", "test"):  # test는 학습/선택에 쓰지 않고 여기서 한 번만 예측
                preds[s].append(predict(model, data[s]["x"], device))
            del model
        runs[f"lam={lam:g}"] = {s: np.mean(preds[s], axis=0) for s in ("val", "test")}

    target_idx = [INTERVIEW_LABEL_NAMES.index(t) for t in args.targets]
    if os.path.exists(os.path.join(FINAL_MODEL_DIR, "ensemble.json")):
        runs["final_model"] = {s: final_model_predict(FINAL_MODEL_DIR, data[s]["raw_feats"], target_idx, device)
                               for s in ("val", "test")}

    rng = np.random.default_rng(0)
    report = {"exp_name": name, "modalities": args.modalities, "targets": args.targets, "lams": lams,
              "min_diff": args.min_diff, "seeds": seeds, "results": {}}
    for run, p in runs.items():
        report["results"][run] = {}
        for s in ("val", "test"):
            d = data[s]
            report["results"][run][s] = {**spearman_metrics(p[s], d["y"], d["users"]),
                                         **pair_metrics(p[s], d["pairs_eval"], d["len"], n_boot, rng)}
    lam_runs = [r for r in runs if r.startswith("lam=")]
    best = max(lam_runs, key=lambda r: report["results"][r]["val"]["model"]["pair_acc_d05"])
    report["selected_by_val"] = best

    json.dump(report, open(os.path.join(out_dir, "metrics_report.json"), "w"), indent=2)
    json.dump({"modalities": args.modalities, "modality_dims": dims, "targets": args.targets, "selected": best,
               "fusion": "prefusion", "d_model": hp.d_model, "dropout": hp.dropout,
               "transformer_layers": hp.transformer_layers, "n_heads": hp.n_heads, "seeds": seeds,
               "label_mean": mean.tolist(), "label_std": std.tolist()},
              open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    pickle.dump(scalers, open(os.path.join(out_dir, "scaler.pkl"), "wb"))

    cols = ["pair_acc_all", "pair_acc_d05", "pair_acc_d10", "pair_acc_len_matched"]
    print("=" * 120)
    print(f"[RESULT] {name}  (타깃 {len(args.targets)}개 macro, 앙상블 {len(seeds)}개, 쌍 정확도는 같은 사람 답변 쌍 기준)")
    print(f"  {'모델':16s} {'split':5s} " + " ".join(f"{c:>21s}" for c in cols) + f" {'person_level':>13s} {'spearman':>9s}")
    for s in ("val", "test"):
        rule = report["results"][lam_runs[0]][s]["length_rule"]
        print(f"  {'짧은쪽규칙':16s} {s:5s} " + " ".join(f"{rule[c]:21.3f}" for c in cols))
        for run in runs:
            r = report["results"][run][s]
            print(f"  {run:16s} {s:5s} " + " ".join(f"{r['model'][c]:21.3f}" for c in cols)
                  + f" {r['person_level']:13.3f} {r['spearman']:9.3f}")
    for s in ("val", "test"):
        ci = report["results"][best][s]["ci95"]
        print(f"  [{best}, {s}] pair_acc_d05 95% 구간 {ci['pair_acc_d05'][0]:.3f}~{ci['pair_acc_d05'][1]:.3f}, "
              f"짧은쪽 규칙 대비 차이 {ci['model_minus_length_rule'][0]:+.3f}~{ci['model_minus_length_rule'][1]:+.3f}")
    print(f"  val로 고른 lam: {best}. 무작위 기준은 0.5. 짧은쪽 규칙 대비 차이 구간이 0을 넘어야 길이 외 신호가 있다고 본다.")
    print(f"[INFO] 저장: {out_dir}")
    print("=" * 120)


if __name__ == "__main__":
    main()
