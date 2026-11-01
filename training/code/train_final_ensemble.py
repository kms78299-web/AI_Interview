"""
현재 최선 구성을 학습하고 서비스용으로 저장한다.

구성: audio = wav2vec2-base-960h의 6층 평균(중간 층) + text(BERT 마지막 층) + video(VideoMAE 기존 임베딩),
      MiniCRMF prefusion, 규제 R1(dropout 0.3, weight decay 0.05), seed 5개 앙상블(예측 평균).
선택 근거(run_combo.py 결과): audio 층 sweep에서 5~7층이 완만한 봉우리, R1이 val 최고, 앙상블이 추가로 상승.
이 실험의 val 기준 선택이라 val 수치는 낙관적일 수 있고, 보고할 성능은 test(모델 선택 후 1회)다.

저장 (checkpoints/<exp_name>/):
  mlp_seed{k}.pt          seed별 best-val epoch 모델 가중치 (5개)
  ensemble.json           멤버 목록 (inference.py가 읽음)
  config.json             구조/규제 설정 + audio_layer=6
  scaler.pkl              {text, audio, video} StandardScaler (audio는 6층 특징 기준으로 새로 fit)
  label_stats.json
  percentile_reference.json  앙상블 예측의 val 분포 (0~100 변환 기준)
  metrics_report.json     seed별 / 단일 모델 평균±표준편차 / 앙상블 val·test 지표

사용:
    python train_final_ensemble.py --smoke   # 동작 확인(seed 1개, 2 epoch, 결과 해석 금지)
    python train_final_ensemble.py
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

from config import (DEFAULT_CHECKPOINTS_DIR, DEFAULT_FEATURES_DIR, DEFAULT_META_PATH, DEFAULT_SPLITS_DIR,
                    INTERVIEW_LABEL_NAMES, SPLIT_SEED)
from train_mini_crmf import (EmbedDataset, build_or_load_split, check_no_user_overlap, evaluate, find_available_ids,
                             load_embeddings, load_labels, resolve_modality_dirs, soft_winsorize, train_one_seed)

MODS = ["text", "audio", "video"]
SPLITS = ("train", "val", "test")
ENCODERS = ["wav2vec2-base", "bert-base-uncased", "videomae"]
AUDIO_LAYER = 6  # wav2vec2-base-960h의 6층(중간 층) 평균: 마지막 층보다 audio 단독 val +0.15


class HP:
    """train_one_seed가 읽는 하이퍼파라미터 (규제 R1: dropout 0.3, weight decay 0.05)."""
    d_model = 256
    dropout = 0.3
    weight_decay = 5e-2
    lr = 1e-3
    transformer_layers = 1
    n_heads = 4
    batch_size = 32
    max_epochs = 30
    patience = 5
    corr_loss = False
    corr_lambda = 0.1
    fusion = "prefusion"


def prepare_data():
    """split(user_no 기준, 겹침 assert) / text,video 기존 임베딩 표준화(train으로 fit) / 라벨 winsorization(train 타깃만)."""
    meta_df = pd.read_csv(DEFAULT_META_PATH, dtype={"id": str, "user_no": str})
    dirs = resolve_modality_dirs(ENCODERS, DEFAULT_FEATURES_DIR)
    available = set(find_available_ids(meta_df, dirs))
    split = build_or_load_split(meta_df, DEFAULT_SPLITS_DIR, seed=SPLIT_SEED)
    check_no_user_overlap(meta_df, split)
    ids = {s: [i for i in split[s] if i in available] for s in SPLITS}
    print(f"[INFO] 샘플 수: train={len(ids['train'])}, val={len(ids['val'])}, test={len(ids['test'])}")

    tv_dirs = {m: dirs[m] for m in ("text", "video")}
    raw = {s: load_embeddings(ids[s], tv_dirs) for s in SPLITS}
    labels = {s: load_labels(meta_df, ids[s]) for s in SPLITS}
    scalers = {m: StandardScaler().fit(raw["train"][m]) for m in tv_dirs}
    scaled = {s: {m: scalers[m].transform(raw[s][m]).astype(np.float32) for m in tv_dirs} for s in SPLITS}

    y = labels["train"]
    mean, std = y.mean(axis=0), y.std(axis=0)
    target = soft_winsorize((y - mean) / np.where(std < 1e-6, 1.0, std), threshold=1.5).astype(np.float32)
    return {"ids_by_split": ids, "raw_labels": labels, "scaled": scaled, "scalers": scalers,
            "train_y_target": target, "label_mean": mean, "label_std": std}


def load_audio_layer(ids_by_split):
    """features/embed_layers/audio/wav2vec2/{id}.npy (2,13,768)에서 [평균, AUDIO_LAYER층] -> (N,768)."""
    out = {}
    for s, id_list in ids_by_split.items():
        paths = [os.path.join(DEFAULT_FEATURES_DIR, "embed_layers", "audio", "wav2vec2", f"{i}.npy") for i in id_list]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"audio 층별 특징이 없는 샘플 {len(missing)}개 — extract_layers.py --what audio 먼저")
        out[s] = np.stack([np.load(p)[0, AUDIO_LAYER] for p in paths])
    return out


def compute_metrics(preds, labels, target_names):
    """타깃별 + macro Spearman/Pearson (원본 라벨 기준, train_mini_crmf.evaluate와 같은 계산)."""
    out, sps, prs = {}, [], []
    for i, name in enumerate(target_names):
        sp = float(spearmanr(preds[:, i], labels[:, i]).statistic)
        pr = float(pearsonr(preds[:, i], labels[:, i]).statistic)
        out[f"spearman_{name}"], out[f"pearson_{name}"] = sp, pr
        sps.append(sp)
        prs.append(pr)
    out["spearman_macro"], out["pearson_macro"] = float(np.mean(sps)), float(np.mean(prs))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="최선 구성(audio 6층 + R1 + 5-seed 앙상블) 학습/저장")
    p.add_argument("--exp_name", default="final_audio6_R1_ens5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--checkpoints_dir", default=DEFAULT_CHECKPOINTS_DIR)
    p.add_argument("--smoke", action="store_true", help="seed 1개, 2 epoch (동작 확인용, 결과 해석 금지)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    seeds = args.seeds[:1] if args.smoke else args.seeds
    epochs = 2 if args.smoke else 30
    exp_name = args.exp_name + ("_smoke" if args.smoke else "")
    out_dir = os.path.join(args.checkpoints_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    data = prepare_data()
    ids = data["ids_by_split"]
    audio = load_audio_layer(ids)  # (N,768)
    audio_scaler = StandardScaler().fit(audio["train"])
    scalers = {"text": data["scalers"]["text"], "audio": audio_scaler, "video": data["scalers"]["video"]}
    emb = {s: {"text": data["scaled"][s]["text"], "video": data["scaled"][s]["video"],
               "audio": audio_scaler.transform(audio[s]).astype(np.float32)} for s in SPLITS}

    val_y, test_y = data["raw_labels"]["val"], data["raw_labels"]["test"]
    loader = lambda split, y: torch.utils.data.DataLoader(  # noqa: E731
        EmbedDataset({m: emb[split][m] for m in MODS}, np.zeros_like(y, dtype=np.float32)), batch_size=32, shuffle=False)
    val_loader, test_loader = loader("val", val_y), loader("test", test_y)
    train_ds = EmbedDataset({m: emb["train"][m] for m in MODS}, data["train_y_target"])

    hp = HP()
    hp.max_epochs = epochs
    cfg = {"dropout": hp.dropout, "weight_decay": hp.weight_decay, "d_model": hp.d_model, "lr": hp.lr}
    dims = {m: 768 for m in MODS}

    members, per_seed, val_preds, test_preds = [], [], [], []
    for seed in seeds:
        print(f"[INFO] === seed={seed} 학습 ({cfg}) ===")
        model, best_val = train_one_seed(seed, hp, dims, train_ds, val_loader, val_y, device)
        fname = f"mlp_seed{seed}.pt"
        torch.save(model.state_dict(), os.path.join(out_dir, fname))
        val_m, val_p = evaluate(model, val_loader, device, INTERVIEW_LABEL_NAMES, val_y)
        test_m, test_p = evaluate(model, test_loader, device, INTERVIEW_LABEL_NAMES, test_y)  # test는 선택 후 1회
        print(f"[INFO] seed={seed} val={val_m['spearman_macro']:.4f} test={test_m['spearman_macro']:.4f}")
        members.append({"seed": seed, "file": fname, "best_val_spearman_macro": best_val})
        per_seed.append({"seed": seed, "val": val_m, "test": test_m})
        val_preds.append(val_p)
        test_preds.append(test_p)
        del model
        torch.cuda.empty_cache()

    ens_val, ens_test = np.mean(val_preds, axis=0), np.mean(test_preds, axis=0)
    ens_val_m = compute_metrics(ens_val, val_y, INTERVIEW_LABEL_NAMES)
    ens_test_m = compute_metrics(ens_test, test_y, INTERVIEW_LABEL_NAMES)

    def agg(split):
        return {k: {"mean": float(np.mean([p[split][k] for p in per_seed])),
                    "std": float(np.std([p[split][k] for p in per_seed]))} for k in per_seed[0][split]}
    single_val, single_test = agg("val"), agg("test")

    with open(os.path.join(out_dir, "ensemble.json"), "w") as f:
        json.dump({"combine": "mean of raw predictions", "members": members}, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"encoders": ["wav2vec2-base", "bert-base-uncased", "videomae"], "modality_dims": dims,
                   "fusion": "prefusion", "d_model": cfg["d_model"], "transformer_layers": hp.transformer_layers,
                   "n_heads": hp.n_heads, "dropout": cfg["dropout"], "weight_decay": cfg["weight_decay"],
                   "lr": cfg["lr"], "max_epochs": epochs, "patience": hp.patience, "seeds": seeds,
                   "audio_layer": 6, "audio_pool": "mean", "regularization": "R1"}, f, indent=2)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    with open(os.path.join(out_dir, "label_stats.json"), "w") as f:
        json.dump({"mean": dict(zip(INTERVIEW_LABEL_NAMES, data["label_mean"].tolist())),
                   "std": dict(zip(INTERVIEW_LABEL_NAMES, data["label_std"].tolist())), "winsor_threshold": 1.5}, f, indent=2)
    with open(os.path.join(out_dir, "percentile_reference.json"), "w") as f:
        json.dump({"source": "validation predictions of the seed ensemble (mean of members)",
                   "n_samples": len(ids["val"]), "exp_name": exp_name,
                   "predictions_sorted": {n: sorted(float(v) for v in ens_val[:, i])
                                          for i, n in enumerate(INTERVIEW_LABEL_NAMES)}}, f)
    report = {
        "exp_name": exp_name, "seeds": seeds, "regularization": cfg,
        "n_train": len(ids["train"]), "n_val": len(ids["val"]), "n_test": len(ids["test"]),
        "per_seed": per_seed,
        "single_model_val": single_val, "single_model_test": single_test,
        "ensemble_val": ens_val_m, "ensemble_test": ens_test_m,
        # inference.py의 설명문이 읽는 키: 서비스가 쓰는 앙상블의 test macro Spearman
        "test_spearman_agg": {"spearman_macro": {"mean": ens_test_m["spearman_macro"], "std": 0.0}},
        "note": "구성은 run_combo.py에서 val 기준으로 골랐다(audio 6층, 규제 R1). test는 모델 선택 후 1회 평가. 앙상블 수치는 단일 실현값이라 표준편차 없음.",
    }
    with open(os.path.join(out_dir, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 90)
    print(f"[RESULT] {exp_name}")
    print(f"  단일 모델({len(seeds)} seed): val {single_val['spearman_macro']['mean']:.4f} ± {single_val['spearman_macro']['std']:.4f}, "
          f"test {single_test['spearman_macro']['mean']:.4f} ± {single_test['spearman_macro']['std']:.4f}")
    print(f"  앙상블: val {ens_val_m['spearman_macro']:.4f}, test {ens_test_m['spearman_macro']:.4f} "
          f"(pearson test {ens_test_m['pearson_macro']:.4f})")
    print("  이전 기준(참고): 단일 val 0.384 / test 0.486, 3-seed 앙상블 val 0.408 / test 0.520")
    print(f"[INFO] 저장 완료: {out_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
