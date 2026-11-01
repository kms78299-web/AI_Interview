"""
Colab(또는 다른 환경)에서 run_inference가 학습 파이프라인과 같은 결과를 내는지 검증하기 위한 기준값 생성.

test split에서 대본이 있는 샘플 N개를 골라서:
  1) 원본 영상을 model_parity/videos/ 로 복사 (영상은 계산 노드 로컬 디스크에만 있어서 그 노드에서 실행해야 함)
  2) 저장된 캐시 특징(학습 때 쓴 것)으로 앙상블의 "기대 예측값"(raw, 백분위)을 계산 -> parity_reference.json
기대값은 inference.py를 거치지 않고 따로 구현한 경로(캐시 특징 -> scaler -> 모델 5개 평균)라서,
추론 코드와 독립적인 비교 기준이 된다.

사용 (영상 복사 때문에 ariel-v7에서): sbatch run_parity_reference.sh
"""

import argparse
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch

from config import DEFAULT_FEATURES_DIR, DEFAULT_META_PATH, DEFAULT_METADATA_PATH, DEFAULT_VIDEO_ROOT, DEFAULT_SPLITS_DIR, INTERVIEW_LABEL_NAMES, SPLIT_SEED
from data_utils import clean_transcript, load_metadata
from label_utils import percentile_to_band, zscore_to_percentile
from train_mini_crmf import MiniCRMF


def parse_args():
    p = argparse.ArgumentParser(description="추론 검증용 기준값 생성")
    p.add_argument("--model_dir", default="/ceph_data/kms78299/model/checkpoints/final_audio6_R1_ens5")
    p.add_argument("--out_dir", default="/ceph_data/kms78299/model_parity")
    p.add_argument("--n", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out_dir, "videos"), exist_ok=True)

    split = json.load(open(os.path.join(DEFAULT_SPLITS_DIR, f"split_seed{SPLIT_SEED}.json")))
    meta = pd.read_csv(DEFAULT_META_PATH, dtype={"id": str, "user_no": str}).set_index("id")
    items = {it["id"]: it for it in load_metadata(DEFAULT_METADATA_PATH)}
    ids = [i for i in split["test"] if bool(meta.loc[i, "has_transcript"])][: args.n]
    print(f"[INFO] 검증 샘플(test split, 대본 있음): {ids}")

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    members = [m["file"] for m in json.load(open(os.path.join(args.model_dir, "ensemble.json")))["members"]]
    scalers = pickle.load(open(os.path.join(args.model_dir, "scaler.pkl"), "rb"))
    ref_sorted = json.load(open(os.path.join(args.model_dir, "percentile_reference.json")))["predictions_sorted"]
    layer = cfg["audio_layer"]

    models = []
    for f in members:
        m = MiniCRMF(cfg["modality_dims"], cfg["d_model"], cfg["fusion"], INTERVIEW_LABEL_NAMES, dropout=cfg["dropout"],
                     n_transformer_layers=cfg["transformer_layers"], n_heads=cfg["n_heads"])
        m.load_state_dict(torch.load(os.path.join(args.model_dir, f), map_location="cpu"))
        models.append(m.eval())

    feats = os.path.join(DEFAULT_FEATURES_DIR)
    entries = []
    for sid in ids:
        text = np.load(os.path.join(feats, "embed", "bert-base-uncased", f"{sid}.npy"))
        video = np.load(os.path.join(feats, "embed", "videomae", f"{sid}.npy"))
        audio = np.load(os.path.join(feats, "embed_layers", "audio", "wav2vec2", f"{sid}.npy"))[0, layer]
        batch = {m: torch.from_numpy(scalers[m].transform(v.reshape(1, -1)).astype(np.float32))
                 for m, v in (("text", text), ("audio", audio), ("video", video))}
        with torch.no_grad():
            raw = np.mean([m(batch).numpy()[0] for m in models], axis=0)
        file_name = items[sid]["file_name"]
        dst = os.path.join(args.out_dir, "videos", os.path.basename(file_name))
        shutil.copy(os.path.join(DEFAULT_VIDEO_ROOT, file_name), dst)
        pct = {n: float(zscore_to_percentile(float(raw[i]), np.array(ref_sorted[n]))) for i, n in enumerate(INTERVIEW_LABEL_NAMES)}
        entries.append({
            "id": sid, "video": os.path.join("videos", os.path.basename(file_name)),
            "transcript": clean_transcript(items[sid].get("transcript")),
            "expected_raw": {n: float(raw[i]) for i, n in enumerate(INTERVIEW_LABEL_NAMES)},
            "expected_percentile": {n: int(round(p)) for n, p in pct.items()},
            "expected_band": {n: percentile_to_band(p) for n, p in pct.items()},
        })
        print(f"[INFO] id={sid} 기대값 계산 완료, 영상 복사: {dst}")

    with open(os.path.join(args.out_dir, "parity_reference.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 저장 완료: {args.out_dir}/parity_reference.json (영상 {len(entries)}개 포함)")


if __name__ == "__main__":
    main()
