"""
audio 층별 임베딩 추출 (최종 모델의 audio 입력 = wav2vec2 6층 평균).

기존 임베딩(features/embed/wav2vec2-base)은 wav2vec2-base-960h의 마지막 층만 평균낸 768차원이었다. 이 모델은
음성 인식으로 fine-tuning돼서 마지막 층은 말 내용 쪽으로 치우치므로, 모든 층(0~12)을 시간축으로 평균/표준편차 pooling해서
저장하고 중간 층(6층)을 쓴다 (실험에서 5~7층이 완만한 봉우리였고 마지막 층이 최악).

입력: features/raw/audio_wave/{id}.npy (16kHz 모노 파형, cache_raw_inputs.py --modality audio로 만든 것)
저장: features/embed_layers/audio/wav2vec2/{id}.npy  float32 (2, 13, 768)  [0]=시간 평균, [1]=시간 표준편차 (층별)
검증: 층 12의 평균은 기존 임베딩과 같아야 하므로 처음 3개 샘플은 기존 파일과 비교해서 [VERIFY]로 출력한다.

사용:
    python extract_layers.py --what audio --limit 5   # 소량 확인 (이어서 전체 돌리면 이미 된 건 스킵)
    python extract_layers.py --what audio
"""

import argparse
import logging
import os
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

from config import AUDIO_EMBED_ENCODERS, DEFAULT_FEATURES_DIR, DEFAULT_LOG_DIR, DEFAULT_METADATA_PATH
from data_utils import load_metadata


def parse_args():
    p = argparse.ArgumentParser(description="audio 층별 임베딩 추출")
    p.add_argument("--what", nargs="+", default=["audio"], choices=["audio"])
    p.add_argument("--metadata_path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR)
    p.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    # run_extract_layers.sh가 예전 인자도 같이 넘기므로 무시하고 받기만 함
    p.add_argument("--video_root", default=None)
    return p.parse_args()


def verify_against(new_vec, ref_path, sample_id, state):
    """층 12 평균이 기존 임베딩 파일과 같은지 처음 3개 샘플만 확인."""
    if state["n"] >= 3 or not os.path.exists(ref_path):
        return
    ref = np.load(ref_path)
    diff, scale = float(np.abs(new_vec - ref).max()), float(np.abs(ref).mean())
    tag = "OK" if diff <= 0.05 * max(scale, 1e-8) else "WARN"
    print(f"[VERIFY] {tag} audio id={sample_id}: 층12 평균 vs 기존 임베딩 max|diff|={diff:.2e} (평균 절댓값 {scale:.3f})")
    state["n"] += 1


def run_audio(items, args, logger):
    from audio_features import load_wav2vec2

    feature_extractor, model = load_wav2vec2(AUDIO_EMBED_ENCODERS["wav2vec2-base"], args.device)
    out_dir = os.path.join(args.features_dir, "embed_layers", "audio", "wav2vec2")
    os.makedirs(out_dir, exist_ok=True)
    state = {"n": 0}
    n_ok = n_skip = n_err = 0
    for item in tqdm(items, desc="layers/audio"):
        sid = item["id"]
        out_path = os.path.join(out_dir, f"{sid}.npy")
        if not args.overwrite and os.path.exists(out_path):
            n_skip += 1
            continue
        try:
            wave_path = os.path.join(args.features_dir, "raw", "audio_wave", f"{sid}.npy")
            if not os.path.exists(wave_path):
                raise FileNotFoundError(f"{wave_path} 없음 — cache_raw_inputs.py --modality audio 먼저")
            inputs = feature_extractor(np.load(wave_path), sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = model(**inputs, output_hidden_states=True).hidden_states
            hs = torch.stack([h[0] for h in hidden])  # (13, T, 768)
            arr = torch.stack([hs.mean(dim=1), hs.std(dim=1, unbiased=False)]).cpu().numpy().astype(np.float32)
            if not np.all(np.isfinite(arr)):
                raise ValueError("NaN/inf")
            verify_against(arr[0, 12], os.path.join(args.features_dir, "embed", "wav2vec2-base", f"{sid}.npy"), sid, state)
            np.save(out_path, arr)
            n_ok += 1
        except Exception as e:
            n_err += 1
            logger.error(f"id={sid}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            tqdm.write(f"[SKIP] id={sid}: {type(e).__name__}: {e}")
    print(f"[INFO] audio 완료: 성공 {n_ok}, 스킵 {n_skip}, 에러 {n_err} -> {out_dir}")


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 사용 불가 -> CPU", file=sys.stderr)
        args.device = "cpu"
    items = load_metadata(args.metadata_path)
    if args.limit is not None:
        items = items[: args.limit]
    print(f"[INFO] 대상 샘플 {len(items)}개")
    os.makedirs(args.log_dir, exist_ok=True)
    logger = logging.getLogger("layers_audio")
    logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, "extract_layers_audio_errors.log")))
    run_audio(items, args, logger)


if __name__ == "__main__":
    main()
