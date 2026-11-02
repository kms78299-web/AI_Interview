"""
RECRUITVIEW feature 추출 — Track 1(embed)/Track 2(stats) 두 갈래 CLI.

Track 1 (embed, MLP 입력용 — 지금 구현됨):
    python extract_features.py --track embed --modality audio --encoder wav2vec2-base --limit 5
    python extract_features.py --track embed --modality text  --encoder bert-base-uncased --limit 5
    python extract_features.py --track embed --modality video --encoder videomae --limit 5

  - 모달리티별로 따로 실행/저장 (한 모달리티가 실패해도 다른 모달리티엔 영향 없음)
  - encoder는 frozen, forward만 함 (fine-tuning 없음)
  - 저장: features/embed/{encoder}/{id}.npy (float32, shape=(768,))
  - 이미 있는 결과는 스킵(resume), 에러난 샘플은 로그 남기고 스킵

Track 2 (stats, LLM 피드백용): extract_stats.py가 audio(librosa)/text(대본 통계)/video(video_stats.py, MediaPipe)를
  features/stats/{id}.json 으로 저장한다. (--track stats는 extract_stats.run_batch로 위임)

주의: 예전 extract_features.py(video/audio/text npy를 recruitview_features/ +
audio_embeddings/ 에 한 번에 저장하던 단일 파이프라인)는 code/extract_features_legacy.py
로 그대로 남겨뒀음 — 필요하면 거기서 재사용 가능.
"""

import argparse
import logging
import os
import sys
import traceback

import numpy as np
from tqdm import tqdm

from config import (
    AUDIO_EMBED_ENCODERS,
    DEFAULT_FEATURES_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_VIDEO_ROOT,
    TEXT_EMBED_ENCODERS,
    VIDEO_EMBED_ENCODERS,
)
from data_utils import clean_transcript, load_metadata, video_path_for_item

_ENCODERS_BY_MODALITY = {
    "audio": AUDIO_EMBED_ENCODERS,
    "text": TEXT_EMBED_ENCODERS,
    "video": VIDEO_EMBED_ENCODERS,
}


def parse_args():
    p = argparse.ArgumentParser(description="RECRUITVIEW feature 추출 (embed/stats 두 트랙)")
    p.add_argument("--track", required=True, choices=["embed", "stats"])
    p.add_argument("--modality", choices=["audio", "text", "video"],
                    help="--track embed일 때 필수")
    p.add_argument("--encoder",
                    help="--track embed일 때 필수. audio=wav2vec2-base, "
                         "text=bert-base-uncased|roberta-base, video=videomae")
    p.add_argument("--metadata_path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR)
    p.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--limit", type=int, default=None, help="앞에서 N개만 처리 (테스트용)")
    p.add_argument("--start", type=int, default=0, help="구간 시작 인덱스")
    p.add_argument("--end", type=int, default=None, help="구간 끝 인덱스(exclusive)")
    p.add_argument("--overwrite", action="store_true", help="이미 완료된 샘플도 다시 추출")
    p.add_argument("--temporal", action="store_true",
                    help="--modality video 전용: 평균내지 않고 시간축 시퀀스(8,768)를 "
                         "features/embed_seq/{encoder}/ 에 저장")
    return p.parse_args()


def setup_logger(log_dir, name):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"recruitview_extract_{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(log_dir, f"extract_{name}_errors.log"))
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def select_items(items, start, end, limit):
    items = items[start:end]
    if limit is not None:
        items = items[:limit]
    return items


def run_embed_track(args):
    if args.modality is None or args.encoder is None:
        print("[ERROR] --track embed 는 --modality 와 --encoder 가 둘 다 필요함.", file=sys.stderr)
        sys.exit(1)

    encoders = _ENCODERS_BY_MODALITY[args.modality]
    if args.encoder not in encoders:
        print(f"[ERROR] modality={args.modality} 에서 지원하는 --encoder: {list(encoders.keys())} "
              f"(받은 값: {args.encoder})", file=sys.stderr)
        sys.exit(1)
    model_id = encoders[args.encoder]

    if args.temporal and args.modality != "video":
        print("[ERROR] --temporal 은 --modality video 에서만 가능.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(args.features_dir, "embed_seq" if args.temporal else "embed", args.encoder)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(args.log_dir, f"embed_{args.modality}_{args.encoder}{'_temporal' if args.temporal else ''}")

    import torch
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("[WARN] CUDA를 사용할 수 없어 CPU로 대체합니다.", file=sys.stderr)
    print(f"[INFO] modality={args.modality}  encoder={args.encoder} -> {model_id}  device={device}")

    items = load_metadata(args.metadata_path)
    items = select_items(items, args.start, args.end, args.limit)
    print(f"[INFO] 대상 샘플 수: {len(items)}")

    # 모달리티별 (로더, 추출함수) 준비 — 전부 frozen encoder, forward만.
    if args.modality == "audio":
        from audio_features import extract_wav2vec2_embedding, load_wav2vec2
        feature_extractor, model = load_wav2vec2(model_id, device)

        def _extract(item):
            video_path = video_path_for_item(item, args.video_root)
            return extract_wav2vec2_embedding(video_path, feature_extractor, model, device)

    elif args.modality == "text":
        from text_features import extract_text_embedding, load_bert
        tokenizer, model = load_bert(model_id, device)

        def _extract(item):
            clean_text = clean_transcript(item.get("transcript"))
            if not clean_text:
                logger.warning(f"id={item['id']}: transcript가 비어있음 -> 0벡터로 채움")
            return extract_text_embedding(clean_text, tokenizer, model, device)

    else:  # video
        from video_embed import extract_videomae_embedding, extract_videomae_temporal_embedding, load_videomae
        processor, model = load_videomae(model_id, device)
        extract_fn = extract_videomae_temporal_embedding if args.temporal else extract_videomae_embedding

        def _extract(item):
            video_path = video_path_for_item(item, args.video_root)
            return extract_fn(video_path, processor, model, device)

    n_success, n_skipped, n_error = 0, 0, 0
    for item in tqdm(items, desc=f"embed{'_seq' if args.temporal else ''}/{args.modality}/{args.encoder}"):
        sample_id = item["id"]
        out_path = os.path.join(out_dir, f"{sample_id}.npy")
        if not args.overwrite and os.path.exists(out_path):
            n_skipped += 1
            continue

        # audio/video는 영상 파일이 실제 있는지 먼저 확인 (text는 영상 필요 없음)
        if args.modality in ("audio", "video"):
            video_path = video_path_for_item(item, args.video_root)
            if not os.path.exists(video_path):
                n_error += 1
                logger.error(f"id={sample_id}: 영상 파일 없음: {video_path}")
                tqdm.write(f"[SKIP] id={sample_id}: 영상 파일 없음")
                continue

        try:
            vec = _extract(item)
        except Exception as e:
            n_error += 1
            logger.error(f"id={sample_id}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            tqdm.write(f"[SKIP] id={sample_id}: {type(e).__name__}: {e}")
            continue

        np.save(out_path, vec)
        n_success += 1

    print(f"[INFO] 완료: 성공 {n_success}, 스킵(이미완료) {n_skipped}, 에러 {n_error}")
    print(f"[INFO] 저장 위치: {out_dir}")


def run_stats_track(args):
    from extract_stats import run_batch
    run_batch(args)


def main():
    args = parse_args()
    if args.track == "embed":
        run_embed_track(args)
    else:
        run_stats_track(args)


if __name__ == "__main__":
    main()
