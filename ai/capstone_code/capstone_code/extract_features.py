"""
RECRUITVIEW Video/Audio/Text feature 추출 메인 파이프라인.

    python extract_features.py \
        --metadata_path /ceph_data/kms78299/repos/GANG2/metadata.jsonl \
        --video_root    /local_datasets/RecruitView \
        --output_dir    /ceph_data/kms78299/repos/GANG2/recruitview_features \
        --embedding_dir /ceph_data/kms78299/repos/GANG2/audio_embeddings

동작:
  1. metadata.jsonl 로딩
  2. 라벨 필드 자동 탐지 (id/file_name/question/transcript 제외 나머지 숫자 필드)
  3. 각 샘플마다 feature 추출 -> 두 폴더에 나눠서 저장
     - output_dir/{id}_video.npy, {id}_audio.npy, {id}_text.npy
       : 사람이 해석 가능한 통계 (LLM 프롬프트용, analyze_interviews_llm.py 참고)
     - embedding_dir/{id}_audio_embedding.npy
       : wav2vec2 임베딩 (MLP 입력용 "무거운" 벡터, 숫자 자체는 해석 불가능).
         text는 이미 BERT 임베딩(text.npy 앞 768차원)이 이 역할을 겸하고 있어서
         별도 파일이 필요 없음.
  4. 두 폴더에 걸쳐 네 파일이 모두 존재하는 샘플은 스킵 (resume 지원)
  5. 에러 발생 샘플은 로그 남기고 스킵 (부분 저장 안 함 -> 다음 실행에서 재시도됨)
  6. 끝나고 나면 완료된 모든 샘플을 다시 스캔해서 output_dir에 labels.npy / ids.npy / label_names.json 생성
  7. feature 통계 출력
"""

import argparse
import json
import logging
import os
import sys
import traceback

import numpy as np
from tqdm import tqdm

from config import (
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_ROOT,
    DEFAULT_WAV2VEC2_MODEL_ID,
    TEXT_MODEL_ENGLISH,
    TEXT_MODEL_KOREAN,
)
from data_utils import (
    clean_transcript,
    discover_label_names,
    get_label_vector,
    load_metadata,
    video_path_for_item,
)
from video_features import extract_video_features, openface_available
from audio_features import extract_audio_features, extract_wav2vec2_embedding, load_wav2vec2
from text_features import extract_text_features, load_bert


def parse_args():
    p = argparse.ArgumentParser(description="RECRUITVIEW feature extraction")
    p.add_argument("--metadata_path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR,
                    help="wav2vec2 등 해석 불가능한 MLP용 임베딩을 저장할 폴더 (output_dir과 분리)")
    p.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--limit", type=int, default=None, help="테스트용: 앞에서 N개만 처리")
    p.add_argument("--overwrite", action="store_true", help="이미 완료된 샘플도 다시 추출")
    p.add_argument("--text_model_ko", default=TEXT_MODEL_KOREAN)
    p.add_argument("--text_model_en", default=TEXT_MODEL_ENGLISH)
    p.add_argument("--wav2vec2_model_id", default=DEFAULT_WAV2VEC2_MODEL_ID)
    return p.parse_args()


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("recruitview_extract")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(log_dir, "extract_errors.log"))
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def sample_paths(output_dir, embedding_dir, sample_id):
    return {
        "video": os.path.join(output_dir, f"{sample_id}_video.npy"),
        "audio": os.path.join(output_dir, f"{sample_id}_audio.npy"),
        "audio_embedding": os.path.join(embedding_dir, f"{sample_id}_audio_embedding.npy"),
        "text": os.path.join(output_dir, f"{sample_id}_text.npy"),
    }


def is_sample_complete(output_dir, embedding_dir, sample_id):
    paths = sample_paths(output_dir, embedding_dir, sample_id)
    return all(os.path.exists(p) for p in paths.values())


def make_bert_loader(text_model_ko, text_model_en, device, logger):
    cache = {}

    def get_bert(is_ko):
        key = "ko" if is_ko else "en"
        if key not in cache:
            model_name = text_model_ko if is_ko else text_model_en
            logger.info(f"BERT 모델 로딩: {model_name} ({key})")
            cache[key] = load_bert(model_name, device)
        return cache[key]

    return get_bert


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.embedding_dir, exist_ok=True)
    logger = setup_logger(args.log_dir)

    import torch

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("[WARN] CUDA를 사용할 수 없어 CPU로 대체합니다.", file=sys.stderr)

    print(f"[INFO] Video backend: {'OpenFace' if openface_available() else 'MediaPipe (fallback)'}")
    print(f"[INFO] Device: {device}")

    items = load_metadata(args.metadata_path)
    if args.limit is not None:
        items = items[: args.limit]
    print(f"[INFO] metadata 샘플 수: {len(items)}")

    label_names = discover_label_names(items)
    print(f"[INFO] 자동 탐지된 라벨 필드 ({len(label_names)}개): {label_names}")

    id_to_item = {item["id"]: item for item in items}
    get_bert = make_bert_loader(args.text_model_ko, args.text_model_en, device, logger)

    print(f"[INFO] wav2vec2 모델 로딩: {args.wav2vec2_model_id}")
    wav2vec2_extractor, wav2vec2_model = load_wav2vec2(args.wav2vec2_model_id, device)

    n_skipped, n_success, n_error = 0, 0, 0

    for item in tqdm(items, desc="extracting"):
        sample_id = item["id"]

        if not args.overwrite and is_sample_complete(args.output_dir, args.embedding_dir, sample_id):
            n_skipped += 1
            continue

        video_path = video_path_for_item(item, args.video_root)
        paths = sample_paths(args.output_dir, args.embedding_dir, sample_id)

        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"영상 파일 없음: {video_path}")

            video_feat = extract_video_features(video_path)
            audio_feat = extract_audio_features(video_path)
            audio_embedding = extract_wav2vec2_embedding(
                video_path, wav2vec2_extractor, wav2vec2_model, device
            )

            clean_text = clean_transcript(item.get("transcript"))
            if not clean_text:
                logger.warning(f"id={sample_id}: transcript가 비어있음 -> text feature를 0으로 채움")
            text_feat = extract_text_features(clean_text, get_bert, device)

        except Exception as e:
            n_error += 1
            logger.error(f"id={sample_id}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            tqdm.write(f"[SKIP] id={sample_id}: {type(e).__name__}: {e}")
            continue

        np.save(paths["video"], video_feat)
        np.save(paths["audio"], audio_feat)
        np.save(paths["audio_embedding"], audio_embedding)
        np.save(paths["text"], text_feat)
        n_success += 1

    print(f"[INFO] 이번 실행: 성공 {n_success}, 스킵(이미 완료) {n_skipped}, 에러 {n_error}")

    # 완료된 모든 샘플(과거 실행분 포함)을 다시 스캔해서 labels.npy 재구성
    completed_ids = sorted(
        sid for sid in id_to_item if is_sample_complete(args.output_dir, args.embedding_dir, sid)
    )
    if not completed_ids:
        print("[WARN] 완료된 샘플이 하나도 없어 labels.npy를 생성하지 않음.")
        return

    label_matrix = np.stack(
        [get_label_vector(id_to_item[sid], label_names) for sid in completed_ids]
    )
    np.save(os.path.join(args.output_dir, "labels.npy"), label_matrix)
    np.save(os.path.join(args.output_dir, "ids.npy"), np.array(completed_ids))
    with open(os.path.join(args.output_dir, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)

    print_stats(args.output_dir, args.embedding_dir, completed_ids, label_matrix, label_names)


def print_stats(output_dir, embedding_dir, completed_ids, label_matrix, label_names):
    video_mat = np.stack([np.load(os.path.join(output_dir, f"{sid}_video.npy")) for sid in completed_ids])
    audio_mat = np.stack([np.load(os.path.join(output_dir, f"{sid}_audio.npy")) for sid in completed_ids])
    audio_emb_mat = np.stack(
        [np.load(os.path.join(embedding_dir, f"{sid}_audio_embedding.npy")) for sid in completed_ids]
    )
    text_mat = np.stack([np.load(os.path.join(output_dir, f"{sid}_text.npy")) for sid in completed_ids])

    print("=" * 60)
    print(f"완료된 샘플 수: {len(completed_ids)}")
    print(f"video feature: shape={video_mat.shape}, mean={video_mat.mean():.4f}, std={video_mat.std():.4f}")
    print(f"audio feature: shape={audio_mat.shape}, mean={audio_mat.mean():.4f}, std={audio_mat.std():.4f}")
    print(f"audio embedding(wav2vec2): shape={audio_emb_mat.shape}, "
          f"mean={audio_emb_mat.mean():.4f}, std={audio_emb_mat.std():.4f}")
    print(f"text  feature: shape={text_mat.shape}, mean={text_mat.mean():.4f}, std={text_mat.std():.4f}")
    print(f"labels: shape={label_matrix.shape} (nan 포함 가능, 컬럼 순서는 label_names.json 참고)")
    print(f"label 필드: {label_names}")
    print(f"저장 위치: output_dir={output_dir}, embedding_dir={embedding_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
