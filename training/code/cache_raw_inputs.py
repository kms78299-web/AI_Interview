"""
encoder fine-tuning용 "원본 입력" 캐시 생성 (최종 임베딩 캐시가 아님).

fine-tuning은 encoder의 마지막 층 일부가 매 step 바뀌므로, features/embed/ 에 이미 저장된
최종 임베딩(768차원)을 쓰면 학습이 무효가 된다. 대신 encoder에 들어가기 전 입력을
디코딩/전처리까지만 해서 저장해 두고, 학습 때 이 입력을 encoder에 통과시킨다.
(영상 디코딩이 매 step마다 반복되면 너무 느려서 디코딩 결과만 미리 저장)

저장:
  features/raw/video_pixels/{id}.npy  float16 (16,3,224,224)  — 기존 임베딩 추출과 같은
      _sample_frames_uniform(16프레임 균등) + VideoMAEImageProcessor 결과 그대로
  features/raw/audio_wave/{id}.npy    float32 (L,)             — ffmpeg 16kHz mono -> librosa
      (feature extractor 정규화는 학습 때 적용, 기존 extract_wav2vec2_embedding과 동일)
텍스트는 캐시 불필요(metadata.jsonl의 transcript를 학습 때 바로 토큰화).

사용 (영상 파일은 계산 노드의 /local_datasets에 있으므로 sbatch로):
    sbatch run_cache_raw.sh --modality video
    sbatch run_cache_raw.sh --modality audio
"""

import argparse
import os
import tempfile

import numpy as np
from tqdm import tqdm

from audio_features import _extract_wav
from config import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_FEATURES_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_VIDEO_ROOT,
    VIDEO_EMBED_ENCODERS,
    VIDEO_EMBED_NUM_FRAMES,
)
from data_utils import load_metadata, video_path_for_item
from video_embed import _sample_frames_uniform


def parse_args():
    p = argparse.ArgumentParser(description="fine-tuning용 원본 입력 캐시 생성")
    p.add_argument("--modality", required=True, choices=["video", "audio"])
    p.add_argument("--metadata_path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    items = load_metadata(args.metadata_path)
    if args.limit is not None:
        items = items[: args.limit]

    sub = "video_pixels" if args.modality == "video" else "audio_wave"
    out_dir = os.path.join(args.features_dir, "raw", sub)
    os.makedirs(out_dir, exist_ok=True)

    processor = None
    if args.modality == "video":
        from transformers import VideoMAEImageProcessor
        processor = VideoMAEImageProcessor.from_pretrained(VIDEO_EMBED_ENCODERS["videomae"])

    n_ok = n_skip = n_err = 0
    for item in tqdm(items, desc=f"raw/{sub}"):
        out_path = os.path.join(out_dir, f"{item['id']}.npy")
        if not args.overwrite and os.path.exists(out_path):
            n_skip += 1
            continue
        video_path = video_path_for_item(item, args.video_root)
        try:
            if args.modality == "video":
                frames = _sample_frames_uniform(video_path, num_frames=VIDEO_EMBED_NUM_FRAMES)
                pixels = processor(frames, return_tensors="pt")["pixel_values"][0]  # (16,3,224,224)
                np.save(out_path, pixels.numpy().astype(np.float16))
            else:
                import librosa
                with tempfile.TemporaryDirectory() as tmp:
                    wav = os.path.join(tmp, "audio.wav")
                    _extract_wav(video_path, wav, sr=AUDIO_SAMPLE_RATE)
                    y, _ = librosa.load(wav, sr=AUDIO_SAMPLE_RATE, mono=True)
                if y.size == 0:
                    raise RuntimeError("오디오 길이 0")
                np.save(out_path, y.astype(np.float32))
            n_ok += 1
        except Exception as e:  # 한 샘플 실패가 전체를 멈추지 않게, 로그 남기고 계속
            n_err += 1
            tqdm.write(f"[SKIP] id={item['id']}: {type(e).__name__}: {e}")

    print(f"[INFO] 완료: 성공 {n_ok}, 스킵(이미 있음) {n_skip}, 에러 {n_err}  -> {out_dir}")
    if n_err:
        print("[WARN] 에러가 난 샘플은 fine-tuning에서 쓸 수 없음. 개수가 0이 될 때까지 원인 확인할 것.")


if __name__ == "__main__":
    main()
