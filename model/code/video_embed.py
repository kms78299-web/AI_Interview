"""
VideoMAE 기반 video embedding — Track 1(embed), MLP 입력용.

AU/gaze/pose 같은 해석 가능한 통계(video_features.py, Track 2: stats)와는
완전히 별개. frozen encoder, forward만 하고 fine-tuning은 안 함.

프레임 리사이즈/정규화(224x224 등)는 VideoMAEImageProcessor가 체크포인트의
preprocessor_config.json 기준으로 알아서 처리해서 여기서 직접 다루지 않음 —
config.VIDEO_EMBED_IMAGE_SIZE는 참고용 상수(실제로 HF config.json에서
image_size=224로 확인됨)일 뿐 이 파일 코드에서 직접 쓰진 않는다.
"""

import numpy as np

from config import VIDEO_EMBED_NUM_FRAMES


def _patch_qkv_bias(model, model_id):
    """MCG-NJU/videomae-base 체크포인트의 attention bias 네이밍 불일치를 고친다.

    safetensors 헤더를 직접 읽어서 확인한 사실(추측 아님): 체크포인트에는
    "query.bias"/"value.bias"가 아니라 "q_bias"/"v_bias"라는 이름으로 bias가
    저장돼 있고, "key.bias"는 아예 없음(BEiT 계열 설계 — key는 원래 bias가 없고
    query/value만 따로 bias를 둠). 현재 transformers의 VideoMAEModel 로더는 이
    옛날 네이밍을 자동으로 못 찾아서, 그냥 로드하면 각 레이어의
    query/key/value.bias 36개(12 layers x 3)가 전부 랜덤 초기화된 채로 남는다
    (실제로 LOAD REPORT의 MISSING/UNEXPECTED로 확인함). weight는 전부 정상 로드됨
    — bias만 이 문제가 있음.

    고치는 법: 체크포인트의 원본 state_dict(허깅페이스 캐시에 이미 받아져 있는
    파일, 재다운로드 없음)에서 q_bias/v_bias 값을 직접 꺼내 query.bias/value.bias
    에 복사하고, key.bias는 원래 설계대로 0으로 고정한다.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    try:
        ckpt_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        raw = load_file(ckpt_path)
    except Exception as e:
        print(f"[WARN] {model_id}: q_bias/v_bias 패치용 원본 state_dict를 못 읽음 "
              f"({type(e).__name__}: {e}) — bias는 랜덤 초기화 상태로 남음.")
        return model

    n_patched = 0
    with torch.no_grad():
        for i, layer in enumerate(model.encoder.layer):
            attn = layer.attention.attention
            q_key = f"videomae.encoder.layer.{i}.attention.attention.q_bias"
            v_key = f"videomae.encoder.layer.{i}.attention.attention.v_bias"
            if q_key not in raw or v_key not in raw:
                continue  # 이 체크포인트는 구조가 다름 -> 패치 스킵 (조용히 넘어감)
            attn.query.bias.copy_(raw[q_key].to(attn.query.bias.dtype))
            attn.value.bias.copy_(raw[v_key].to(attn.value.bias.dtype))
            attn.key.bias.zero_()  # 원래 설계상 key는 bias가 없음(0으로 고정)
            n_patched += 1

    if n_patched == len(model.encoder.layer):
        print(f"[INFO] {model_id}: attention q/v bias {n_patched}개 레이어 전부 원본값으로 패치 완료 "
              f"(key.bias는 설계상 0으로 고정)")
    elif n_patched > 0:
        print(f"[WARN] {model_id}: attention bias {n_patched}/{len(model.encoder.layer)} 레이어만 패치됨")
    else:
        print(f"[WARN] {model_id}: q_bias/v_bias 키를 찾지 못해 패치 안 함 (다른 체크포인트 구조일 수 있음)")
    return model


def load_videomae(model_id, device):
    from transformers import VideoMAEImageProcessor, VideoMAEModel

    processor = VideoMAEImageProcessor.from_pretrained(model_id)
    model = VideoMAEModel.from_pretrained(model_id).to(device).eval()
    model = _patch_qkv_bias(model, model_id)
    return processor, model


def _sample_frames_uniform(video_path, num_frames=VIDEO_EMBED_NUM_FRAMES):
    """영상 전체에서 num_frames개를 균등 간격으로 뽑아 RGB np.array 리스트로 반환.

    VideoMAE는 학습 때 고정된 프레임 수(기본 16)로 position embedding을 만들어서,
    그 수를 맞춰서 넣어야 함 — 영상이 그보다 짧으면 마지막 프레임을 반복해서 채움.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_by_idx = {}
    if total > 0:
        idx_set = set(np.linspace(0, total - 1, num_frames).round().astype(int).tolist())
        i = 0
        while len(frames_by_idx) < len(idx_set):
            ok, frame = cap.read()
            if not ok:
                break
            if i in idx_set:
                frames_by_idx[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    else:
        # 컨테이너에 따라 프레임 수를 못 읽는 경우가 있어서, 끝까지 읽어서 직접 센다.
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_by_idx[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    cap.release()

    if not frames_by_idx:
        raise RuntimeError(f"프레임을 하나도 못 읽음: {video_path}")

    ordered_keys = sorted(frames_by_idx.keys())
    if total <= 0:
        # 전체를 다 읽었으니 여기서 균등 샘플링
        pick_idx = np.linspace(0, len(ordered_keys) - 1, num_frames).round().astype(int)
        ordered_keys = [ordered_keys[i] for i in pick_idx]

    frames = [frames_by_idx[k] for k in ordered_keys]
    while len(frames) < num_frames:  # 영상이 num_frames보다 짧을 때 마지막 프레임으로 패딩
        frames.append(frames[-1])
    return frames[:num_frames]


def extract_videomae_embedding(video_path, processor, model, device, num_frames=VIDEO_EMBED_NUM_FRAMES):
    """영상 하나 -> 768차원 임베딩 (시간/패치 토큰 전체 mean pooling)."""
    import torch

    frames = _sample_frames_uniform(video_path, num_frames=num_frames)
    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy().astype(np.float32)
    return embedding
