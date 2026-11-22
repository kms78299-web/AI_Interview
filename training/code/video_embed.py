"""
VideoMAE로 영상 임베딩(768차원)을 뽑는다. encoder는 고정하고 forward만 한다.
"""

import numpy as np

from config import VIDEO_EMBED_NUM_FRAMES


def _patch_qkv_bias(model, model_id):
    """videomae-base 체크포인트의 q_bias/v_bias를 query/value.bias로 옮기고 key.bias는 0으로 둔다.

        체크포인트는 bias를 "q_bias"/"v_bias"로 저장하는데 transformers가 이 이름을 못 찾아서,
        그대로 로드하면 bias가 랜덤 초기화된다."""
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
    """영상 전체에서 num_frames개를 균등 간격으로 뽑아 RGB 프레임 리스트로 반환한다. 짧은 영상은 마지막 프레임으로 채운다."""
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


def extract_videomae_temporal_embedding(video_path, processor, model, device, num_frames=VIDEO_EMBED_NUM_FRAMES):
    """영상 1개 -> (num_frames // tubelet_size, 768) 시간축 시퀀스. 시간 위치별로 공간 패치만 평균낸다."""
    import torch

    frames = _sample_frames_uniform(video_path, num_frames=num_frames)
    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state[0]  # (L, D), L = T' * H' * W'
    t_tokens = num_frames // model.config.tubelet_size
    length, dim = hidden.shape
    if length % t_tokens != 0:
        raise RuntimeError(f"토큰 수 {length}가 시간 위치 수 {t_tokens}로 나누어떨어지지 않음")
    seq = hidden.reshape(t_tokens, length // t_tokens, dim).mean(dim=1)
    return seq.cpu().numpy().astype(np.float32)
