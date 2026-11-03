"""
metadata.jsonl 로딩 + transcript 정제 + 라벨 벡터 구성.
"""

import json
import os
import re

import numpy as np

from config import METADATA_NON_LABEL_KEYS, TIMESTAMP_PATTERN

_TIMESTAMP_RE = re.compile(TIMESTAMP_PATTERN)
_WHITESPACE_RE = re.compile(r"\s+")


def load_metadata(metadata_path):
    """metadata.jsonl -> list[dict]. 빈 줄은 무시."""
    items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{metadata_path}:{line_no} JSON 파싱 실패: {e}")
    return items


def clean_transcript(raw_transcript):
    """"[00:01 - 00:11] Hello..." 형태에서 타임스탬프를 지우고 공백을 정리한다. 없거나 비면 빈 문자열."""
    if not raw_transcript:
        return ""
    text = _TIMESTAMP_RE.sub(" ", raw_transcript)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def discover_label_names(items):
    """metadata 레코드에서 숫자 필드(id, file_name, question, transcript 제외)를 라벨로 보고, 정렬한 이름 리스트를 반환한다."""
    label_names = set()
    for item in items:
        for key, value in item.items():
            if key in METADATA_NON_LABEL_KEYS:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                label_names.add(key)
    return sorted(label_names)


def get_label_vector(item, label_names):
    """item에서 label_names 순서대로 값을 뽑는다. 없는 필드는 NaN."""
    return np.array(
        [float(item[name]) if name in item and item[name] is not None else np.nan for name in label_names],
        dtype=np.float32,
    )


def video_path_for_item(item, video_root):
    return os.path.join(video_root, item["file_name"])
