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
    """"[00:01 - 00:11] Hello everyone..." -> "Hello everyone...".

    타임스탬프 패턴을 전부 제거하고 공백을 정리한다.
    transcript가 없거나(None) 빈 문자열이면 빈 문자열을 반환한다.
    """
    if not raw_transcript:
        return ""
    text = _TIMESTAMP_RE.sub(" ", raw_transcript)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def discover_label_names(items):
    """metadata 레코드들에서 라벨 필드 이름을 동적으로 뽑아낸다.

    metadata.jsonl에 12개 라벨이 있다고 했지만 실제 필드명 전체가 예시에
    나와있지 않아서, id/file_name/question/transcript를 제외한 나머지
    숫자(int/float) 필드를 라벨로 취급한다. 레코드마다 필드 구성이 다를 수
    있으니 전체 아이템을 훑어서 합집합을 만들고 정렬해 순서를 고정한다.
    """
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
