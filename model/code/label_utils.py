"""
z-score 라벨을 0~100 백분위로 변환한다.

RecruitView 라벨은 쌍비교 데이터에서 추정한 상대 점수라 원래 점수로 되돌릴 수 없다(arXiv:2512.00450).
그래서 전체 데이터셋(2011개) 안에서 이 값이 상위 몇 %인지를 백분위로 보여 준다.
"""

import numpy as np


def build_label_distributions(items, label_names):
    """label_name -> 전체 샘플 z-score를 정렬한 배열. 전체 metadata를 넘겨야 한다(백분위는 전체 분포 기준)."""
    dists = {}
    for name in label_names:
        values = [
            item[name] for item in items
            if name in item and item[name] is not None and isinstance(item[name], (int, float))
        ]
        dists[name] = np.sort(np.array(values, dtype=np.float64))
    return dists


def zscore_to_percentile(z, sorted_distribution):
    """sorted_distribution에서 z 이하인 값의 비율 * 100 (0~100). z나 분포가 없으면 None."""
    if z is None or sorted_distribution is None or sorted_distribution.size == 0:
        return None
    try:
        z = float(z)
    except (TypeError, ValueError):
        return None
    rank = np.searchsorted(sorted_distribution, z, side="right")
    return float(rank / sorted_distribution.size * 100.0)


def percentile_to_band(p):
    """0~100 백분위 -> 하/중/상 (3등분). None이면 None."""
    if p is None:
        return None
    if p <= 100.0 / 3:
        return "하"
    if p <= 200.0 / 3:
        return "중"
    return "상"


def label_dict_to_percentiles(label_values, distributions):
    """{label_name: z} -> {label_name: percentile(0~100) or None}"""
    return {
        name: zscore_to_percentile(z, distributions.get(name))
        for name, z in label_values.items()
    }
