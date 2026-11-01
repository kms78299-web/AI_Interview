"""
z-score 라벨을 0~100 백분위로 변환.

RECRUITVIEW 라벨(metadata.jsonl의 interview_score 등)은 절대 척도(1~100점 같은
거)가 아니라, 임상 심리학자들의 27,000+ 쌍비교(pairwise comparison) 데이터에서
Multinomial Logit + nuclear norm regularization으로 역산한 "상대적 잠재
능력치"임 (RECRUITVIEW 논문 arXiv:2512.00450 / GitHub README 참고). 그래서
z-score를 "원래 몇 점짜리 답변이었는지"로 복원하는 공식 자체가 없다.

대신 여기서는 전체 데이터셋(metadata.jsonl 2011개 샘플) 안에서 이 값이 상위
몇 %인지를 0~100 백분위로 보여준다 — "이 사람이 다른 지원자들 대비 상대적으로
어느 위치냐"는 원래 라벨의 의미(상대 비교)를 그대로 보존하는 해석이라 min-max나
정규분포 가정보다 안전함.
"""

import numpy as np


def build_label_distributions(items, label_names):
    """label_name -> 전체 샘플의 z-score를 정렬한 numpy 배열.

    items: data_utils.load_metadata()로 읽은 전체 metadata (일부만 넣지 말 것 —
    백분위는 전체 분포 기준이어야 의미가 있음).
    """
    dists = {}
    for name in label_names:
        values = [
            item[name] for item in items
            if name in item and item[name] is not None and isinstance(item[name], (int, float))
        ]
        dists[name] = np.sort(np.array(values, dtype=np.float64))
    return dists


def zscore_to_percentile(z, sorted_distribution):
    """sorted_distribution(오름차순 정렬된 numpy 배열) 안에서 z가 상위 몇 %인지.

    0~100 사이 float. z보다 작거나 같은 값의 비율 * 100 (= "이 값 이하가 몇 %").
    z나 분포가 없으면 None.
    """
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
