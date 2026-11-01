"""
features/meta.csv 생성 — train_mini_crmf.py가 쓰는 샘플별 메타/라벨 테이블.

metadata.jsonl에서 바로 만들 수 있어서 embed/stats 트랙 완료 여부와 무관하게
독립적으로 실행 가능. id, user_no, duration, question_id, has_transcript,
그리고 라벨 12개(discover_label_names로 자동 탐지, 원본 그대로 — 가공 안 함)를
담는다.

    python build_meta.py
"""

import argparse
import csv
import os

from config import DEFAULT_META_PATH, DEFAULT_METADATA_PATH
from data_utils import clean_transcript, discover_label_names, load_metadata


def parse_args():
    p = argparse.ArgumentParser(description="features/meta.csv 생성")
    p.add_argument("--metadata_path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--out_path", default=DEFAULT_META_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    items = load_metadata(args.metadata_path)
    label_names = discover_label_names(items)
    print(f"[INFO] 샘플 수: {len(items)}, 라벨 필드({len(label_names)}개): {label_names}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fieldnames = ["id", "user_no", "duration", "question_id", "has_transcript"] + label_names

    n_no_user_no = 0
    with open(args.out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            if not item.get("user_no"):
                n_no_user_no += 1
            row = {
                "id": item["id"],
                "user_no": item.get("user_no", ""),
                "duration": item.get("duration", ""),
                "question_id": item.get("question_id", ""),
                "has_transcript": bool(clean_transcript(item.get("transcript"))),
            }
            for name in label_names:
                row[name] = item.get(name, "")
            writer.writerow(row)

    if n_no_user_no:
        print(f"[WARN] user_no가 비어있는 샘플 {n_no_user_no}개 — user_no 기준 split에서 제외될 수 있음")
    print(f"[INFO] 저장 완료: {args.out_path}")


if __name__ == "__main__":
    main()
