"""
Qwen2.5-Omni-7B로 면접 답변 몇 개를 실제로 분석해보는 데모 스크립트.

metadata.jsonl에서 영상 파일이 실제로 존재하는 샘플을 N개 골라서, 이미
GANG2에 있는 text/audio/video feature 추출 코드를 그대로 재사용해 각 샘플의
transcript + 오디오/영상 통계(수치)를 뽑고, dummy_data_v2-2.json의 5개 예시
(우수/평균/미흡/내용↑비언어↓/비언어↑내용↓)를 few-shot calibration으로
프롬프트에 넣는다. 여기에 더해 실제 분석 대상 영상 파일 자체도 Qwen한테 같이
넘겨서 — 수치 통계만 보고 짐작하는 게 아니라 실제 화면(표정/자세)과 음성
(톤/속도)을 직접 보고 듣고 판단하게 한다. Qwen에게 6개 면접 점수 + 카테고리 +
reasoning을 예측시킨다.

    python analyze_interviews_llm.py --num_samples 5

결과는 --output_path (기본 GANG2/llm_outputs/interview_analysis.json) 에
샘플별 {ground_truth, prediction, raw_output} 리스트로 저장된다. MLP 학습용
파이프라인(extract_features.py)과는 별개의 실험용 스크립트임.

영상을 통째로 넣으면 텍스트만 넣을 때보다 토큰/메모리/시간이 훨씬 많이 든다
(fps/max_frames로 조절 가능, 기본값은 보수적으로 낮게 잡아둠).
"""

import argparse
import json
import os
import sys
import traceback

from tqdm import tqdm

from audio_features import audio_vector_to_nested_dict, extract_audio_features
from config import (
    DEFAULT_DUMMY_DATA_PATH,
    DEFAULT_LLM_OUTPUT_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_QWEN_MODEL_ID,
    DEFAULT_VIDEO_ROOT,
    INTERVIEW_LABEL_NAMES,
    VIDEO_FEATURE_NAMES,
)
from data_utils import clean_transcript, load_metadata, video_path_for_item
from interview_prompts import build_analysis_prompt, load_calibration_examples
from label_utils import build_label_distributions, label_dict_to_percentiles
from text_features import extract_text_stats_for_prompt
from video_features import extract_video_features

SYSTEM_TEXT = (
    "You are an expert interview coach and multimodal behavior analyst. "
    "You always answer with a single strict JSON object and nothing else."
)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen2.5-Omni-7B interview analysis demo")
    p.add_argument("--metadata_path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--dummy_data_path", default=DEFAULT_DUMMY_DATA_PATH)
    p.add_argument("--model_id", default=DEFAULT_QWEN_MODEL_ID)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--cuda_device", default="0")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--output_path", default=DEFAULT_LLM_OUTPUT_PATH)
    p.add_argument("--fps", type=float, default=1.0, help="영상에서 초당 몇 프레임을 뽑아 Qwen에 줄지")
    p.add_argument("--max_frames", type=int, default=32, help="영상당 최대 프레임 수 (메모리/속도 제한)")
    return p.parse_args()


def pick_samples_with_video(items, video_root, num_samples):
    picked = []
    for item in items:
        video_path = video_path_for_item(item, video_root)
        if os.path.exists(video_path):
            picked.append(item)
        if len(picked) >= num_samples:
            break
    return picked


def build_sample_features(item, video_root):
    video_path = video_path_for_item(item, video_root)

    clean_text = clean_transcript(item.get("transcript"))
    text_stats = extract_text_stats_for_prompt(clean_text)

    audio_vec = extract_audio_features(video_path)
    audio_dict = audio_vector_to_nested_dict(audio_vec)

    video_vec = extract_video_features(video_path)
    video_dict = dict(zip(VIDEO_FEATURE_NAMES, [round(float(v), 4) for v in video_vec]))

    return {
        "id": item["id"],
        "question": item.get("question", ""),
        "text": text_stats,
        "audio": audio_dict,
        "video": video_dict,
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # torch/transformers가 CUDA_VISIBLE_DEVICES를 보기 전에 import되면 안 되므로
    # qwen_utils(및 그 안의 torch/transformers)는 env var를 세팅한 뒤에 import한다.
    from qwen_utils import load_qwen_omni_thinker, parse_json_object, run_chat

    items = load_metadata(args.metadata_path)
    # 백분위는 metadata.jsonl 전체(2011개) 분포 기준이어야 의미가 있음 — 이번에
    # 분석할 num_samples개가 아니라 items 전체로 계산.
    label_distributions = build_label_distributions(items, INTERVIEW_LABEL_NAMES)
    calibration_examples = load_calibration_examples(args.dummy_data_path)
    print(f"[INFO] calibration 예시: {list(calibration_examples.keys())}")

    samples = pick_samples_with_video(items, args.video_root, args.num_samples)
    print(f"[INFO] 영상이 존재하는 샘플 {len(samples)}/{args.num_samples}개 선택: "
          f"{[s['id'] for s in samples]}")
    if not samples:
        print(f"[ERROR] {args.video_root} 밑에서 영상을 하나도 못 찾음.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 모델 로딩: {args.model_id}")
    model, processor = load_qwen_omni_thinker(args.model_id)

    results = []
    for item in tqdm(samples, desc="analyzing"):
        sample_id = item["id"]
        ground_truth = {name: item.get(name) for name in INTERVIEW_LABEL_NAMES}
        record = {
            "id": sample_id,
            "ground_truth": ground_truth,
            "ground_truth_percentile": label_dict_to_percentiles(ground_truth, label_distributions),
            "prediction": None,
            "prediction_percentile": None,
            "raw_output": None,
            "error": None,
        }
        try:
            sample = build_sample_features(item, args.video_root)
            prompt = build_analysis_prompt(calibration_examples, sample)
            video_path = video_path_for_item(item, args.video_root)
            raw_output = run_chat(
                model, processor,
                contents=[
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": prompt},
                ],
                system_text=SYSTEM_TEXT,
                do_sample=False, temperature=0.0, top_p=1.0,
                max_new_tokens=args.max_new_tokens,
                use_audio_in_video=True, fps=args.fps, max_frames=args.max_frames,
            )
            record["raw_output"] = raw_output
            record["prediction"] = parse_json_object(raw_output)
            pred_labels = record["prediction"].get("labels", {}) if record["prediction"] else {}
            record["prediction_percentile"] = label_dict_to_percentiles(pred_labels, label_distributions)
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            tqdm.write(f"[ERROR] id={sample_id}: {record['error']}\n{traceback.format_exc()}")

        results.append(record)
        tqdm.write(f"--- id={sample_id} ---")
        tqdm.write(json.dumps(record["prediction"] or record["error"], ensure_ascii=False))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 저장 완료: {args.output_path}")

    print("=" * 78)
    print("gt/pred는 원래 z-score(mean=0, std~1), %ile은 metadata.jsonl 전체 2011개")
    print("샘플 분포 기준 백분위(0~100, 높을수록 상위권) — label_utils.py 참고")
    print("=" * 78)
    for r in results:
        pred_labels = (r["prediction"] or {}).get("labels", {}) if r["prediction"] else {}
        gt_pct = r["ground_truth_percentile"] or {}
        pred_pct = r["prediction_percentile"] or {}
        print(f"id={r['id']} category={((r['prediction'] or {}).get('category'))}")
        for name in INTERVIEW_LABEL_NAMES:
            gt = r["ground_truth"].get(name)
            pr = pred_labels.get(name)
            gt_str = f"{gt:.2f} ({gt_pct.get(name):.0f}%ile)" if gt is not None and gt_pct.get(name) is not None else str(gt)
            pr_str = f"{pr:.2f} ({pred_pct.get(name):.0f}%ile)" if pr is not None and pred_pct.get(name) is not None else str(pr)
            print(f"  {name:22s} gt={gt_str:>18s} pred={pr_str:>18s}")
        feedback = (r["prediction"] or {}).get("feedback") if r["prediction"] else None
        if feedback:
            print("  [강점]", "; ".join(feedback.get("strengths", [])))
            print("  [개선점]", "; ".join(feedback.get("improvements", [])))
            print("  [총평]", feedback.get("summary", ""))
    print("=" * 78)


if __name__ == "__main__":
    main()
