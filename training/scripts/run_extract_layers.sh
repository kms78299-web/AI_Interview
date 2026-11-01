#!/bin/bash

#SBATCH --job-name=recruitview_layers
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -w ariel-v7
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# 층별 임베딩 추출 (code/extract_layers.py) — 특징 비교용. 인자를 그대로 전달.
#
# Usage (audio 층별 임베딩만 추출 — 최종 모델의 audio 입력(6층)용. 먼저 run_cache_raw.sh --modality audio 필요):
#   sbatch run_extract_layers.sh --what audio --limit 5   # (선택) 소량 확인, 이어서 전체 돌리면 된 건 스킵
#   sbatch run_extract_layers.sh --what audio              # 약 45분
# =============================================================================

set -e

cd "/ceph_data/kms78299/repos/GANG2/code"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python extract_layers.py \
    --metadata_path "/ceph_data/kms78299/repos/GANG2/metadata.jsonl" \
    --video_root    "/local_datasets/RecruitView" \
    --features_dir  "/ceph_data/kms78299/repos/GANG2/features" \
    --log_dir       "/ceph_data/kms78299/repos/GANG2/logs" \
    --device cuda \
    "$@"
