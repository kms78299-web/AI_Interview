#!/bin/bash

#SBATCH --job-name=recruitview_stats
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -w ariel-v7
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# LLM 피드백용 통계 추출 (code/extract_stats.py): audio(librosa) + text(대본 통계) + video(MediaPipe 얼굴)
# 영상 파일이 ariel-v7에만 있어서 -w ariel-v7. GPU는 쓰지 않지만 이 클러스터는 gres가 필요해서 요청함.
#
# Usage:
#   sbatch run_extract_stats.sh --limit 5   # 소량 확인 (이어서 전체 돌리면 된 건 스킵)
#   sbatch run_extract_stats.sh              # 전체 -> features/stats/{id}.json
#   (전체가 끝난 뒤, 영상 없이 로그인 노드에서) python extract_stats.py --reference
# =============================================================================

set -e

cd "/ceph_data/kms78299/repos/GANG2/code"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python extract_stats.py \
    --metadata_path "/ceph_data/kms78299/repos/GANG2/metadata.jsonl" \
    --video_root    "/local_datasets/RecruitView" \
    --features_dir  "/ceph_data/kms78299/repos/GANG2/features" \
    --log_dir       "/ceph_data/kms78299/repos/GANG2/logs" \
    "$@"
