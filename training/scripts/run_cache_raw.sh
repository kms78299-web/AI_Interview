#!/bin/bash

#SBATCH --job-name=recruitview_cacheraw
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -w ariel-v7
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# encoder fine-tuning용 원본 입력 캐시 생성 (code/cache_raw_inputs.py)
# 영상은 계산 노드의 /local_datasets에 있어서 sbatch로 실행해야 함. GPU는 쓰지 않음.
#
# Usage (fine-tuning할 모달리티만 필요, text는 캐시 불필요):
#   sbatch run_cache_raw.sh --modality video     # 약 10GB, 영상 디코딩이라 30분~1시간
#   sbatch run_cache_raw.sh --modality audio
# =============================================================================

set -e

cd "/ceph_data/kms78299/repos/GANG2/code"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python cache_raw_inputs.py \
    --metadata_path "/ceph_data/kms78299/repos/GANG2/metadata.jsonl" \
    --video_root    "/local_datasets/RecruitView" \
    --features_dir  "/ceph_data/kms78299/repos/GANG2/features" \
    "$@"
