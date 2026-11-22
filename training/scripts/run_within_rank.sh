#!/bin/bash

#SBATCH --job-name=recruitview_within_rank
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=0-8
#SBATCH --partition=batch_ugrad
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# 같은 사람 답변 순위 학습 실험 (code/train_within_rank.py). 캐시된 임베딩만 쓰므로 노드 제한 없음.
#
# Usage:
#   sbatch run_within_rank.sh --smoke                         # 동작 확인 (결과 해석 금지)
#   sbatch run_within_rank.sh                                 # text+audio+video, lam 0 / 0.5 / 1 / 2 (seed 5개씩)
#   sbatch run_within_rank.sh --modalities audio video        # audio+video
# 결과: checkpoints/within_rank_{modalities}/metrics_report.json (로그 끝에 비교표)
# =============================================================================

set -e

cd "/ceph_data/kms78299/repos/GANG2/code"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python train_within_rank.py --device cuda "$@"
