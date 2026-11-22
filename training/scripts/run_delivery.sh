#!/bin/bash

#SBATCH --job-name=recruitview_delivery
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=0-6
#SBATCH --partition=batch_ugrad
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# 전달력 전용 모델 실험 (code/train_delivery.py). 캐시된 임베딩만 쓰므로 노드 제한 없음.
#
# Usage:
#   sbatch run_delivery.sh --smoke                     # 동작 확인 (결과 해석 금지)
#   sbatch run_delivery.sh --label_mode raw            # audio+video, 원래 라벨
#   sbatch run_delivery.sh --label_mode length_resid   # audio+video, 길이 잔차 라벨
# 결과: checkpoints/delivery_audio-video_{label_mode}/metrics_report.json (로그 끝에 비교표)
# =============================================================================

set -e

cd "/ceph_data/kms78299/repos/GANG2/code"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python train_delivery.py --device cuda "$@"
