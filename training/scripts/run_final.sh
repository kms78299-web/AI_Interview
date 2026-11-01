#!/bin/bash

#SBATCH --job-name=recruitview_final
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -w ariel-v7
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# 현재 최선 구성 학습/저장 (code/train_final_ensemble.py) — 인자를 그대로 전달하는 래퍼.
#
# Usage:
#   sbatch run_final.sh --smoke   # 동작 확인(결과 해석 금지)
#   sbatch run_final.sh           # 본 학습 (audio 6층 + 규제 R1 + seed 5개 앙상블)
# 결과: checkpoints/final_audio6_R1_ens5/ (이후 model/ 패키지로 복사)
# =============================================================================

set -e

SCRIPT_DIR="/ceph_data/kms78299/repos/GANG2/code"
cd "$SCRIPT_DIR"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python train_final_ensemble.py --device cuda "$@"
