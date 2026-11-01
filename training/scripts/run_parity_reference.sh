#!/bin/bash

#SBATCH --job-name=recruitview_parity
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -w ariel-v7
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# 추론 검증용 기준값 생성 (code/make_parity_reference.py) — 인자를 그대로 전달하는 래퍼.
#
# Usage (영상이 ariel-v7 로컬 디스크에 있어서 그 노드에서 실행됨):
#   sbatch run_parity_reference.sh
# 결과: /ceph_data/kms78299/model_parity/ (parity_reference.json + videos/ 5개)
# =============================================================================

set -e

SCRIPT_DIR="/ceph_data/kms78299/repos/GANG2/code"
cd "$SCRIPT_DIR"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python make_parity_reference.py "$@"
