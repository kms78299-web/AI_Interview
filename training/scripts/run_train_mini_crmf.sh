#!/bin/bash

#SBATCH --job-name=recruitview_mini_crmf
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -w ariel-v7
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# mini CRMF 학습 (code/train_mini_crmf.py) — 인자를 그대로 전달하는 래퍼.
#
# Usage:
#   sbatch scripts/run_train_mini_crmf.sh --encoders wav2vec2-base bert-base-uncased --fusion prefusion
#   sbatch scripts/run_train_mini_crmf.sh --encoders wav2vec2-base bert-base-uncased --fusion concat
#   sbatch scripts/run_train_mini_crmf.sh --encoders wav2vec2-base bert-base-uncased videomae --smoke_test
#
# 주의: --smoke_test는 샘플 20개가 필요함. 지금은 embed/{audio,text,video}에
# id 0001~0005 5개씩만 있어서(테스트용 --limit 5) smoke_test가 바로는 안 돌아감.
# 먼저 extract_features.py(embed 트랙)로 더 많은 샘플을 뽑아야 함, 예:
#   sbatch scripts/run_extract.sh --track embed --modality audio --encoder wav2vec2-base --limit 30
#   sbatch scripts/run_extract.sh --track embed --modality text  --encoder bert-base-uncased --limit 30
#   sbatch scripts/run_extract.sh --track embed --modality video --encoder videomae --limit 30
# =============================================================================

set -e

# sbatch는 스크립트를 /var/spool/slurmd/.../slurm_script 로 복사해서 실행하기
# 때문에 BASH_SOURCE 기준 상대경로가 깨진다. 리포 경로를 직접 고정해서 사용.
SCRIPT_DIR="/ceph_data/kms78299/repos/GANG2/code"
cd "$SCRIPT_DIR"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python train_mini_crmf.py \
    --features_dir     "/ceph_data/kms78299/repos/GANG2/features" \
    --meta_path        "/ceph_data/kms78299/repos/GANG2/features/meta.csv" \
    --splits_dir        "/ceph_data/kms78299/repos/GANG2/splits" \
    --checkpoints_dir  "/ceph_data/kms78299/repos/GANG2/checkpoints" \
    --device cuda \
    "$@"
