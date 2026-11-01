#!/bin/bash

#SBATCH --job-name=recruitview_feat
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -w ariel-v7
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# RECRUITVIEW feature 추출 (embed/stats 두 트랙) — 인자를 그대로 전달하는 래퍼.
# --track/--modality/--encoder는 조합이 많아서 하드코딩 안 하고 전부 "$@"로 받음.
#
# Usage (Track 1: embed — 지금 구현됨):
#   sbatch scripts/run_extract.sh --track embed --modality audio --encoder wav2vec2-base --limit 5
#   sbatch scripts/run_extract.sh --track embed --modality text  --encoder bert-base-uncased --limit 5
#   sbatch scripts/run_extract.sh --track embed --modality video --encoder videomae --limit 5
#
# Track 2(stats)는 아직 미구현 (code/extract_features.py 참고).
#
# 예전 단일 파이프라인(video/audio/text npy를 한 번에 뽑던 거)은
# code/extract_features_legacy.py 로 남아있음 — 그거 쓰려면 이 스크립트 말고
# `python extract_features_legacy.py ...`로 직접 실행.
# =============================================================================

set -e

# sbatch는 스크립트를 /var/spool/slurmd/.../slurm_script 로 복사해서 실행하기
# 때문에 BASH_SOURCE 기준 상대경로가 깨진다. 리포 경로를 직접 고정해서 사용.
SCRIPT_DIR="/ceph_data/kms78299/repos/GANG2/code"
cd "$SCRIPT_DIR"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python extract_features.py \
    --metadata_path "/ceph_data/kms78299/repos/GANG2/metadata.jsonl" \
    --video_root    "/local_datasets/RecruitView" \
    --features_dir  "/ceph_data/kms78299/repos/GANG2/features" \
    --log_dir       "/ceph_data/kms78299/repos/GANG2/logs" \
    --device        cuda \
    "$@"
