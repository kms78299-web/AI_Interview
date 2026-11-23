#!/bin/bash

#SBATCH --job-name=llm_eval_prepare
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -w ariel-v7
#SBATCH -o /ceph_data/kms78299/repos/GANG2/logs/slurm-%A-%x.out

# =============================================================================
# LLM 프롬프트 검증 1단계 (code/llm_eval_prepare.py): 사람별로 서비스와 같은 analyze_session을 돌려 A/B/C 프롬프트 저장
# 영상이 ariel-v7에만 있어서 -w ariel-v7. Whisper, 모델, 측정값 추출에 GPU 사용.
#
# Usage:
#   sbatch run_llm_eval_prepare.sh --split val --limit 2   # 소량 확인 (이어서 전체 돌리면 된 사람은 건너뜀)
#   sbatch run_llm_eval_prepare.sh --split val             # -> llm_eval/val/analysis/{user_no}.json
#   sbatch run_llm_eval_prepare.sh --split test            # 최종 프롬프트가 정해진 뒤 한 번만
# =============================================================================

set -e

cd "/ceph_data/kms78299/repos/GANG2/code"

source /ceph_data/kms78299/anaconda3/etc/profile.d/conda.sh
conda activate recruitview

python llm_eval_prepare.py "$@"
