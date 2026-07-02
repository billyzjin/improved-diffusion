#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

if [ -z "${DATASET:-${1:-}}" ]; then
    echo "ERROR: set DATASET or pass it as the first argument."
    echo "Supported datasets are those handled by submit_image_folder_full_slate.sh."
    exit 1
fi

export DATASET=${DATASET:-$1}
export OBJECTIVES=hybrid
export SCHEDULES=${SCHEDULES:-linear,cosine,geometric_linear,geometric_cosine}
export HYBRID_VB_WEIGHTS=${HYBRID_VB_WEIGHTS:-0,1e-4,3e-4,1e-3,3e-3,1e-2}
export SUBMISSION_DIR=${SUBMISSION_DIR:-/project_gpfs/bata0/bjin0/${DATASET}_hybrid_vb_weight_grid_$(date +%Y%m%d_%H%M%S)}

echo "=========================================="
echo "Submitting hybrid VB-weight grid"
echo "Dataset: $DATASET"
echo "Schedules: $SCHEDULES"
echo "Hybrid VB weights: $HYBRID_VB_WEIGHTS"
echo "Submission dir: $SUBMISSION_DIR"
echo "=========================================="

exec ./submit_image_folder_full_slate.sh
