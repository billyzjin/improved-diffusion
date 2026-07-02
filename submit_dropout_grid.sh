#!/bin/bash
set -euo pipefail

cd /home/bjin0/improved-diffusion

if [ -z "${DATASET:-${1:-}}" ]; then
    echo "ERROR: set DATASET or pass it as the first argument."
    echo "Supported datasets are those handled by submit_image_folder_full_slate.sh."
    exit 1
fi

export DATASET=${DATASET:-$1}
export SCHEDULES=${SCHEDULES:-geometric_linear,geometric_cosine}
export OBJECTIVES=${OBJECTIVES:-hybrid,vlb}
export DROPOUTS=${DROPOUTS:-0.1,0.2}
export SUBMISSION_DIR=${SUBMISSION_DIR:-/project_gpfs/bata0/bjin0/${DATASET}_dropout_grid_$(date +%Y%m%d_%H%M%S)}

echo "=========================================="
echo "Submitting dropout grid"
echo "Dataset: $DATASET"
echo "Schedules: $SCHEDULES"
echo "Objectives: $OBJECTIVES"
echo "Dropouts: $DROPOUTS"
echo "Submission dir: $SUBMISSION_DIR"
echo "=========================================="

exec ./submit_image_folder_full_slate.sh
