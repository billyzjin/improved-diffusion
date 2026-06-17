#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=svhn_prepare

set -euo pipefail

echo "=========================================="
echo "SVHN data preparation"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node:   ${SLURM_NODELIST:-N/A}"
echo "Start:  $(date -Is)"
echo "=========================================="

module load python/booth/3.12

cd /home/bjin0/improved-diffusion

SVHN_ROOT=${SVHN_ROOT:-/project_gpfs/bata0/bjin0/svhn_32x32}
SVHN_SOURCE_ROOT=${SVHN_SOURCE_ROOT:-${SVHN_ROOT}/source}
SVHN_SPLITS=${SVHN_SPLITS:-train test}
SVHN_OVERWRITE=${SVHN_OVERWRITE:-0}

args=(
    --root "$SVHN_SOURCE_ROOT"
    --out_root "$SVHN_ROOT"
    --splits $SVHN_SPLITS
)

if [ "$SVHN_OVERWRITE" = "1" ]; then
    args+=(--overwrite)
fi

echo "SVHN_ROOT:        $SVHN_ROOT"
echo "SVHN_SOURCE_ROOT: $SVHN_SOURCE_ROOT"
echo "SVHN_SPLITS:      $SVHN_SPLITS"
echo "SVHN_OVERWRITE:   $SVHN_OVERWRITE"

python3 -c "import torchvision; print('torchvision OK')"
python3 datasets/svhn.py "${args[@]}"

echo "=========================================="
echo "SVHN preparation complete"
echo "Train dir: $SVHN_ROOT/train"
echo "Test dir:  $SVHN_ROOT/test"
echo "End:       $(date -Is)"
echo "=========================================="
