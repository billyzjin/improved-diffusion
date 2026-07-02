#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=richer_metrics

set -euo pipefail

module load python/booth/3.12
if module load cuda/12.8 2>/dev/null; then
  echo "Loaded cuda/12.8"
elif module load cuda/12.4 2>/dev/null; then
  echo "Loaded cuda/12.4"
elif module load cuda/12.2 2>/dev/null; then
  echo "Loaded cuda/12.2"
elif module load cuda/11.8 2>/dev/null; then
  echo "Loaded cuda/11.8"
fi

cd /home/bjin0/improved-diffusion

if [ -f /home/bjin0/improved-diffusion/.venv/bin/activate ]; then
  source /home/bjin0/improved-diffusion/.venv/bin/activate
fi

MANIFEST=${MANIFEST:-results/richer_metrics/manifest.tsv}
OUTPUT_DIR=${OUTPUT_DIR:-results/richer_metrics}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-/project_gpfs/bata0/bjin0/richer_metrics_features}
METRICS=${METRICS:-cmmd,kid,density_coverage}
ROW_INDEX=${ROW_INDEX:-${SLURM_ARRAY_TASK_ID:-}}
DEVICE=${DEVICE:-cuda}
CLIP_MODEL=${CLIP_MODEL:-ViT-B-32}
CLIP_PRETRAINED=${CLIP_PRETRAINED:-/project_gpfs/bata0/bjin0/model_cache/openclip/open_clip_pytorch_model.bin}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-1234}
CMMD_N=${CMMD_N:-10000}
DC_N=${DC_N:-10000}
DC_K=${DC_K:-5}
KID_SUBSETS=${KID_SUBSETS:-100}
KID_SUBSET_SIZE=${KID_SUBSET_SIZE:-1000}
DISTANCE_CHUNK_SIZE=${DISTANCE_CHUNK_SIZE:-1024}

if [ -z "$ROW_INDEX" ]; then
  echo "ERROR: ROW_INDEX or SLURM_ARRAY_TASK_ID must be set."
  exit 1
fi

METRICS_COMMA=${METRICS//;/,}

python3 - <<'PY'
import importlib.util
missing = [m for m in ("torch", "torchvision", "pytorch_fid") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing required Python packages: {missing}")
PY

if [[ ",$METRICS_COMMA," == *",cmmd,"* || ",$METRICS_COMMA," == *",clip_density_coverage,"* ]]; then
  python3 - <<'PY'
import importlib.util
if importlib.util.find_spec("open_clip") is None and importlib.util.find_spec("clip") is None:
    raise SystemExit(
        "CLIP metrics requested, but neither open_clip_torch nor OpenAI clip is installed. "
        "Install/load one of them, or submit without CMMD/CLIP density-coverage metrics."
    )
PY
  if [ ! -f "$CLIP_PRETRAINED" ]; then
    echo "ERROR: CLIP_PRETRAINED does not exist: $CLIP_PRETRAINED"
    exit 1
  fi
fi

python3 scripts/run_richer_metrics.py \
  --manifest "$MANIFEST" \
  --row_index "$ROW_INDEX" \
  --metrics "$METRICS_COMMA" \
  --output_dir "$OUTPUT_DIR" \
  --feature_cache_root "$FEATURE_CACHE_ROOT" \
  --device "$DEVICE" \
  --clip_model "$CLIP_MODEL" \
  --clip_pretrained "$CLIP_PRETRAINED" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --cmmd_n "$CMMD_N" \
  --dc_n "$DC_N" \
  --dc_k "$DC_K" \
  --kid_subsets "$KID_SUBSETS" \
  --kid_subset_size "$KID_SUBSET_SIZE" \
  --distance_chunk_size "$DISTANCE_CHUNK_SIZE"
