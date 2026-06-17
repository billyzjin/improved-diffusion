#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=eval_image_folder

set -euo pipefail

required_vars=(
    DATASET
    EVAL_NAME
    RUN_NAME
    SCHEDULE_NAME
    OBJECTIVE
    MODEL_PATH
    PARENT_EVAL_DIR
    TRAIN_DIR
    TEST_DIR
    STATS_FILE
    IMAGE_SIZE
)
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: required environment variable $var is not set"
        exit 1
    fi
done

echo "=========================================="
echo "IMAGE-FOLDER NLL+FID EVALUATION"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Dataset: $DATASET"
echo "Run: $RUN_NAME"
echo "Eval name: $EVAL_NAME"
echo "Schedule: $SCHEDULE_NAME"
echo "Objective: $OBJECTIVE"
echo "Model: $MODEL_PATH"
echo "Saving results to: $PARENT_EVAL_DIR"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "=========================================="

module load python/booth/3.12 || { echo "ERROR: Failed to load python/booth/3.12"; exit 1; }

if module load cuda/12.8 2>/dev/null; then
    echo "Loaded cuda/12.8"
elif module load cuda/12.4 2>/dev/null; then
    echo "Loaded cuda/12.4"
elif module load cuda/12.2 2>/dev/null; then
    echo "Loaded cuda/12.2"
elif module load cuda/11.8 2>/dev/null; then
    echo "Loaded cuda/11.8"
else
    echo "ERROR: No CUDA module found"
    exit 1
fi

cd /home/bjin0/improved-diffusion
export PYTHONPATH="/home/bjin0/improved-diffusion:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

python3 - <<'PY'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this job")
PY

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
    exit 1
fi
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: training/reference data dir missing: $TRAIN_DIR"
    exit 1
fi
if [ ! -d "$TEST_DIR" ]; then
    echo "ERROR: test data dir missing: $TEST_DIR"
    exit 1
fi

case "$SCHEDULE_NAME" in
    linear|cosine|geometric_linear|geometric_cosine) ;;
    *)
        echo "ERROR: unknown SCHEDULE_NAME=$SCHEDULE_NAME"
        exit 1
        ;;
esac

NLL_NUM_SAMPLES=${NLL_NUM_SAMPLES:-10000}
NUM_SAMPLES=${NUM_SAMPLES:-50000}
NLL_BATCH_SIZE=${NLL_BATCH_SIZE:-128}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-256}
FORCE=${FORCE:-0}
SKIP_NLL=${SKIP_NLL:-0}
SKIP_FID=${SKIP_FID:-0}
NUM_CHANNELS=${NUM_CHANNELS:-128}
NUM_RES_BLOCKS=${NUM_RES_BLOCKS:-3}
NUM_HEADS=${NUM_HEADS:-4}
ATTENTION_RESOLUTIONS=${ATTENTION_RESOLUTIONS:-16,8}

if [ -n "${DROPOUT:-}" ]; then
    dropout="$DROPOUT"
elif [ "$SCHEDULE_NAME" = "linear" ]; then
    dropout=0.1
else
    dropout=0.3
fi

learn_sigma="False"
use_kl="False"
rescale_learned_sigmas="True"
case "$OBJECTIVE" in
    simple)
        ;;
    hybrid)
        learn_sigma="True"
        use_kl="False"
        rescale_learned_sigmas="True"
        ;;
    vlb)
        learn_sigma="True"
        use_kl="True"
        rescale_learned_sigmas="True"
        ;;
    *)
        echo "ERROR: unknown OBJECTIVE=$OBJECTIVE"
        exit 1
        ;;
esac

EXP_DIR="$PARENT_EVAL_DIR/$EVAL_NAME"
mkdir -p "$EXP_DIR"

export OPENAI_LOGDIR="$EXP_DIR/logger"
export OPENAI_LOG_FORMAT="stdout,log"
mkdir -p "$OPENAI_LOGDIR"

SAMPLES_FILE="$EXP_DIR/samples_${NUM_SAMPLES}x${IMAGE_SIZE}x${IMAGE_SIZE}x3.npz"

{
    echo -e "dataset\t$DATASET"
    echo -e "run_name\t$RUN_NAME"
    echo -e "eval_name\t$EVAL_NAME"
    echo -e "schedule_name\t$SCHEDULE_NAME"
    echo -e "objective\t$OBJECTIVE"
    echo -e "dropout\t$dropout"
    echo -e "learn_sigma\t$learn_sigma"
    echo -e "use_kl\t$use_kl"
    echo -e "rescale_learned_sigmas\t$rescale_learned_sigmas"
    echo -e "model_path\t$MODEL_PATH"
    echo -e "train_dir\t$TRAIN_DIR"
    echo -e "test_dir\t$TEST_DIR"
    echo -e "stats_file\t$STATS_FILE"
    echo -e "image_size\t$IMAGE_SIZE"
    echo -e "nll_num_samples\t$NLL_NUM_SAMPLES"
    echo -e "fid_num_samples\t$NUM_SAMPLES"
    echo -e "sample_file\t$SAMPLES_FILE"
    echo -e "start_time\t$(date -Is)"
} > "$EXP_DIR/metadata.tsv"

MODEL_FLAGS=(
    --image_size "$IMAGE_SIZE"
    --num_channels "$NUM_CHANNELS"
    --num_res_blocks "$NUM_RES_BLOCKS"
    --num_heads "$NUM_HEADS"
    --attention_resolutions "$ATTENTION_RESOLUTIONS"
    --use_scale_shift_norm True
    --dropout "$dropout"
    --learn_sigma "$learn_sigma"
    --use_kl "$use_kl"
    --rescale_learned_sigmas "$rescale_learned_sigmas"
    --diffusion_steps 4000
    --noise_schedule "$SCHEDULE_NAME"
    --class_cond False
    --use_checkpoint False
    --rescale_timesteps True
    --predict_xstart False
    --clip_denoised True
)

ensure_stats_file() {
    if [ -f "$STATS_FILE" ]; then
        echo "Using existing train stats: $STATS_FILE"
        return 0
    fi

    local lock_dir="${STATS_FILE}.lock"
    local tmp_stats="${STATS_FILE}.tmp.${SLURM_JOB_ID:-$$}.npz"

    if mkdir "$lock_dir" 2>/dev/null; then
        echo "Precomputing train stats to: $STATS_FILE"
        trap 'rm -rf "$lock_dir" "$tmp_stats"' EXIT
        python3 -m pytorch_fid "$TRAIN_DIR" "$tmp_stats" --save-stats --device cuda \
            2>&1 | tee "$EXP_DIR/fid_stats_build.txt"
        mv "$tmp_stats" "$STATS_FILE"
        rmdir "$lock_dir"
        trap - EXIT
        return 0
    fi

    echo "Another job is building train stats; waiting for $STATS_FILE"
    for _ in $(seq 1 240); do
        if [ -f "$STATS_FILE" ]; then
            echo "Train stats are ready: $STATS_FILE"
            return 0
        fi
        sleep 30
    done

    echo "ERROR: timed out waiting for train stats: $STATS_FILE"
    exit 1
}

if [ "$SKIP_NLL" = "1" ]; then
    echo "Skipping NLL (SKIP_NLL=1)"
elif [ -s "$EXP_DIR/nll_results.txt" ] && [ "$FORCE" != "1" ]; then
    echo "Skipping NLL; existing result found: $EXP_DIR/nll_results.txt"
else
    echo "Calculating NLL..."
    python3 scripts/image_nll_no_mpi.py \
        --model_path "$MODEL_PATH" \
        --data_dir "$TEST_DIR" \
        --batch_size "$NLL_BATCH_SIZE" \
        --num_samples "$NLL_NUM_SAMPLES" \
        "${MODEL_FLAGS[@]}" \
        2>&1 | tee "$EXP_DIR/nll_results.txt"
fi

if [ "$SKIP_FID" = "1" ]; then
    echo "Skipping FID and sampling (SKIP_FID=1)"
else
    RESPACING_FLAG=()
    if [ -n "${EVAL_TIMESTEP_RESPACING:-}" ]; then
        RESPACING_FLAG=(--timestep_respacing "$EVAL_TIMESTEP_RESPACING")
        echo "Using timestep respacing for sampling: $EVAL_TIMESTEP_RESPACING"
    else
        echo "Using full sampling (no timestep respacing)."
    fi

    if [ -s "$SAMPLES_FILE" ] && [ "$FORCE" != "1" ]; then
        echo "Skipping sampling; existing samples found: $SAMPLES_FILE"
    else
        echo "Generating samples for FID..."
        rm -f "$OPENAI_LOGDIR"/samples_*.npz "$SAMPLES_FILE"
        python3 scripts/image_sample_no_mpi.py \
            --model_path "$MODEL_PATH" \
            --num_samples "$NUM_SAMPLES" \
            --batch_size "$SAMPLE_BATCH_SIZE" \
            "${MODEL_FLAGS[@]}" \
            "${RESPACING_FLAG[@]}" \
            --use_ddim False \
            2>&1 | tee "$EXP_DIR/sample_results.txt"

        sample_file_source=$(ls -1 "$OPENAI_LOGDIR"/samples_*.npz 2>/dev/null | head -1)
        if [ -n "$sample_file_source" ] && [ -f "$sample_file_source" ]; then
            cp "$sample_file_source" "$SAMPLES_FILE"
            echo "Samples saved to: $SAMPLES_FILE"
        else
            echo "ERROR: sample file not found in $OPENAI_LOGDIR"
            ls -la "$OPENAI_LOGDIR" 2>/dev/null || true
            exit 1
        fi
    fi

    python3 - <<'PY'
import importlib.util
if importlib.util.find_spec("pytorch_fid") is None:
    raise SystemExit("pytorch_fid is not installed in this environment")
PY

    ensure_stats_file

    if [ -s "$EXP_DIR/fid_results.txt" ] && [ "$FORCE" != "1" ]; then
        echo "Skipping FID; existing result found: $EXP_DIR/fid_results.txt"
    else
        echo "Computing FID..."
        python3 scripts/compute_fid_from_npz.py \
            --samples_npz "$SAMPLES_FILE" \
            --stats_npz "$STATS_FILE" \
            --device cuda \
            --tmp_root "/project_gpfs/bata0/bjin0" \
            --output_txt "$EXP_DIR/fid_results.txt" \
            2>&1 | tee "$EXP_DIR/fid_compute_log.txt"
    fi
fi

{
    echo -e "end_time\t$(date -Is)"
    echo -e "exit_code\t0"
} >> "$EXP_DIR/metadata.tsv"

echo "=========================================="
echo "IMAGE-FOLDER EVALUATION COMPLETE"
echo "Results saved in: $EXP_DIR"
echo "=========================================="
