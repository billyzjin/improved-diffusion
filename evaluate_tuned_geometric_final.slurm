#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=eval_tuned_geometric

set -euo pipefail

required_vars=(
    DATASET
    RUN_NAME
    EVAL_NAME
    SCHEDULE_NAME
    OBJECTIVE
    GEOMETRIC_BETA1
    GEOMETRIC_ALPHA_BAR_T
    MODEL_PATH
    PARENT_EVAL_DIR
)
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: required environment variable $var is not set"
        exit 1
    fi
done

echo "=========================================="
echo "EVALUATING TUNED GEOMETRIC MODEL"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Dataset: $DATASET"
echo "Run: $RUN_NAME"
echo "Eval name: $EVAL_NAME"
echo "Objective: $OBJECTIVE"
echo "beta_1: $GEOMETRIC_BETA1"
echo "alpha_bar_T: $GEOMETRIC_ALPHA_BAR_T"
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

python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "ERROR: PyTorch test failed"
    exit 1
}

cd /home/bjin0/improved-diffusion

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
    exit 1
fi

case "$DATASET" in
    mnist)
        DATA_TRAIN_DIR="./mnist_train"
        DATA_TEST_DIR="./mnist_test"
        STATS_FILE=${STATS_FILE:-/project_gpfs/bata0/bjin0/mnist_train_stats.npz}
        IMAGE_SIZE=32
        DEFAULT_NUM_SAMPLES=50000
        DEFAULT_NLL_NUM_SAMPLES=10000
        DEFAULT_SAMPLE_BATCH_SIZE=256
        DEFAULT_NLL_BATCH_SIZE=128
        ;;
    fashionmnist)
        DATA_TRAIN_DIR="./fashion_train"
        DATA_TEST_DIR="./fashion_test"
        STATS_FILE=${STATS_FILE:-/project_gpfs/bata0/bjin0/fashionmnist_train_stats.npz}
        IMAGE_SIZE=32
        DEFAULT_NUM_SAMPLES=50000
        DEFAULT_NLL_NUM_SAMPLES=10000
        DEFAULT_SAMPLE_BATCH_SIZE=256
        DEFAULT_NLL_BATCH_SIZE=128
        ;;
    cifar10)
        DATA_TRAIN_DIR="./cifar_train"
        DATA_TEST_DIR="./cifar_test"
        STATS_FILE=${STATS_FILE:-/project_gpfs/bata0/bjin0/cifar10_train_stats.npz}
        IMAGE_SIZE=32
        DEFAULT_NUM_SAMPLES=50000
        DEFAULT_NLL_NUM_SAMPLES=10000
        DEFAULT_SAMPLE_BATCH_SIZE=256
        DEFAULT_NLL_BATCH_SIZE=128
        ;;
    imagenet64)
        IMAGENET_DATA_ROOT=${IMAGENET_DATA_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505}
        DATA_TRAIN_DIR=${IMAGENET_TRAIN_DIR:-${IMAGENET_DATA_ROOT}/train}
        DATA_TEST_DIR=${IMAGENET_VAL_DIR:-${IMAGENET_DATA_ROOT}/val}
        STATS_FILE=${STATS_FILE:-/project_gpfs/bata0/bjin0/imagenet64_official_verified_20260505_train_stats.npz}
        IMAGE_SIZE=64
        DEFAULT_NUM_SAMPLES=10000
        DEFAULT_NLL_NUM_SAMPLES=10000
        DEFAULT_SAMPLE_BATCH_SIZE=64
        DEFAULT_NLL_BATCH_SIZE=64
        ;;
    *)
        echo "ERROR: unknown DATASET=$DATASET"
        exit 1
        ;;
esac

if [ ! -d "$DATA_TRAIN_DIR" ]; then
    echo "ERROR: training/reference data dir missing: $DATA_TRAIN_DIR"
    exit 1
fi
if [ ! -d "$DATA_TEST_DIR" ]; then
    echo "ERROR: test/validation data dir missing: $DATA_TEST_DIR"
    exit 1
fi

NLL_NUM_SAMPLES=${NLL_NUM_SAMPLES:-$DEFAULT_NLL_NUM_SAMPLES}
NUM_SAMPLES=${NUM_SAMPLES:-$DEFAULT_NUM_SAMPLES}
NLL_BATCH_SIZE=${NLL_BATCH_SIZE:-$DEFAULT_NLL_BATCH_SIZE}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-$DEFAULT_SAMPLE_BATCH_SIZE}
FORCE=${FORCE:-0}
SKIP_NLL=${SKIP_NLL:-0}
SKIP_FID=${SKIP_FID:-0}

learn_sigma="False"
use_kl="False"
rescale_learned_sigmas="False"
case "$OBJECTIVE" in
    simple)
        ;;
    hybrid)
        learn_sigma="True"
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

{
    echo -e "dataset\t$DATASET"
    echo -e "run_name\t$RUN_NAME"
    echo -e "eval_name\t$EVAL_NAME"
    echo -e "schedule_name\t$SCHEDULE_NAME"
    echo -e "objective\t$OBJECTIVE"
    echo -e "geometric_beta1\t$GEOMETRIC_BETA1"
    echo -e "geometric_alpha_bar_T\t$GEOMETRIC_ALPHA_BAR_T"
    echo -e "model_path\t$MODEL_PATH"
    echo -e "data_train_dir\t$DATA_TRAIN_DIR"
    echo -e "data_test_dir\t$DATA_TEST_DIR"
    echo -e "stats_file\t$STATS_FILE"
    echo -e "nll_num_samples\t$NLL_NUM_SAMPLES"
    echo -e "fid_num_samples\t$NUM_SAMPLES"
    echo -e "start_time\t$(date -Is)"
} > "$EXP_DIR/metadata.tsv"

MODEL_FLAGS=(
    --image_size "$IMAGE_SIZE"
    --num_channels 128
    --num_res_blocks 3
    --num_heads 4
    --attention_resolutions 16,8
    --use_scale_shift_norm True
    --dropout 0.3
    --learn_sigma "$learn_sigma"
    --use_kl "$use_kl"
    --rescale_learned_sigmas "$rescale_learned_sigmas"
    --diffusion_steps 4000
    --noise_schedule geometric
    --geometric_beta1 "$GEOMETRIC_BETA1"
    --geometric_alpha_bar_T "$GEOMETRIC_ALPHA_BAR_T"
    --class_cond False
    --use_checkpoint False
    --rescale_timesteps True
    --predict_xstart False
    --clip_denoised True
)

if [ "$SKIP_NLL" = "1" ]; then
    echo "Skipping NLL (SKIP_NLL=1)"
elif [ -s "$EXP_DIR/nll_results.txt" ] && [ "$FORCE" != "1" ]; then
    echo "Skipping NLL; existing result found: $EXP_DIR/nll_results.txt"
else
    echo "Calculating NLL..."
    python3 scripts/image_nll_no_mpi.py \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_TEST_DIR" \
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

    if [ -s "$EXP_DIR/samples.npz" ] && [ "$FORCE" != "1" ]; then
        echo "Skipping sampling; existing samples found: $EXP_DIR/samples.npz"
    else
        echo "Generating samples for FID..."
        rm -f "$OPENAI_LOGDIR"/samples_*.npz "$EXP_DIR/samples.npz"
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
            cp "$sample_file_source" "$EXP_DIR/samples.npz"
            echo "Samples saved to: $EXP_DIR/samples.npz"
        else
            echo "ERROR: Sample file not found in $OPENAI_LOGDIR"
            ls -la "$OPENAI_LOGDIR" 2>/dev/null || true
            exit 1
        fi
    fi

    if ! python3 -c "import importlib.util; spec=importlib.util.find_spec('pytorch_fid'); print('pytorch_fid found:', spec is not None)" >/dev/null 2>&1; then
        echo "ERROR: pytorch_fid is not installed in this environment."
        exit 1
    fi

    if [ ! -f "$STATS_FILE" ]; then
        echo "Precomputing train stats to: $STATS_FILE"
        if [ "$DATASET" = "imagenet64" ]; then
            python3 scripts/compute_fid_stats_from_dir_recursive.py \
                --real_dir "$DATA_TRAIN_DIR" \
                --stats_npz "$STATS_FILE" \
                --device cuda \
                --batch_size 64 \
                --num_workers 4 \
                2>&1 | tee "$EXP_DIR/fid_stats_build.txt"
        else
            python3 -m pytorch_fid "$DATA_TRAIN_DIR" "$STATS_FILE" --save-stats --device cuda \
                2>&1 | tee "$EXP_DIR/fid_stats_build.txt"
        fi
    fi

    if [ -s "$EXP_DIR/fid_results.txt" ] && [ "$FORCE" != "1" ]; then
        echo "Skipping FID; existing result found: $EXP_DIR/fid_results.txt"
    else
        echo "Computing FID..."
        python3 scripts/compute_fid_from_npz.py \
            --samples_npz "$EXP_DIR/samples.npz" \
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
echo "TUNED GEOMETRIC EVALUATION COMPLETE"
echo "Results saved in: $EXP_DIR"
echo "=========================================="
