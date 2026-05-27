#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PROBE_SLURM=${PROBE_SLURM:-"$SCRIPT_DIR/train_probe_geometric.slurm"}
JOB_PREFIX=${JOB_PREFIX:-probe_hyb}
DRY_RUN=${DRY_RUN:-0}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"$SCRIPT_DIR/slurm_logs"}
SEARCH_PRESET=${SEARCH_PRESET:-coarse_global}
SUBMISSION_TSV=${SUBMISSION_TSV:-}

# Search presets:
# - coarse_global: broad search spanning cosine-like through near-linear alpha_bar_T.
# - coarse_cosine_local: local refinement around the cosine-matched endpoint.
if [[ -z "${BETA_VALUES+x}" ]]; then
    case "$SEARCH_PRESET" in
        coarse_global) BETA_VALUES="1e-6 3e-6 1e-5 3e-5" ;;
        coarse_cosine_local) BETA_VALUES="3e-7 1e-6 3e-6 1e-5 3e-5" ;;
        *)
            echo "ERROR: Unknown SEARCH_PRESET=$SEARCH_PRESET" >&2
            exit 1
            ;;
    esac
fi
if [[ -z "${ALPHA_VALUES+x}" ]]; then
    case "$SEARCH_PRESET" in
        coarse_global) ALPHA_VALUES="1e-10 1e-8 1e-6 1e-4" ;;
        coarse_cosine_local) ALPHA_VALUES="1e-11 3e-11 1e-10 3e-10 1e-9" ;;
        *)
            echo "ERROR: Unknown SEARCH_PRESET=$SEARCH_PRESET" >&2
            exit 1
            ;;
    esac
fi

usage() {
    cat <<EOF
Usage:
  ./submit_probe_geometric_grid.sh

Environment overrides:
  PROBE_SLURM   Path to the probe SLURM script.
  JOB_PREFIX    Prefix for PROBE_NAME / --job-name. Default: ${JOB_PREFIX}
  DRY_RUN       Set to 1 to print sbatch commands without submitting.
  SLURM_LOG_DIR Directory for sbatch stdout/stderr files.
  SEARCH_PRESET Default grid preset. One of: coarse_global, coarse_cosine_local.
  BETA_VALUES   Space-separated beta_1 values.
  ALPHA_VALUES  Space-separated alpha_bar_T values.
  SUBMISSION_TSV Optional TSV manifest to record submitted job IDs and parameters.

Example:
  DRY_RUN=1 \\
  BETA_VALUES="1e-6 3e-6 1e-5" \\
  ALPHA_VALUES="3e-11 1e-10 3e-10" \\
  ./submit_probe_geometric_grid.sh
EOF
}

sanitize_tag() {
    local value="$1"
    value="${value//+/}"
    value="${value//./p}"
    value="${value//\//_}"
    value="${value//:/_}"
    value="${value//,/__}"
    printf '%s' "$value"
}

validate_pair() {
    local beta="$1"
    local alpha="$2"
    python3 - "$beta" "$alpha" <<'PY'
import sys

beta = float(sys.argv[1])
alpha = float(sys.argv[2])
if not (0.0 < beta < 1.0):
    raise SystemExit(f"invalid beta_1={beta}")
if not (0.0 < alpha < 1.0):
    raise SystemExit(f"invalid alpha_bar_T={alpha}")
if not (alpha < 1.0 - beta):
    raise SystemExit(
        f"invalid pair: need alpha_bar_T < 1 - beta_1, got alpha_bar_T={alpha}, 1-beta_1={1.0-beta}"
    )
PY
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$PROBE_SLURM" ]]; then
    echo "ERROR: Probe SLURM script not found: $PROBE_SLURM" >&2
    exit 1
fi

read -r -a beta_array <<< "$BETA_VALUES"
read -r -a alpha_array <<< "$ALPHA_VALUES"

if [[ ${#beta_array[@]} -eq 0 ]]; then
    echo "ERROR: BETA_VALUES is empty" >&2
    exit 1
fi
if [[ ${#alpha_array[@]} -eq 0 ]]; then
    echo "ERROR: ALPHA_VALUES is empty" >&2
    exit 1
fi

mkdir -p "$SLURM_LOG_DIR"
if [[ -n "$SUBMISSION_TSV" ]]; then
    mkdir -p "$(dirname "$SUBMISSION_TSV")"
    if [[ ! -f "$SUBMISSION_TSV" ]]; then
        printf 'submitted_at\tjob_id\tprobe_name\tbeta_1\talpha_bar_T\tprobe_slurm\tslurm_log_dir\n' > "$SUBMISSION_TSV"
    fi
fi

total_jobs=$(( ${#beta_array[@]} * ${#alpha_array[@]} ))
echo "Submitting geometric probe grid"
echo "  probe_slurm:  $PROBE_SLURM"
echo "  job_prefix:   $JOB_PREFIX"
echo "  dry_run:      $DRY_RUN"
echo "  search_preset:$SEARCH_PRESET"
echo "  beta_count:   ${#beta_array[@]}"
echo "  alpha_count:  ${#alpha_array[@]}"
echo "  total_jobs:   $total_jobs"
echo "  slurm_logs:   $SLURM_LOG_DIR"
if [[ -n "$SUBMISSION_TSV" ]]; then
    echo "  manifest:     $SUBMISSION_TSV"
fi

submitted=0
for beta in "${beta_array[@]}"; do
    for alpha in "${alpha_array[@]}"; do
        validate_pair "$beta" "$alpha"
        beta_tag=$(sanitize_tag "$beta")
        alpha_tag=$(sanitize_tag "$alpha")
        probe_name="${JOB_PREFIX}_b${beta_tag}_a${alpha_tag}"

        cmd=(
            env
            -u SBATCH_PARTITION
            -u SBATCH_ACCOUNT
            -u SBATCH_QOS
            -u SBATCH_GRES
            -u SBATCH_CONSTRAINT
            sbatch
            --account=bata0-external
            --partition=long_hopper
            --job-name="$probe_name"
            --output="$SLURM_LOG_DIR/${probe_name}_%j.out"
            --error="$SLURM_LOG_DIR/${probe_name}_%j.err"
            --export="ALL,GEOMETRIC_BETA1=${beta},GEOMETRIC_ALPHA_BAR_T=${alpha},PROBE_NAME=${probe_name}"
            "$PROBE_SLURM"
        )

        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'DRY RUN:'
            printf ' %q' "${cmd[@]}"
            printf '\n'
            job_id="DRY_RUN"
        else
            sbatch_output=$("${cmd[@]}")
            echo "$sbatch_output"
            job_id=$(awk '/Submitted batch job/ {print $4; exit}' <<< "$sbatch_output")
            if [[ -z "$job_id" ]]; then
                job_id="UNKNOWN"
            fi
        fi

        if [[ -n "$SUBMISSION_TSV" ]]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -Is)" \
                "$job_id" \
                "$probe_name" \
                "$beta" \
                "$alpha" \
                "$PROBE_SLURM" \
                "$SLURM_LOG_DIR" >> "$SUBMISSION_TSV"
        fi

        submitted=$((submitted + 1))
    done
done

echo "Processed $submitted jobs."
