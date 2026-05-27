#!/bin/bash
set -euo pipefail

if [ "${1:-}" ]; then
    EVAL_DIR="$1"
else
    EVAL_DIR=$(ls -1d /project_gpfs/bata0/bjin0/tuned_geometric_evaluation_* 2>/dev/null | sort | tail -1)
fi

if [ -z "${EVAL_DIR:-}" ] || [ ! -d "$EVAL_DIR" ]; then
    echo "ERROR: evaluation directory not found"
    exit 1
fi

RESULTS_TSV="$EVAL_DIR/results_summary.tsv"
printf "dataset\tschedule_name\tobjective\tbeta_1\talpha_bar_T\teval_name\tnll_bpd\tfid\tmodel_path\n" > "$RESULTS_TSV"

for exp_dir in "$EVAL_DIR"/*; do
    if [ ! -d "$exp_dir" ] || [ ! -f "$exp_dir/metadata.tsv" ]; then
        continue
    fi

    dataset=$(awk -F'\t' '$1=="dataset"{print $2; exit}' "$exp_dir/metadata.tsv")
    schedule_name=$(awk -F'\t' '$1=="schedule_name"{print $2; exit}' "$exp_dir/metadata.tsv")
    objective=$(awk -F'\t' '$1=="objective"{print $2; exit}' "$exp_dir/metadata.tsv")
    beta=$(awk -F'\t' '$1=="geometric_beta1"{print $2; exit}' "$exp_dir/metadata.tsv")
    alpha=$(awk -F'\t' '$1=="geometric_alpha_bar_T"{print $2; exit}' "$exp_dir/metadata.tsv")
    eval_name=$(awk -F'\t' '$1=="eval_name"{print $2; exit}' "$exp_dir/metadata.tsv")
    model_path=$(awk -F'\t' '$1=="model_path"{print $2; exit}' "$exp_dir/metadata.tsv")

    nll="NA"
    if [ -f "$exp_dir/nll_results.txt" ]; then
        nll=$(grep "done .* samples: bpd=" "$exp_dir/nll_results.txt" | tail -1 | sed -E 's/.*bpd=([0-9.]+|nan).*/\1/' || true)
        if [ -z "$nll" ]; then
            nll="NA"
        fi
    fi

    fid="NA"
    if [ -f "$exp_dir/fid_results.txt" ]; then
        fid=$(tail -1 "$exp_dir/fid_results.txt" | tr -d '\r\n' | awk '{print $1}')
        if [ -z "$fid" ]; then
            fid="NA"
        fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset" "$schedule_name" "$objective" "$beta" "$alpha" "$eval_name" "$nll" "$fid" "$model_path" >> "$RESULTS_TSV"
done

echo "Wrote: $RESULTS_TSV"
