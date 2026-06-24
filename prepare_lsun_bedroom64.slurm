#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=standard_l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=prepare_lsun_bedroom64

set -euo pipefail

module load python/booth/3.12
cd /home/bjin0/improved-diffusion
export PYTHONPATH="/home/bjin0/improved-diffusion:${PYTHONPATH:-}"

if [ -f /home/bjin0/improved-diffusion/.venv/bin/activate ]; then
    source /home/bjin0/improved-diffusion/.venv/bin/activate
fi

python3 - <<'PY'
import importlib.util

if importlib.util.find_spec("lmdb") is None:
    raise SystemExit(
        "python-lmdb is required. Install it in the repo-local .venv with: "
        "python -m pip install lmdb"
    )
PY

LSUN_ROOT=${LSUN_ROOT:-/project_gpfs/bata0/bjin0/lsun_bedroom_64x64}
LSUN_SOURCE_ROOT=${LSUN_SOURCE_ROOT:-${LSUN_ROOT}/source}
LSUN_CATEGORY=${LSUN_CATEGORY:-bedroom}
LSUN_BASE_URL=${LSUN_BASE_URL:-http://dl.yf.io/lsun/scenes}
LSUN_SPLITS=${LSUN_SPLITS:-train val}
LSUN_IMAGE_SIZE=${LSUN_IMAGE_SIZE:-64}
LSUN_OVERWRITE=${LSUN_OVERWRITE:-0}
LSUN_DOWNLOAD=${LSUN_DOWNLOAD:-1}
LSUN_CONVERT_TO_PNG=${LSUN_CONVERT_TO_PNG:-0}
LSUN_MAX_TRAIN_IMAGES=${LSUN_MAX_TRAIN_IMAGES:-0}
LSUN_MAX_VAL_IMAGES=${LSUN_MAX_VAL_IMAGES:-0}

mkdir -p "$LSUN_SOURCE_ROOT" "$LSUN_ROOT"

download_split() {
    local split=$1
    local name="${LSUN_CATEGORY}_${split}_lmdb"
    local zip_path="${LSUN_SOURCE_ROOT}/${name}.zip"
    local lmdb_dir="${LSUN_SOURCE_ROOT}/${name}"
    local url="${LSUN_BASE_URL}/${name}.zip"

    if [ -d "$lmdb_dir" ]; then
        echo "LMDB exists: $lmdb_dir"
        return 0
    fi

    if [ ! -s "$zip_path" ]; then
        if [ "$LSUN_DOWNLOAD" != "1" ]; then
            echo "ERROR: missing $zip_path and LSUN_DOWNLOAD is not 1"
            exit 1
        fi
        echo "Downloading $url -> $zip_path"
        curl -fL --retry 10 --retry-delay 30 -C - "$url" -o "$zip_path"
    else
        echo "Using existing zip: $zip_path"
    fi

    echo "Unzipping $zip_path"
    unzip -q -n "$zip_path" -d "$LSUN_SOURCE_ROOT"
    if [ ! -d "$lmdb_dir" ]; then
        echo "ERROR: unzip did not create expected LMDB directory: $lmdb_dir"
        exit 1
    fi
}

for split in $LSUN_SPLITS; do
    case "$split" in
        train|val) ;;
        *)
            echo "ERROR: unsupported LSUN split: $split"
            exit 1
            ;;
    esac
    download_split "$split"
done

export LSUN_ROOT LSUN_SOURCE_ROOT LSUN_CATEGORY LSUN_SPLITS
python3 - <<'PY'
import os
from pathlib import Path

import lmdb

root = Path(os.environ["LSUN_ROOT"])
source_root = Path(os.environ["LSUN_SOURCE_ROOT"])
category = os.environ["LSUN_CATEGORY"]
splits = os.environ["LSUN_SPLITS"].split()

manifest = root / "_source_manifest.tsv"
with manifest.open("w") as f:
    f.write("status\tcomplete\n")
    f.write(f"category\t{category}\n")
    f.write(f"source_root\t{source_root}\n")
    for split in splits:
        lmdb_path = source_root / f"{category}_{split}_lmdb"
        env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=32,
        )
        try:
            entries = int(env.stat()["entries"])
        finally:
            env.close()
        f.write(f"{split}_lmdb\t{lmdb_path}\n")
        f.write(f"{split}_entries\t{entries}\n")
print(f"Wrote LSUN source manifest: {manifest}")
PY

if [ "$LSUN_CONVERT_TO_PNG" != "1" ]; then
    echo "LSUN_CONVERT_TO_PNG=$LSUN_CONVERT_TO_PNG; leaving dataset in LMDB format for training/evaluation."
    exit 0
fi

overwrite_arg=()
if [ "$LSUN_OVERWRITE" = "1" ]; then
    overwrite_arg=(--overwrite)
fi

python3 datasets/lsun_bedroom.py \
    --root "$LSUN_SOURCE_ROOT" \
    --out_root "$LSUN_ROOT" \
    --category "$LSUN_CATEGORY" \
    --size "$LSUN_IMAGE_SIZE" \
    --splits $LSUN_SPLITS \
    --max_train_images "$LSUN_MAX_TRAIN_IMAGES" \
    --max_val_images "$LSUN_MAX_VAL_IMAGES" \
    "${overwrite_arg[@]}"
