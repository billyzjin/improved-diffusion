#!/bin/bash

#SBATCH --account=bata0-external
#SBATCH --partition=long_hopper
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=imagenet64_download
#SBATCH --output=imagenet64_download_%j.out
#SBATCH --error=imagenet64_download_%j.err

set -euo pipefail

echo "=========================================="
echo "ImageNet-64 official NPZ download"
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST = ${SLURM_NODELIST:-}"
echo "Start          = $(date)"
echo "=========================================="

module load python/booth/3.12

DOWNLOAD_ROOT=${DOWNLOAD_ROOT:-/project_gpfs/bata0/bjin0/imagenet64_downloads}
ARCHIVE_DIR=${ARCHIVE_DIR:-${DOWNLOAD_ROOT}/archives}
UNZIPPED_DIR=${UNZIPPED_DIR:-${DOWNLOAD_ROOT}/unzipped}

PART1_URL=${PART1_URL:-https://www.image-net.org/data/downsample/Imagenet64_train_part1_npz.zip}
PART2_URL=${PART2_URL:-https://www.image-net.org/data/downsample/Imagenet64_train_part2_npz.zip}
VAL_URL=${VAL_URL:-https://www.image-net.org/data/downsample/Imagenet64_val_npz.zip}

# Optional browser cookie export if ImageNet redirects direct downloads to login.
COOKIE_FILE=${COOKIE_FILE:-}

mkdir -p "$ARCHIVE_DIR" "$UNZIPPED_DIR"

echo "DOWNLOAD_ROOT = $DOWNLOAD_ROOT"
echo "ARCHIVE_DIR   = $ARCHIVE_DIR"
echo "UNZIPPED_DIR  = $UNZIPPED_DIR"
echo "PART1_URL     = $PART1_URL"
echo "PART2_URL     = $PART2_URL"
echo "VAL_URL       = $VAL_URL"
if [[ -n "$COOKIE_FILE" ]]; then
  echo "COOKIE_FILE   = $COOKIE_FILE"
fi

download_one() {
  local url="$1"
  local basename
  basename=$(basename "${url%%\?*}")
  local out="$ARCHIVE_DIR/$basename"

  local wget_args=(
    --continue
    --tries=10
    --timeout=60
    --waitretry=30
    --output-document "$out"
  )
  if [[ -n "$COOKIE_FILE" ]]; then
    wget_args+=(--load-cookies "$COOKIE_FILE" --keep-session-cookies)
  fi

  echo "Downloading $url"
  wget "${wget_args[@]}" "$url"

  echo "Downloaded: $out"
  ls -lh "$out"

  if ! python3 - "$out" <<'PY'
import sys
import zipfile
from pathlib import Path

path = Path(sys.argv[1])
if not zipfile.is_zipfile(path):
    data = path.read_bytes()[:256].lower()
    if b"<html" in data or b"<!doctype html" in data:
        raise SystemExit(f"{path} looks like an HTML login/error page, not a zip archive")
    raise SystemExit(f"{path} is not a valid zip archive")
print(f"zip OK: {path}")
PY
  then
    echo "ERROR: Download did not produce a valid zip archive: $out"
    exit 1
  fi
}

extract_one() {
  local archive="$1"
  echo "Extracting $archive -> $UNZIPPED_DIR"
  python3 - "$archive" "$UNZIPPED_DIR" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out_dir)
print(f"extracted {archive.name}")
PY
}

download_one "$PART1_URL"
download_one "$PART2_URL"
download_one "$VAL_URL"

extract_one "$ARCHIVE_DIR/$(basename "${PART1_URL%%\?*}")"
extract_one "$ARCHIVE_DIR/$(basename "${PART2_URL%%\?*}")"
extract_one "$ARCHIVE_DIR/$(basename "${VAL_URL%%\?*}")"

echo "Checking extracted NPZ files..."
find "$UNZIPPED_DIR" -maxdepth 3 -type f \( -name 'train_data_batch_*.npz' -o -name 'val_data.npz' \) -printf '%p %s bytes\n' | sort

python3 - "$UNZIPPED_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
train = sorted(root.rglob("train_data_batch_*.npz"))
val = sorted(root.rglob("val_data.npz"))
expected = {f"train_data_batch_{i}.npz" for i in range(1, 11)}
found = {p.name for p in train}
missing = sorted(expected - found)
extra = sorted(found - expected)
print(f"train shards found: {len(train)}")
print(f"val shards found:   {len(val)}")
if missing:
    print("missing train shards:", ", ".join(missing))
if extra:
    print("unexpected train shard names:", ", ".join(extra))
if len(val) != 1:
    print("expected exactly one val_data.npz")
if missing or extra or len(val) != 1:
    raise SystemExit(1)
PY

echo "=========================================="
echo "Download/extract complete"
echo "End = $(date)"
echo "Verifier source root: $UNZIPPED_DIR"
echo "Next: ./submit_verify_imagenet64_official.sh"
echo "=========================================="
