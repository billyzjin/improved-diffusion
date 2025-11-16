#!/bin/bash

# This script downloads all the .npz sample files from your cluster to your local machine.

# --- Configuration ---
# Your username and the server address of the cluster's login node.
# Please fill this in. For example: "bjin0@pythia.alcf.anl.gov"
REMOTE_HOST="bjin0@pythia.chicagobooth.edu"

# The full path to the parent evaluation directory on the cluster.
REMOTE_EVAL_DIR="/project_gpfs/bjin0/evaluation_parallel_20251102_124357"

# The local directory where you want to save the samples.
# This script will create it if it doesn't exist.
LOCAL_SAMPLES_DIR="./downloaded_samples"
# --- End of Configuration ---

# Check if the user has configured the remote host
if [ "$REMOTE_HOST" == "USER@CLUSTER_ADDRESS" ]; then
    echo "ERROR: Please edit this script and set the REMOTE_HOST variable."
    exit 1
fi

# Define the path for the SSH control socket
# This creates a temporary, reusable connection
CONTROL_PATH="/tmp/ssh_mux_$(date +%s)"

# Setup a trap to automatically close the SSH connection when the script exits
# This is a cleanup mechanism to ensure we don't leave connections open.
trap 'echo "Closing SSH master connection..."; ssh -S "${CONTROL_PATH}" -O exit "${REMOTE_HOST}" 2>/dev/null' EXIT

echo "Preparing to download samples to: $LOCAL_SAMPLES_DIR"
mkdir -p "$LOCAL_SAMPLES_DIR"

echo "Establishing a persistent SSH connection... (You will be prompted for your password now)"
# Create the master connection in the background. You only authenticate once.
ssh -M -S "${CONTROL_PATH}" -fnN "${REMOTE_HOST}"

echo "Connection established. Getting list of experiment directories from the cluster..."
# Use the master connection to get the list of remote directories
REMOTE_DIRS=$(ssh -S "${CONTROL_PATH}" "${REMOTE_HOST}" "find ${REMOTE_EVAL_DIR} -mindepth 1 -maxdepth 1 -type d")

if [ -z "$REMOTE_DIRS" ]; then
    echo "ERROR: Could not find any experiment sub-directories in ${REMOTE_EVAL_DIR}"
    exit 1
fi

echo "Found directories. Starting download..."

# Loop through each directory and use rsync for a progress bar
for dir_path in $REMOTE_DIRS; do
    exp_name=$(basename "$dir_path")
    remote_file_path="${dir_path}/samples_50000x32x32x3.npz"
    local_file_path="${LOCAL_SAMPLES_DIR}/${exp_name}_samples.npz"
    
    echo "--> Downloading samples for ${exp_name}..."
    # Use rsync with the --progress flag and tell it to use our persistent SSH connection
    rsync --progress -e "ssh -S ${CONTROL_PATH}" "${REMOTE_HOST}:${remote_file_path}" "$local_file_path"
done

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE!"
echo "All sample files have been saved and renamed in the '$LOCAL_SAMPLES_DIR' directory."
echo "=========================================="