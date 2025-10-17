#!/bin/bash

echo "=========================================="
echo "FIND AVAILABLE CHECKPOINTS"
echo "=========================================="

echo "Searching for model checkpoints in scratch space..."
echo ""

# Find all model checkpoints
find /scratch/bjin0 -name "model*.pt" -type f 2>/dev/null | sort | while read checkpoint; do
    # Extract experiment name from path
    experiment=$(echo "$checkpoint" | grep -o "cifar10_[^/]*" | head -1)
    
    # Get file size and modification time
    size=$(ls -lh "$checkpoint" | awk '{print $5}')
    mtime=$(ls -l "$checkpoint" | awk '{print $6, $7, $8}')
    
    # Extract step number from filename
    step=$(echo "$checkpoint" | grep -o "model[0-9]*" | grep -o "[0-9]*")
    
    echo "Experiment: $experiment"
    echo "  Checkpoint: $checkpoint"
    echo "  Step: $step"
    echo "  Size: $size"
    echo "  Modified: $mtime"
    echo ""
done

echo "=========================================="
echo "USAGE:"
echo "=========================================="
echo "To resume training:"
echo "  ./resume_training.sh <checkpoint_path> [experiment_name]"
echo ""
echo "Example:"
echo "  ./resume_training.sh /scratch/bjin0/job123/logs/cifar10_ours_simple/model100000.pt ours_simple"
