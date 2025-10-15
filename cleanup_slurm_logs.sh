#!/bin/bash

echo "=========================================="
echo "CLEANING UP SLURM LOG FILES"
echo "=========================================="

# Count files before deletion
OUT_FILES=0
ERR_FILES=0
for file in *.out; do
    if [ -f "$file" ]; then
        OUT_FILES=$((OUT_FILES + 1))
    fi
done 2>/dev/null || true

for file in *.err; do
    if [ -f "$file" ]; then
        ERR_FILES=$((ERR_FILES + 1))
    fi
done 2>/dev/null || true

echo "Found $OUT_FILES .out files and $ERR_FILES .err files"

if [ $OUT_FILES -eq 0 ] && [ $ERR_FILES -eq 0 ]; then
    echo "No SLURM log files found to clean up."
    exit 0
fi

# Show what will be deleted
echo ""
echo "Files to be deleted:"
for file in *.out; do
    if [ -f "$file" ]; then
        ls -la "$file"
    fi
done 2>/dev/null || true

for file in *.err; do
    if [ -f "$file" ]; then
        ls -la "$file"
    fi
done 2>/dev/null || true

echo ""
read -p "Are you sure you want to delete these files? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Delete the files
    rm -f *.out 2>/dev/null
    rm -f *.err 2>/dev/null
    
    # Verify deletion
    REMAINING_OUT=0
    REMAINING_ERR=0
    for file in *.out; do
        if [ -f "$file" ]; then
            REMAINING_OUT=$((REMAINING_OUT + 1))
        fi
    done 2>/dev/null || true
    
    for file in *.err; do
        if [ -f "$file" ]; then
            REMAINING_ERR=$((REMAINING_ERR + 1))
        fi
    done 2>/dev/null || true
    
    if [ $REMAINING_OUT -eq 0 ] && [ $REMAINING_ERR -eq 0 ]; then
        echo "✅ Successfully deleted all SLURM log files!"
    else
        echo "⚠️  Some files may not have been deleted. Remaining: $REMAINING_OUT .out files, $REMAINING_ERR .err files"
    fi
else
    echo "❌ Cleanup cancelled."
fi

echo "=========================================="
