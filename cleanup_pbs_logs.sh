#!/bin/bash

echo "=========================================="
echo "CLEANING UP PBS LOG FILES"
echo "=========================================="

# Count files before deletion
O_FILES=0
E_FILES=0
for file in *.o*; do
    if [ -f "$file" ]; then
        O_FILES=$((O_FILES + 1))
    fi
done 2>/dev/null || true

for file in *.e*; do
    if [ -f "$file" ]; then
        E_FILES=$((E_FILES + 1))
    fi
done 2>/dev/null || true

echo "Found $O_FILES .o* files and $E_FILES .e* files"

if [ $O_FILES -eq 0 ] && [ $E_FILES -eq 0 ]; then
    echo "No PBS log files found to clean up."
    exit 0
fi

# Show what will be deleted
echo ""
echo "Files to be deleted:"
for file in *.o*; do
    if [ -f "$file" ]; then
        ls -la "$file"
    fi
done 2>/dev/null || true

for file in *.e*; do
    if [ -f "$file" ]; then
        ls -la "$file"
    fi
done 2>/dev/null || true

echo ""
read -p "Are you sure you want to delete these files? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Delete the files
    rm -f *.o* 2>/dev/null
    rm -f *.e* 2>/dev/null
    
    # Verify deletion
    REMAINING_O=0
    REMAINING_E=0
    for file in *.o*; do
        if [ -f "$file" ]; then
            REMAINING_O=$((REMAINING_O + 1))
        fi
    done 2>/dev/null || true
    
    for file in *.e*; do
        if [ -f "$file" ]; then
            REMAINING_E=$((REMAINING_E + 1))
        fi
    done 2>/dev/null || true
    
    if [ $REMAINING_O -eq 0 ] && [ $REMAINING_E -eq 0 ]; then
        echo "✅ Successfully deleted all PBS log files!"
    else
        echo "⚠️  Some files may not have been deleted. Remaining: $REMAINING_O .o* files, $REMAINING_E .e* files"
    fi
else
    echo "❌ Cleanup cancelled."
fi

echo "=========================================="
