#!/bin/bash
# Monitor batch processing progress

RESULTS_DIR="/home/adeb/data/Collembola_results"
TOTAL_IMAGES=91

echo "Collembola Batch Processing Monitor"
echo "===================================="
echo ""

# Check if process is running
if ps aux | grep -q "[p]rocess_batch.py"; then
    echo "✓ Batch process is RUNNING"
    PID=$(ps aux | grep "[p]rocess_batch.py" | awk '{print $2}')
    echo "  PID: $PID"
else
    echo "✗ Batch process is NOT running"
fi

echo ""

# Count completed folders
COMPLETED=$(ls -d $RESULTS_DIR/*/ 2>/dev/null | wc -l)
echo "Progress: $COMPLETED / $TOTAL_IMAGES images processed"

# Calculate percentage
if [ $TOTAL_IMAGES -gt 0 ]; then
    PERCENT=$(( 100 * COMPLETED / TOTAL_IMAGES ))
    echo "  $PERCENT% complete"
fi

echo ""

# Show recent folders
echo "Recently processed:"
ls -dt $RESULTS_DIR/*/ 2>/dev/null | head -5 | while read dir; do
    basename "$dir"
done

echo ""
echo "Latest measurements CSV files:"
find $RESULTS_DIR -name "*_measurements.csv" -type f -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -3 | while read timestamp file; do
    basename "$file"
done

echo ""
echo "To watch live progress:"
echo "  watch -n 5 bash scripts/monitor_batch.sh"
