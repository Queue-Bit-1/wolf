#!/bin/bash
# Latin Square Name Bias Experiment
# 7 rotations x 50 games = 350 total games
# All players use gpt-oss:20b — only names and positions differ

set -e

LOG_DIR="/tmp/wolf_latin_square"
mkdir -p "$LOG_DIR"

echo "=== Latin Square Name Bias Experiment ==="
echo "7 rotations x 50 games = 350 total"
echo "Log dir: $LOG_DIR"
echo ""

for r in 1 2 3 4 5 6 7; do
    CONFIG="configs/benchmarks/name_bias_r${r}.yaml"
    LOG="$LOG_DIR/rotation_${r}.log"

    echo "──────────────────────────────────────"
    echo "Starting rotation $r / 7 (50 games)"
    echo "Config: $CONFIG"
    echo "Log: $LOG"
    date
    echo ""

    python3 -m wolf benchmark --config "$CONFIG" --games 50 2>&1 | tee "$LOG"

    COMPLETED=$(grep -c "completed" "$LOG" 2>/dev/null || echo 0)
    echo ""
    echo "Rotation $r done: $COMPLETED games completed"
    echo ""
done

echo "=== ALL 7 ROTATIONS COMPLETE ==="
date
echo "Logs in $LOG_DIR/"
