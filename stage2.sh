#!/bin/bash
# Stage 2 Validation Script for AutoCkt
# This script runs the sizing tool and validates the output circuit

# Usage: ./stage2.sh <model_checkpoint> <spec_file> [student_id]
# Example: ./stage2.sh ~/ray_results/.../checkpoint-100 spec1.json 314510144

set -e  # Exit on error

# Parse arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_checkpoint> <spec_file> [student_id]"
    echo ""
    echo "Arguments:"
    echo "  model_checkpoint : Path to trained agent checkpoint"
    echo "  spec_file        : Path to specification JSON file"
    echo "  student_id       : Your student ID (default: 314510144)"
    echo ""
    echo "Example:"
    echo "  $0 ~/ray_results/train_45nm_ngspice/PPO_TwoStageAmp_xxx/checkpoint_100/checkpoint-100 spec1.json"
    exit 1
fi

MODEL_CHECKPOINT=$1
SPEC_FILE=$2
STUDENT_ID=${3:-314510144}  # Default to 314510144 if not provided

SIZING_SCRIPT="sizing_${STUDENT_ID}.py"

echo "========================================"
echo "AutoCkt Stage 2 Validation"
echo "========================================"
echo "Model checkpoint: $MODEL_CHECKPOINT"
echo "Spec file: $SPEC_FILE"
echo "Student ID: $STUDENT_ID"
echo "Sizing script: $SIZING_SCRIPT"
echo "========================================"
echo ""

# Check if sizing script exists
if [ ! -f "$SIZING_SCRIPT" ]; then
    echo "ERROR: Sizing script not found: $SIZING_SCRIPT"
    echo "Please ensure sizing_${STUDENT_ID}.py exists in the current directory"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_CHECKPOINT"
    exit 1
fi

# Check if spec file exists
if [ ! -f "$SPEC_FILE" ]; then
    echo "ERROR: Spec file not found: $SPEC_FILE"
    exit 1
fi

# Initialize conda
echo "Initializing conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh || eval "$(conda shell.bash hook)"
source ~/.bashrc

# Activate autockt environment
echo "Activating autockt environment..."
conda activate autockt

# Set environment variables
echo "Setting environment variables..."
export PATH=$HOME/ngspice-27/opt/bin:$PATH
export PYTHONPATH=$PWD
export RAY_TMPDIR=$HOME/AutoCkt/ray_tmp

# Create RAY_TMPDIR if it doesn't exist
mkdir -p $RAY_TMPDIR

echo ""
echo "========================================"
echo "Step 1: Running Sizing Tool"
echo "========================================"
echo ""

# Run the sizing script
python $SIZING_SCRIPT --model "$MODEL_CHECKPOINT" --spec "$SPEC_FILE"

SIZING_EXIT_CODE=$?

if [ $SIZING_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Sizing script failed with exit code $SIZING_EXIT_CODE"
    exit $SIZING_EXIT_CODE
fi

# Check if final_design.cir was generated
if [ ! -f "final_design.cir" ]; then
    echo ""
    echo "ERROR: final_design.cir was not generated"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 2: Validating Output Circuit"
echo "========================================"
echo ""

# Check if the checker script exists
CHECKER_SCRIPT="/tmp/ngspice_model/ngspice_checker.py"

if [ ! -f "$CHECKER_SCRIPT" ]; then
    echo "WARNING: Checker script not found at $CHECKER_SCRIPT"
    echo "Skipping validation step."
    echo "Note: For official grading, validation will be performed at workstation 140.113.201.202"
    echo ""
    echo "To validate manually later, run:"
    echo "  python3 $CHECKER_SCRIPT --netlist final_design.cir --spec $SPEC_FILE"
else
    # Run the checker
    python3 "$CHECKER_SCRIPT" --netlist final_design.cir --spec "$SPEC_FILE"
    
    CHECKER_EXIT_CODE=$?
    
    echo ""
    if [ $CHECKER_EXIT_CODE -eq 0 ]; then
        echo "========================================"
        echo "✓ VALIDATION PASSED"
        echo "========================================"
        echo "The final circuit meets all specifications!"
    else
        echo "========================================"
        echo "✗ VALIDATION FAILED"
        echo "========================================"
        echo "The final circuit does not meet all specifications."
        echo "Exit code: $CHECKER_EXIT_CODE"
    fi
fi

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Sizing script: $SIZING_SCRIPT"
echo "Model: $MODEL_CHECKPOINT"
echo "Spec file: $SPEC_FILE"
echo "Output: final_design.cir"
echo ""
echo "Files generated:"
ls -lh final_design.cir 2>/dev/null || echo "  final_design.cir (not found)"
echo ""
echo "Stage 2 validation complete!"
echo "========================================"
