# AutoCkt Sizing Script - Usage Guide

## Overview
`sizing_314510144.py` is an automated circuit sizing tool that uses a trained RL agent to tune transistor parameters to meet target specifications for a two-stage operational amplifier.

## Prerequisites
- Conda environment `autockt` must be activated
- NGSpice must be installed and in PATH
- Trained AutoCkt model checkpoint
- Environment variables set (PYTHONPATH, RAY_TMPDIR)

## Usage

### Basic Command
```bash
conda activate autockt
export PATH=$HOME/ngspice-27/opt/bin:$PATH
export PYTHONPATH=$PWD
export RAY_TMPDIR=$HOME/AutoCkt/ray_tmp

python sizing_314510144.py --model <checkpoint_path> --spec <spec_file>
```

### Arguments
- `--model`: Path to trained agent checkpoint (e.g., `/path/to/checkpoint-100`)
- `--spec`: Path to JSON specification file
- `--traj_len`: Maximum trajectory length (default: 50, optional)

### Example
```bash
python sizing_314510144.py \
    --model ~/ray_results/train_45nm_ngspice/PPO_TwoStageAmp_xxx/checkpoint_100/checkpoint-100 \
    --spec spec1.json \
    --traj_len 30
```

## Specification File Format

The JSON spec file must contain these four parameters:

```json
{
    "gain_min": 300,
    "ibias_max": 0.006,
    "phm_min": 60.0,
    "ugbw_min": 20000000.0
}
```

### Valid Ranges
- `gain_min`: 200 - 400
- `ibias_max`: 0.0001 - 0.01
- `phm_min`: 60 (fixed)
- `ugbw_min`: 1.0e6 - 2.5e7

## Output

### Console Output
The script provides detailed progress information:
- Target specifications
- Step-by-step optimization progress
- Final performance metrics
- Tuned parameter values

### Output File
**`final_design.cir`**: Complete SPICE netlist with tuned parameters

This file is ready for NGSpice simulation and validation.

## How It Works

1. **Load Specifications**: Reads target specs from JSON file
2. **Initialize Agent**: Loads trained RL agent from checkpoint
3. **Optimization Loop**: 
   - Agent proposes parameter adjustments
   - NGSpice simulates the circuit
   - Reward calculated based on spec satisfaction
   - Process repeats until specs are met or max steps reached
4. **Generate Netlist**: Fills template with final tuned parameters

## Tunable Parameters

The script optimizes these 7 parameters:
- `mp1`, `mn1`: Multipliers for differential pair transistors
- `mp3`, `mn3`: Multipliers for current mirror transistors  
- `mn4`, `mn5`: Multipliers for bias and output transistors
- `cc`: Compensation capacitor value

Widths (0.5μm) and lengths (90nm) are fixed.

## Performance Tips

### For Faster Results
- Use lower `--traj_len` (e.g., 20-30) for quicker convergence
- Trade-off: May not reach optimal solution

### For Better Quality
- Use higher `--traj_len` (e.g., 50-100)
- Trade-off: Takes longer to execute

### Improving Success Rate
- Ensure your checkpoint is well-trained (85%+ validation success)
- Specs closer to training distribution work better
- Try multiple runs if first attempt doesn't succeed

## Troubleshooting

### "Could not find params.json"
- Verify checkpoint path is correct
- Path should point to `checkpoint-###` file, not directory

### "Template file not found"
- Ensure `final_design_template_v2.cir` exists in AutoCkt root
- Run from AutoCkt directory: `cd ~/AutoCkt`

### "Ray init error"
- Check RAY_TMPDIR is set and directory exists
- Try: `ray stop` before running

### Low success rate
- Increase `--traj_len` value
- Check if specs are within valid training ranges
- Retrain agent with more diverse specs

## Validation

After generating `final_design.cir`, validate it:

```bash
/tmp/ngspice_model/ngspice_checker.py \
    --netlist final_design.cir \
    --spec spec1.json
```

## File Structure

```
AutoCkt/
├── sizing_314510144.py          # Main sizing script
├── final_design_template_v2.cir # SPICE netlist template
├── spec1.json                   # Example spec file
├── spec2.json                   # Example spec file
└── autockt/
    └── envs/
        └── ngspice_vanilla_opamp.py  # Environment definition
```

## Notes

- **Student ID**: Replace `314510144` with your actual student ID
- **Model Path**: Use your own trained checkpoint, not the example path
- **Execution Location**: Always run from AutoCkt root directory
- **Python Packages**: Only uses packages from `environment.yml`

## Example Session

```bash
$ conda activate autockt
$ cd ~/AutoCkt
$ export PATH=$HOME/ngspice-27/opt/bin:$PATH
$ export PYTHONPATH=$PWD
$ export RAY_TMPDIR=$HOME/AutoCkt/ray_tmp

$ python sizing_314510144.py \
    --model ~/ray_results/train_45nm_ngspice/PPO_TwoStageAmp_xxx/checkpoint_50/checkpoint-50 \
    --spec spec1.json

============================================================
AutoCkt Sizing Tool
============================================================
Model checkpoint: /home/.../checkpoint-50
Specification file: spec1.json
Max trajectory length: 50
============================================================

Loading target specifications...
Target specs loaded: {'gain_min': 300, 'ibias_max': 0.006, ...}

Initializing Ray...
Loading agent from checkpoint...
Agent restored successfully!

Running agent to find optimal design...
============================================================
Target specs: gain_min=300.00, ibias_max=0.006000, ...
------------------------------------------------------------
Step 0: reward=-0.5234, gain=245.32, ...
Step 5: reward=-0.2156, gain=278.45, ...
...
Step 25: reward=10.0000, gain=310.52, ...

============================================================
SUCCESS! Design meets all specifications!
============================================================

Final Performance:
  Gain: 310.52 (target: >=300)
  Ibias: 0.00545 (target: <=0.006)
  Phase Margin: 62.34° (target: >=60)
  UGBW: 2.15e+07 Hz (target: >=2.00e+07)

Generating final netlist...

Final netlist written to: final_design.cir

Tuned Parameters:
  mp1=12, mn1=45
  mp3=5, mn3=8
  mn4=22, mn5=58
  cc=4.250p (4.25e-12F)

============================================================
Sizing completed successfully!
============================================================
```
