#!/usr/bin/env python
"""
Sizing script for AutoCkt - Automatically tunes transistor parameters
to meet target specifications using a trained RL agent.

Usage:
    python sizing_314510144.py --model <checkpoint_path> --spec <spec_json_file>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import numpy as np
import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from collections import OrderedDict

# Import the environment
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AutoCkt sizing tool - tune transistor parameters to meet specs"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained agent checkpoint"
    )
    parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Path to specification JSON file"
    )
    parser.add_argument(
        "--traj_len",
        type=int,
        default=50,
        help="Maximum trajectory length (default: 50)"
    )
    return parser.parse_args()


def load_specifications(spec_file):
    """Load target specifications from JSON file"""
    with open(spec_file, 'r') as f:
        specs = json.load(f)
    
    # Validate required specifications
    required_specs = ['gain_min', 'ibias_max', 'phm_min', 'ugbw_min']
    for spec in required_specs:
        if spec not in specs:
            raise ValueError(f"Missing required specification: {spec}")
    
    return specs


def create_custom_env_config(target_specs):
    """
    Create environment configuration with custom target specifications.
    This modifies the environment to use the provided specs instead of random ones.
    """
    # Save custom specs to a temporary pickle file that the environment can load
    custom_specs = OrderedDict([
        ('gain_min', (target_specs['gain_min'],)),
        ('ibias_max', (target_specs['ibias_max'],)),
        ('phm_min', (target_specs['phm_min'],)),
        ('ugbw_min', (target_specs['ugbw_min'],))
    ])
    
    # Create temporary spec file
    temp_spec_path = os.path.join(
        os.getcwd(),
        "autockt/gen_specs/ngspice_specs_gen_two_stage_opamp"
    )
    
    with open(temp_spec_path, 'wb') as f:
        pickle.dump(custom_specs, f)
    
    env_config = {
        "generalize": True,
        "num_valid": 1,
        "save_specs": False,
        "run_valid": True,
    }
    
    return env_config


def unlookup(norm_spec, goal_spec):
    """Convert normalized specs back to actual values"""
    spec = -1 * np.multiply((norm_spec + 1), goal_spec) / (norm_spec - 1)
    return spec


def run_agent(agent, env, max_steps):
    """
    Run the trained agent to find a design that meets specifications.
    Returns the final parameters and whether specs were reached.
    """
    state = env.reset()
    done = False
    steps = 0
    best_reward = float('-inf')
    best_params = None
    best_specs = None
    
    print(f"Target specs: gain_min={env.specs_ideal[0]:.2f}, "
          f"ibias_max={env.specs_ideal[1]:.6f}, "
          f"phm_min={env.specs_ideal[2]:.2f}, "
          f"ugbw_min={env.specs_ideal[3]:.2e}")
    print("-" * 60)
    
    while not done and steps < max_steps:
        # Compute action from agent
        action = agent.compute_action(state)
        
        # Take step in environment
        next_state, reward, done, _ = env.step(action)
        
        # Track best design found so far
        if reward > best_reward:
            best_reward = reward
            best_params = env.cur_params_idx.copy()
            best_specs = env.cur_specs.copy()
        
        if steps % 5 == 0:
            print(f"Step {steps}: reward={reward:.4f}, "
                  f"gain={env.cur_specs[0]:.2f}, "
                  f"ibias={env.cur_specs[1]:.6f}, "
                  f"phm={env.cur_specs[2]:.2f}, "
                  f"ugbw={env.cur_specs[3]:.2e}")
        
        steps += 1
        state = next_state
        
        if done:
            print("\n" + "=" * 60)
            print("SUCCESS! Design meets all specifications!")
            print("=" * 60)
            break
    
    if not done:
        print("\n" + "=" * 60)
        print(f"Maximum steps ({max_steps}) reached. Using best design found.")
        print(f"Best reward: {best_reward:.4f}")
        print("=" * 60)
        # Use best params found if goal not reached
        if best_params is not None:
            env.cur_params_idx = best_params
            env.cur_specs = best_specs
    
    return env.cur_params_idx, env.cur_specs, done


def extract_param_values(param_indices, env):
    """Extract actual parameter values from indices"""
    param_values = {}
    for i, param_id in enumerate(env.params_id):
        param_values[param_id] = env.params[i][param_indices[i]]
    return param_values


def generate_netlist(param_values, output_file="final_design.cir"):
    """
    Generate the final SPICE netlist with tuned parameters.
    Uses the template structure and fills in the TODO section.
    """
    # Read the template
    template_path = os.path.join(os.getcwd(), "final_design_template_v2.cir")
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    with open(template_path, 'r') as f:
        template = f.read()
    
    # The parameters from the environment are: mp1, mn1, mp3, mn3, mn4, mn5, cc
    # These are the 7 tunable parameters according to the YAML file
    # Widths are fixed at 0.5u, lengths at 90n
    mp1 = int(param_values.get('mp1', 10))
    mn1 = int(param_values.get('mn1', 38))
    mp3 = int(param_values.get('mp3', 4))
    mn3 = int(param_values.get('mn3', 9))
    mn4 = int(param_values.get('mn4', 20))
    mn5 = int(param_values.get('mn5', 60))
    cc = param_values.get('cc', 3.0e-12)  # cc is in Farads
    
    # Create the parameter lines (fill in TODO section)
    # Widths are fixed at 0.5u, lengths are fixed at 90n
    # cc needs to be converted from Farads to picoFarads
    cc_pf = cc * 1e12
    
    param_lines = f""".param wp1=0.5u lp1=90n mp1={mp1}
.param wn1=0.5u ln1=90n mn1={mn1}
.param wn3=0.5u ln3=90n mn3={mn3}
.param wp3=0.5u lp3=90n mp3={mp3}
.param wn4=0.5u ln4=90n mn4={mn4}
.param wn5=0.5u ln5=90n mn5={mn5}
.param cc={cc_pf}p"""
    
    # Replace the TODO section in template
    # Find the TODO section and replace it
    start_marker = "*********** TODO ***********"
    end_marker = "****************************"
    
    start_idx = template.find(start_marker)
    end_idx = template.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find TODO markers in template")
    
    # Reconstruct the netlist
    netlist = (template[:start_idx + len(start_marker)] + "\n" +
               param_lines + "\n" +
               template[end_idx:])
    
    # Write the final netlist
    with open(output_file, 'w') as f:
        f.write(netlist)
    
    print(f"\nFinal netlist written to: {output_file}")
    print("\nTuned Parameters:")
    print(f"  mp1={mp1}, mn1={mn1}")
    print(f"  mp3={mp3}, mn3={mn3}")
    print(f"  mn4={mn4}, mn5={mn5}")
    print(f"  cc={cc_pf:.3f}p ({cc:.2e}F)")


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("=" * 60)
    print("AutoCkt Sizing Tool")
    print("=" * 60)
    print(f"Model checkpoint: {args.model}")
    print(f"Specification file: {args.spec}")
    print(f"Max trajectory length: {args.traj_len}")
    print("=" * 60 + "\n")
    
    # Load target specifications
    print("Loading target specifications...")
    target_specs = load_specifications(args.spec)
    print(f"Target specs loaded: {target_specs}\n")
    
    # Create custom environment configuration
    env_config = create_custom_env_config(target_specs)
    
    # Initialize Ray
    print("Initializing Ray...")
    tmp = os.environ.get("RAY_TMPDIR")
    ray.shutdown()
    ray.init(temp_dir=tmp, ignore_reinit_error=True)
    
    # Register environment
    register_env("opamp-v0", lambda cfg: TwoStageAmp(env_config=cfg))
    
    # Load agent configuration from checkpoint
    print(f"Loading agent from checkpoint: {args.model}...")
    config_dir = os.path.dirname(args.model)
    config_path = os.path.join(config_dir, "params.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.json")
    
    if not os.path.exists(config_path):
        raise ValueError("Could not find params.json in checkpoint directory")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Update config for single evaluation
    if "num_workers" in config:
        config["num_workers"] = 0
    
    config["env"] = "opamp-v0"
    config["env_config"] = env_config
    
    # Create and restore agent
    print("Creating PPO agent...")
    cls = get_agent_class("PPO")
    agent = cls(env="opamp-v0", config=config)
    agent.restore(args.model)
    print("Agent restored successfully!\n")
    
    # Create environment for running
    env = TwoStageAmp(env_config=env_config)
    
    # Run agent to find design
    print("Running agent to find optimal design...")
    print("=" * 60)
    final_params_idx, final_specs, success = run_agent(agent, env, args.traj_len)
    
    # Extract actual parameter values
    param_values = extract_param_values(final_params_idx, env)
    
    # Display final results
    print("\nFinal Performance:")
    print(f"  Gain: {final_specs[0]:.2f} (target: >={target_specs['gain_min']})")
    print(f"  Ibias: {final_specs[1]:.6f} (target: <={target_specs['ibias_max']})")
    print(f"  Phase Margin: {final_specs[2]:.2f}° (target: >={target_specs['phm_min']})")
    print(f"  UGBW: {final_specs[3]:.2e} Hz (target: >={target_specs['ugbw_min']})")
    
    # Generate final netlist
    print("\nGenerating final netlist...")
    generate_netlist(param_values, "final_design.cir")
    
    # Cleanup
    ray.shutdown()
    
    print("\n" + "=" * 60)
    print("Sizing completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
