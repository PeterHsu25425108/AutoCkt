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
    
    # print specs for verification
    print("Loaded Specifications:")
    for key, value in specs.items():
        print("{}: {}".format(key, value))
    
    # Validate required specifications
    required_specs = ['gain_min', 'ibias_max', 'phm_min', 'ugbw_min']
    for spec in required_specs:
        if spec not in specs:
            raise ValueError("Missing required specification: {}".format(spec))
    
    
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
    
    print("Target specs: gain_min={:.2f}, ibias_max={:.6f}, phm_min={:.2f}, ugbw_min={:.2e}".format(
          env.specs_ideal[0], env.specs_ideal[1], env.specs_ideal[2], env.specs_ideal[3]))
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
            print("Step {}: reward={:.4f}, gain={:.2f}, ibias={:.6f}, phm={:.2f}, ugbw={:.2e}".format(
                  steps, reward, env.cur_specs[0], env.cur_specs[1], env.cur_specs[2], env.cur_specs[3]))
        
        steps += 1
        state = next_state
        
        if done:
            print("\n" + "=" * 60)
            print("SUCCESS! Design meets all specifications!")
            print("=" * 60)
            break
    
    if not done:
        print("\n" + "=" * 60)
        print("Maximum steps ({}) reached. Using best design found.".format(max_steps))
        print("Best reward: {:.4f}".format(best_reward))
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
        raise FileNotFoundError("Template file not found: {}".format(template_path))
    
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
    
    param_lines = """.param wp1=0.5u lp1=90n mp1={}
.param wn1=0.5u ln1=90n mn1={}
.param wn3=0.5u ln3=90n mn3={}
.param wp3=0.5u lp3=90n mp3={}
.param wn4=0.5u ln4=90n mn4={}
.param wn5=0.5u ln5=90n mn5={}
.param cc={}p""".format(mp1, mn1, mn3, mp3, mn4, mn5, cc_pf)
    
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
    
    print("\nFinal netlist written to: {}".format(output_file))
    print("\nTuned Parameters:")
    print("  mp1={}, mn1={}".format(mp1, mn1))
    print("  mp3={}, mn3={}".format(mp3, mn3))
    print("  mn4={}, mn5={}".format(mn4, mn5))
    print("  cc={:.3f}p ({:.2e}F)".format(cc_pf, cc))


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("=" * 60)
    print("AutoCkt Sizing Tool")
    print("=" * 60)
    print("Model checkpoint: {}".format(args.model))
    print("Specification file: {}".format(args.spec))
    print("Max trajectory length: {}".format(args.traj_len))
    print("=" * 60 + "\n")
    
    # Load target specifications
    print("Loading target specifications...")
    target_specs = load_specifications(args.spec)
    print("Target specs loaded: {}\n".format(target_specs))
    
    # Create custom environment configuration
    env_config = create_custom_env_config(target_specs)
    
    # Initialize Ray
    print("Initializing Ray...")
    tmp = os.environ.get("RAY_TMPDIR")
    ray.shutdown()
    ray.init(temp_dir=tmp, ignore_reinit_error=True)
    
    # Register environment
    register_env("opamp-v0", lambda cfg: TwoStageAmp(env_config=cfg))
    
    # Hard-coded agent configuration (from trained model)
    print("Loading agent from checkpoint: {}...".format(args.model))
    config = {
        "env": "opamp-v0",
        "env_config": env_config,
        "horizon": 30,
        "model": {
            "fcnet_hiddens": [64, 64]
        },
        "num_gpus": 0,
        "num_workers": 0,
        "train_batch_size": 1200
    }
    
    # Create and restore agent
    print("Creating PPO agent...")
    cls = get_agent_class("PPO")
    agent = cls(env="opamp-v0", config=config)
    agent._restore(args.model)
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
    print("  Gain: {:.2f} (target: >={:.2f})".format(final_specs[0], target_specs['gain_min']))
    print("  Ibias: {:.6f} (target: <={:.6f})".format(final_specs[1], target_specs['ibias_max']))
    print("  Phase Margin: {:.2f}° (target: >={:.2f})".format(final_specs[2], target_specs['phm_min']))
    print("  UGBW: {:.2e} Hz (target: >={:.2e})".format(final_specs[3], target_specs['ugbw_min']))
    
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
