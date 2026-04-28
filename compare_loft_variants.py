"""Compare LOFT variants across all five domains.

Variants:
  LOFT            — original LOFT baseline
  SafeRefinements — + subsumption filtering, Laplace smoothing, adaptive backtracking
  IPS             — + Iterative Predicate Selection
  Macro           — + macro-operator mining (full pipeline)

Usage:
    python compare_loft_variants.py --env cover --start_seed 0 --num_seeds 5
    python compare_loft_variants.py --env kitchen --start_seed 0 --num_seeds 5
    python compare_loft_variants.py --env blocks --start_seed 0 --num_seeds 5 --approaches LOFT,Macro
"""

import os
import json
import pickle as pkl
import time
import argparse
from collections import defaultdict
import numpy as np

from envs import create_env
from approaches import (LOFT, LOFTSafeRefinements, LOFTIPS, LOFTMacro,
                        ApproachTimeout, ApproachFailed)
from settings import create_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str,
                        choices=["cover", "blocks", "painting", "warehouse", "kitchen"])
    parser.add_argument("--start_seed", required=True, type=int)
    parser.add_argument("--num_seeds", required=True, type=int)
    parser.add_argument("--collect_data", type=int, default=0)
    parser.add_argument("--no-timeout", action="store_true",
                        help="Disable planning timeout (use 24h effective limit)")
    parser.add_argument("--approaches", type=str, default=None,
                        help="Comma-separated list: LOFT,SafeRefinements,IPS,Macro (default: all)")
    parser.add_argument("--gui", action="store_true",
                        help="Enable PyBullet GUI for visualization")
    return parser.parse_args()


def run_approach(approach_cls, config, env, data, test_problems, seed):
    """Run a single approach and collect metrics."""
    # Reset env for fair comparison
    env.set_seed(seed)
    
    simulator = env.get_next_state
    state_preds = env.get_state_predicates()
    action_preds = env.get_action_predicates()
    
    approach = approach_cls(config, simulator, state_preds, action_preds)
    approach.set_seed(seed)
    
    # Train
    train_start = time.time()
    approach.train(data)
    train_time = time.time() - train_start
    num_operators = len(approach._operators) if approach._operators else 0
    
    # Count predicates used (for IPS metric)
    num_preds = len(approach._state_preds)
    
    # Test
    solved, plan_times, plan_lengths = 0, [], []
    first_plan_data = None
    for init_state, goal in test_problems:
        plan_start = time.time()
        try:
            plan = approach.plan(init_state, goal, config.approach_timeout)
            plan_time = time.time() - plan_start
            
            # Validate
            state = init_state
            for act in plan:
                state = env.get_next_state(state, act)
            
            if goal.holds(state):
                solved += 1
                plan_times.append(plan_time)
                plan_lengths.append(len(plan))
                if first_plan_data is None:
                    first_plan_data = (init_state, goal, plan)
        except (ApproachFailed, ApproachTimeout):
            pass
    
    return {
        "train_time": train_time,
        "num_operators": num_operators,
        "num_preds": num_preds,
        "solved": solved,
        "total": len(test_problems),
        "plan_times": plan_times,
        "plan_lengths": plan_lengths,
        "first_plan": first_plan_data,
    }


def main():
    args = parse_args()
    config = create_config(args)
    if getattr(args, "no_timeout", False):
        config.approach_timeout = 86400  # 24 hours

    all_approaches = {
        "LOFT":             LOFT,
        "SafeRefinements":  LOFTSafeRefinements,
        "IPS":              LOFTIPS,
        "Macro":            LOFTMacro,
    }
    if args.approaches:
        names = [s.strip() for s in args.approaches.split(",")]
        approaches = {k: v for k, v in all_approaches.items() if k in names}
        if not approaches:
            raise SystemExit(f"Unknown approach names. Choose from: {list(all_approaches.keys())}")
    else:
        approaches = all_approaches
    
    # Load data once
    with open(os.path.join(config.data_dir, f"{args.env}.p"), "rb") as f:
        data = pkl.load(f)
    
    results = {name: defaultdict(list) for name in approaches}
    viz_data = None
    
    print(f"Comparing {list(approaches.keys())} on {args.env}")
    
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        print(f"\nSeed {seed}:")
        env = create_env(config)
        env.set_seed(seed)
        test_problems = env.get_test_problems()
        
        viz_data = None
        for name, cls in approaches.items():
            print(f"  Running {name}...", end=" ", flush=True)
            m = run_approach(cls, config, env, data, test_problems, seed)
            if getattr(config, "with_gui", False) and name == "Macro" and m.get("first_plan"):
                viz_data = m["first_plan"]
            
            results[name]["train"].append(m["train_time"])
            results[name]["ops"].append(m["num_operators"])
            results[name]["preds"].append(m["num_preds"])
            results[name]["solved"].append(m["solved"])
            results[name]["total"].append(m["total"])
            results[name]["plan_times"].extend(m["plan_times"])
            # Store mean plan length for this seed (NaN if no plans)
            if m["plan_lengths"]:
                results[name]["plan_lengths"].append(np.mean(m["plan_lengths"]))
            else:
                results[name]["plan_lengths"].append(float('nan'))
            
            avg_time = f"{np.mean(m['plan_times']):.4f}s" if m['plan_times'] else "N/A"
            print(f"Solved {m['solved']}/{m['total']} ({avg_time} avg)")
    
    # Results
    col_w = 22  # column width for mean ± std
    print(f"\n{'='*90}")
    print(f"RESULTS: {args.env.upper()} ({args.num_seeds} seeds)")
    print(f"{'='*90}")
    
    metrics = ["Training (s)", "Operators", "Predicates", "Success Rate",
               "Plan Time (s)", "Plan Length"]
    header = f"{'Metric':<18}"
    for name in approaches:
        header += f" {name:<{col_w}}"
    print(header)
    print("-" * (18 + (col_w + 1) * len(approaches)))
    
    # Helper to get stats with mean ± std
    def get_stats(data, key, is_rate=False):
        if key == "plan_times":
            # For plan_times, compute per-seed means first for std
            vals = data[key]  # flattened list
        elif key == "plan_lengths":
            vals = [v for v in data[key] if not np.isnan(v)]
        elif is_rate:
            vals = [s/t for s, t in zip(data["solved"], data["total"])]
        else:
            vals = data[key]
            
        if not vals: return "N/A"
        mean = np.mean(vals)
        std = np.std(vals)
        if is_rate:
            return f"{100*mean:.1f} ± {100*std:.1f}%"
        if key in ["ops", "preds"]:
            if std == 0:
                return f"{mean:.0f}"
            return f"{mean:.1f} ± {std:.1f}"
        if key == "plan_lengths":
            return f"{mean:.1f} ± {std:.1f}"
        return f"{mean:.4f} ± {std:.4f}"

    # Print Table
    for metric in metrics:
        key_map = {
            "Training (s)": "train",
            "Operators": "ops",
            "Predicates": "preds",
            "Success Rate": "solved",
            "Plan Time (s)": "plan_times",
            "Plan Length": "plan_lengths"
        }
        key = key_map[metric]
        is_rate = (metric == "Success Rate")
        
        row = f"{metric:<18}"
        for name in approaches:
            val = get_stats(results[name], key, is_rate)
            row += f" {val:<{col_w}}"
        print(row)
        
    print("-" * (18 + (col_w + 1) * len(approaches)))

    # Save structured results to JSON
    json_results = {}
    for name in approaches:
        json_results[name] = {
            "train_time": results[name]["train"],
            "operators": results[name]["ops"],
            "predicates": results[name]["preds"],
            "solved": results[name]["solved"],
            "total": results[name]["total"],
            "plan_times": results[name]["plan_times"],
            "plan_lengths": [v if not np.isnan(v) else None
                             for v in results[name]["plan_lengths"]],
        }
    os.makedirs("experiment_results", exist_ok=True)
    json_path = os.path.join("experiment_results", f"{args.env}_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "env": args.env,
            "num_seeds": args.num_seeds,
            "start_seed": args.start_seed,
            "results": json_results
        }, f, indent=2)
    print(f"\nResults saved to {json_path}")

    if getattr(config, "with_gui", False):
        # Animate first successful plan if available (Macro on Kitchen, etc.)
        if viz_data and hasattr(env, "_reset_pybullet_from_state"):
            init_state, goal, plan = viz_data
            print("\n[GUI] Animating plan execution...")
            state = init_state
            env._reset_pybullet_from_state(state)
            for step, act in enumerate(plan):
                time.sleep(0.6)
                state = env.get_next_state(state, act)
                env._reset_pybullet_from_state(state)
            print(f"[GUI] Plan complete ({len(plan)} steps).")
        print("\n[GUI] Press Enter to close the PyBullet window...")
        input()

if __name__ == "__main__":
    main()
