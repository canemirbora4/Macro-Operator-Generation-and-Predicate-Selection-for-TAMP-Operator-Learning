"""Proper Ablation Study: Full model minus one component at a time.

Usage:
    python run_ablation_proper.py --env cover --start_seed 0 --num_seeds 5
    python run_ablation_proper.py --env blocks --start_seed 0 --num_seeds 5
"""

import os
import json
import pickle as pkl
import time
import argparse
from collections import defaultdict
import numpy as np

from envs import create_env
from approaches import LOFTMacro, LOFTIPS, ApproachTimeout, ApproachFailed
from approaches.ablation_proper import FullMinusIPS, FullMinusMacro
from settings import create_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str,
                        choices=["cover", "blocks", "painting", "kitchen"])
    parser.add_argument("--start_seed", required=True, type=int)
    parser.add_argument("--num_seeds", required=True, type=int)
    parser.add_argument("--collect_data", type=int, default=0)
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated subset of: Full Model,- IPS,- Macro")
    return parser.parse_args()


def run_approach(approach_cls, config, env, data, test_problems, seed):
    """Run a single approach and collect metrics."""
    env.set_seed(seed)
    simulator = env.get_next_state
    state_preds = env.get_state_predicates()
    action_preds = env.get_action_predicates()
    
    approach = approach_cls(config, simulator, state_preds, action_preds)
    approach.set_seed(seed)
    
    train_start = time.time()
    approach.train(data)
    train_time = time.time() - train_start
    num_operators = len(approach._operators) if approach._operators else 0
    num_preds = len(approach._state_preds)
    
    solved, plan_times, plan_lengths = 0, [], []
    for init_state, goal in test_problems:
        plan_start = time.time()
        try:
            plan = approach.plan(init_state, goal, config.approach_timeout)
            plan_time = time.time() - plan_start
            
            state = init_state
            for act in plan:
                state = env.get_next_state(state, act)
            
            if goal.holds(state):
                solved += 1
                plan_times.append(plan_time)
                plan_lengths.append(len(plan))
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
    }


def main():
    args = parse_args()
    config = create_config(args)

    all_approaches = {
        "Full Model": LOFTMacro,
        "- IPS":      FullMinusIPS,
        "- Macro":    FullMinusMacro,
    }
    if args.variants:
        names = [v.strip() for v in args.variants.split(",")]
        approaches = {k: v for k, v in all_approaches.items() if k in names}
    else:
        approaches = all_approaches
    
    with open(os.path.join(config.data_dir, f"{args.env}.p"), "rb") as f:
        data = pkl.load(f)
    
    results = {name: defaultdict(list) for name in approaches}
    
    print(f"PROPER ABLATION on {args.env} ({args.num_seeds} seeds)")
    print(f"Variants: {list(approaches.keys())}")
    
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        print(f"\nSeed {seed}:")
        env = create_env(config)
        env.set_seed(seed)
        test_problems = env.get_test_problems()
        
        for name, cls in approaches.items():
            print(f"  {name}...", end=" ", flush=True)
            m = run_approach(cls, config, env, data, test_problems, seed)
            
            results[name]["train"].append(m["train_time"])
            results[name]["ops"].append(m["num_operators"])
            results[name]["preds"].append(m["num_preds"])
            results[name]["solved"].append(m["solved"])
            results[name]["total"].append(m["total"])
            results[name]["plan_times"].extend(m["plan_times"])
            if m["plan_lengths"]:
                results[name]["plan_lengths"].append(np.mean(m["plan_lengths"]))
            else:
                results[name]["plan_lengths"].append(float('nan'))
            
            avg_time = f"{np.mean(m['plan_times']):.4f}s" if m['plan_times'] else "N/A"
            print(f"Solved {m['solved']}/{m['total']} ({avg_time} avg)")
    
    # Results table
    col_w = 22
    print(f"\n{'='*110}")
    print(f"PROPER ABLATION: {args.env.upper()} ({args.num_seeds} seeds)")
    print(f"{'='*110}")
    
    metrics = ["Training (s)", "Operators", "Predicates", "Success Rate",
               "Plan Time (s)", "Plan Length"]
    header = f"{'Metric':<18}"
    for name in approaches:
        header += f" {name:<{col_w}}"
    print(header)
    print("-" * (18 + (col_w + 1) * len(approaches)))
    
    def get_stats(data, key, is_rate=False):
        if key == "plan_times":
            vals = data[key]
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
            return f"{mean:.0f}" if std == 0 else f"{mean:.1f} ± {std:.1f}"
        if key == "plan_lengths":
            return f"{mean:.1f} ± {std:.1f}"
        return f"{mean:.4f} ± {std:.4f}"

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

    # Save JSON
    json_results = {}
    for name in approaches:
        json_results[name] = {
            "train_time": results[name]["train"],
            "operators": results[name]["ops"],
            "predicates": list(map(int, results[name]["preds"])),
            "solved": results[name]["solved"],
            "total": results[name]["total"],
            "plan_times": results[name]["plan_times"],
            "plan_lengths": [v if not np.isnan(v) else None
                             for v in results[name]["plan_lengths"]],
        }
    os.makedirs("experiment_results", exist_ok=True)
    json_path = os.path.join("experiment_results", f"{args.env}_ablation_proper.json")
    with open(json_path, "w") as f:
        json.dump({
            "env": args.env,
            "num_seeds": args.num_seeds,
            "start_seed": args.start_seed,
            "results": json_results
        }, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
