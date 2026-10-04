"""Summarize saved experiment JSONs with one statistical definition.

Every table entry is  mean ± sample std (ddof=1)  over the seeds, where the
value of a seed is computed over that seed's test problems:
  - success rate : fraction of test problems solved
  - plan time    : mean time over the solved problems
  - plan length  : mean length over the solved problems
Problems that fail or time out are excluded from time and length and are
counted in the success rate. Per-instance statistics (median, p90, max of plan
time, pooled over seeds and solved problems) are reported separately.

Usage:
    python summarize_results.py [experiment_results]
"""

import glob
import json
import os
import sys

import numpy as np


def per_seed_lists(m, key_seed, key_flat):
    """Per-seed lists. Old result files only stored a flat list; the seeds are
    recovered from the per-seed solved counts (times are appended in order)."""
    if key_seed in m:
        return [list(x) for x in m[key_seed]]
    out, i = [], 0
    for s in m["solved"]:
        out.append(list(m[key_flat][i:i + s]))
        i += s
    return out


def mean_std(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def summarize(m):
    times = per_seed_lists(m, "plan_times_per_seed", "plan_times")
    rate = [100.0 * s / t for s, t in zip(m["solved"], m["total"])]
    out = {
        "success": mean_std(rate),
        "time": mean_std([np.mean(t) if t else None for t in times]),
        "train": mean_std(m["train_time"]),
        "ops": mean_std(m["operators"]),
        "preds": mean_std(m["predicates"]),
        "n_failed": int(sum(m["total"]) - sum(m["solved"])),
    }
    if "plan_lengths_per_seed" in m:
        lens = m["plan_lengths_per_seed"]
        out["length"] = mean_std([np.mean(l) if l else None for l in lens])
        flat = [x for l in lens for x in l]
        out["length_range"] = (min(flat), max(flat)) if flat else None
    else:
        out["length"] = mean_std(m["plan_lengths"])
        out["length_range"] = None
    pooled = [x for t in times for x in t]
    if pooled:
        out["tail"] = (float(np.median(pooled)), float(np.percentile(pooled, 90)),
                       float(np.max(pooled)), len(pooled))
    return out


def fmt(ms, nd=3):
    if ms is None or ms[0] is None:
        return "Failed"
    return f"{ms[0]:.{nd}f} ± {ms[1]:.{nd}f}"


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "experiment_results"
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        d = json.load(open(path))
        print(f"\n## {os.path.basename(path)}  ({d['num_seeds']} seeds)")
        for name, m in d["results"].items():
            s = summarize(m)
            tail = (f"median {s['tail'][0]:.4f} / p90 {s['tail'][1]:.4f} / "
                    f"max {s['tail'][2]:.2f} (n={s['tail'][3]})") if "tail" in s else "no solved problem"
            rng = f" range {s['length_range']}" if s["length_range"] else ""
            print(f"  {name:11s} success {fmt(s['success'], 1)}% (failed {s['n_failed']}) | "
                  f"time {fmt(s['time'])} s | length {fmt(s['length'], 1)}{rng} | "
                  f"ops {fmt(s['ops'], 1)} | preds {fmt(s['preds'], 1)} | train {fmt(s['train'])} s")
            print(f"  {'':11s} per-instance time: {tail}")


if __name__ == "__main__":
    main()
