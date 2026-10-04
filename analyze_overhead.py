"""Where does the planning time go?  Per-instance diagnostics for one domain.

For every test problem it records whether the problem was solved, the total plan
time, the number of skeletons tried, the time spent in the continuous refinement
(sampling and simulation) and the number of simulator calls. It also stores the
learned operators of every seed.

The planner itself is not modified. BaseApproach.plan is wrapped in this script by
an instrumented copy with identical behaviour.

Usage:
    python analyze_overhead.py --env cover --num_seeds 50 --tag 50seeds
"""

import os
import json
import time
import argparse
import pickle as pkl
import contextlib
import io
import numpy as np

from envs import create_env
from approaches import (LOFT, LOFTMacro, BaseApproach,
                        ApproachTimeout, ApproachFailed)
from settings import create_config


def instrumented_plan(self, init_state, goal, timeout):
    """Same logic as BaseApproach.plan, plus bookkeeping."""
    start_time = time.time()
    stats = {"skeletons": 0, "refine_time": 0.0, "outcome": "solved"}
    self.last_stats = stats
    skeleton_gen = self._skeleton_generator(init_state, goal, timeout)
    sampler_rng = np.random.RandomState(self._seed + self._num_calls)
    self._num_calls += 1
    try:
        while True:
            try:
                skeleton, expected_lits_sequence = next(skeleton_gen)
            except StopIteration:
                break
            stats["skeletons"] += 1
            t0 = time.time()
            try:
                plan = self._sample_continuous_values(
                    init_state, goal, skeleton, expected_lits_sequence,
                    sampler_rng, start_time, timeout)
            finally:
                stats["refine_time"] += time.time() - t0
            if plan is not None:
                return plan
        if time.time() - start_time > timeout:
            stats["outcome"] = "timeout_in_search"
            raise ApproachTimeout("Timed out in skeleton search!")
        stats["outcome"] = "out_of_skeletons"
        raise ApproachFailed("Ran out of skeletons!")
    except ApproachTimeout:
        if stats["outcome"] == "solved":
            stats["outcome"] = "timeout_in_refinement"
        raise
    finally:
        stats["total"] = time.time() - start_time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True,
                   choices=["cover", "blocks", "painting", "kitchen"])
    p.add_argument("--start_seed", type=int, default=0)
    p.add_argument("--num_seeds", type=int, default=50)
    p.add_argument("--collect_data", type=int, default=0)
    p.add_argument("--approaches", type=str, default="LOFT,Macro")
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    config = create_config(args)
    BaseApproach.plan = instrumented_plan
    classes = {"LOFT": LOFT, "Macro": LOFTMacro}
    names = [n.strip() for n in args.approaches.split(",")]
    with open(os.path.join(config.data_dir, f"{args.env}.p"), "rb") as f:
        data = pkl.load(f)

    records = {n: [] for n in names}
    operators = {n: [] for n in names}
    seeds_used = []
    for seed in range(args.start_seed, args.start_seed + args.num_seeds + 100):
        if len(seeds_used) >= args.num_seeds:
            break
        env = create_env(config)
        env.set_seed(seed)
        try:
            problems = env.get_test_problems()
        except Exception:
            continue
        for n in names:
            env.set_seed(seed)
            counter = {"sim": 0, "samples": 0}

            def sim(state, action, _f=env.get_next_state, _c=counter):
                _c["sim"] += 1
                return _f(state, action)

            ap = classes[n](config, sim, env.get_state_predicates(),
                            env.get_action_predicates())
            ap.set_seed(seed)
            orig_sample = ap._sample_act_args

            def sample(*a, _o=orig_sample, _c=counter, **k):
                _c["samples"] += 1
                return _o(*a, **k)

            ap._sample_act_args = sample
            with contextlib.redirect_stdout(io.StringIO()):
                ap.train(data)
            operators[n].append(sorted(str(op) for op in ap._operators))
            for i, (init_state, goal) in enumerate(problems):
                counter["sim"] = counter["samples"] = 0
                t0 = time.time()
                solved, plan_len = False, None
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        plan = ap.plan(init_state, goal, config.approach_timeout)
                    total = time.time() - t0
                    state = init_state
                    for act in plan:
                        state = env.get_next_state(state, act)
                    solved, plan_len = bool(goal.holds(state)), len(plan)
                except (ApproachFailed, ApproachTimeout):
                    total = time.time() - t0
                st = ap.last_stats
                records[n].append(dict(
                    seed=seed, problem=i, solved=solved, total=total,
                    skeletons=st["skeletons"], refine=st["refine_time"],
                    outcome=st["outcome"] if not solved else "solved",
                    sim_calls=counter["sim"], samples=counter["samples"],
                    plan_len=plan_len))
        seeds_used.append(seed)
        print(f"seed {seed} done", flush=True)

    os.makedirs("experiment_results", exist_ok=True)
    path = os.path.join("experiment_results",
                        f"{args.env}_overhead{('_' + args.tag) if args.tag else ''}.json")
    json.dump(dict(env=args.env, seeds_used=seeds_used, records=records,
                   operators=operators), open(path, "w"))
    print("saved", path)


if __name__ == "__main__":
    main()
