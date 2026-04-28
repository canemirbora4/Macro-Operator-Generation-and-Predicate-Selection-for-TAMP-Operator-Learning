"""Macro-Operator Mining for LOFT.

Full pipeline: safe refinements + IPS + automated macro-operator discovery
via effect-complement analysis.

The mining procedure consists of five phases:
  1. Compute typical ADD/DELETE effects per action type from demonstrations.
  2. Identify causal pairs: ADD(a1) ∩ DEL(a2) contains an object-specific
     predicate (arity > 0), meaning a1 produces a condition a2 requires.
  3. Validate pairs empirically (must occur consecutively in data) and
     resolve symmetric conflicts by keeping the higher-frequency direction.
  4. Build macro transitions: each consecutive (a1 → a2) occurrence becomes
     a single macro step (x_i, M, x_{i+2}) via argument unification.
  5. Generate synthetic negative examples to balance the training set.

Discovered macros are registered as new action types and passed through
LOFT's standard operator-learning pipeline identically to individual actions.
At execution time, every macro in the returned plan is expanded back into
its two underlying sub-actions.
"""

import copy
from collections import Counter, defaultdict

from approaches.ips import LOFTIPS
from structs import Predicate, Literal, WORLD
from utils import construct_effects


class LOFTMacro(LOFTIPS):
    """LOFTIPS + automatic macro-operator mining via effect-complement analysis."""

    def __init__(self, config, simulator, state_preds, action_preds):
        super().__init__(config, simulator, state_preds, action_preds)
        self._macro_preds = {}       # name → (pred, sub_actions_info, indices_info)
        self._original_simulator = simulator
        self._simulator = self._macro_simulator

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data):
        demos, random_data = data
        all_trajs = demos + random_data

        # ==== Phase 1: Typical ADD/DELETE effects per action type ====
        # Use only successful demonstration trajectories for effect profiling;
        # random_data contains many failed actions that dilute effect ratios.
        action_adds = defaultdict(lambda: defaultdict(int))
        action_dels = defaultdict(lambda: defaultdict(int))
        action_count = defaultdict(int)
        pred_arity = {}

        for traj in demos:
            for (state, action, next_state, *_) in traj:
                if not hasattr(action, 'predicate'):
                    continue
                name = action.predicate.name
                action_count[name] += 1

                hl_state = self._parser(state)
                hl_next = self._parser(next_state)
                effects = construct_effects(hl_state, hl_next)

                for eff in effects:
                    if hasattr(eff, 'is_anti') and eff.is_anti:
                        inv = eff.inverted_anti
                        action_dels[name][inv.predicate.name] += 1
                        pred_arity[inv.predicate.name] = inv.predicate.arity
                    else:
                        action_adds[name][eff.predicate.name] += 1
                        pred_arity[eff.predicate.name] = eff.predicate.arity

        # Typical effects: appear in >50% of the action's transitions.
        typical_adds = {}
        typical_dels = {}
        for act_name in action_count:
            n = action_count[act_name]
            typical_adds[act_name] = {
                p for p, c in action_adds[act_name].items() if c / n > 0.5}
            typical_dels[act_name] = {
                p for p, c in action_dels[act_name].items() if c / n > 0.5}

        # ==== Phase 2: Causal pairs via effect-complement ====
        # Keep only overlapping predicates with arity > 0 (object-specific).
        # Arity-0 predicates such as HandEmpty cycle in both directions and
        # do not indicate a true subtask dependency.
        causal_pairs = set()
        for a1 in action_count:
            for a2 in action_count:
                if a1 == a2:
                    continue
                overlap = typical_adds[a1] & typical_dels[a2]
                meaningful = {p for p in overlap if pred_arity.get(p, 0) > 0}
                if meaningful:
                    causal_pairs.add((a1, a2))

        print(f"[MacroMine] Effect-complement candidates: {len(causal_pairs)}")
        for a1, a2 in sorted(causal_pairs):
            overlap = typical_adds[a1] & typical_dels[a2]
            meaningful = {p for p in overlap if pred_arity.get(p, 0) > 0}
            print(f"  {a1} → {a2}  (via: {', '.join(sorted(meaningful))})")

        # ==== Phase 3: Frequency validation and symmetric-pair resolution ====
        pair_freq = Counter()
        total_pairs = 0
        for traj in all_trajs:
            for i in range(len(traj) - 1):
                act1 = traj[i][1]
                act2 = traj[i + 1][1]
                if hasattr(act1, 'predicate') and hasattr(act2, 'predicate'):
                    pair_freq[(act1.predicate.name, act2.predicate.name)] += 1
                    total_pairs += 1

        validated = {pair for pair in causal_pairs if pair_freq[pair] > 0}

        # For symmetric pairs keep only the higher-frequency direction.
        to_remove = set()
        for (a1, a2) in validated:
            if (a2, a1) in validated and (a2, a1) not in to_remove:
                if pair_freq[(a1, a2)] >= pair_freq[(a2, a1)]:
                    to_remove.add((a2, a1))
                else:
                    to_remove.add((a1, a2))
        validated -= to_remove

        print(f"[MacroMine] Validated macros after frequency + symmetric filter:"
              f" {len(validated)}")
        for a1, a2 in sorted(validated):
            f = pair_freq[(a1, a2)]
            pct = 100 * f / total_pairs if total_pairs else 0
            print(f"  {a1} → {a2}  (freq={f}, {pct:.1f}%)")

        # ==== Phase 4: Build macro transitions ====
        macro_transitions = []
        for traj in all_trajs:
            for i in range(len(traj) - 1):
                trans1 = traj[i]
                trans2 = traj[i + 1]
                act1 = trans1[1]
                act2 = trans2[1]
                if not hasattr(act1, 'predicate') or \
                   not hasattr(act2, 'predicate'):
                    continue
                pair_key = (act1.predicate.name, act2.predicate.name)
                if pair_key in validated:
                    macro_info = self._create_macro_from_actions(act1, act2)
                    if macro_info is not None:
                        macro_action, _, _ = macro_info
                        macro_transitions.append(
                            (trans1[0], macro_action, trans2[2], []))

        print(f"[MacroMine] Generated {len(macro_transitions)} macro transitions.")

        # ==== Phase 5: Synthetic negative examples ====
        macro_negatives = self._generate_macro_negatives(demos, random_data)
        print(f"[MacroMine] Generated {len(macro_negatives)} negative examples.")

        # Register macro predicates as action types.
        for name, (pred, _, _) in self._macro_preds.items():
            self._action_preds.add(pred)

        augmented_demos = list(demos) + [[t] for t in macro_transitions]
        augmented_random = list(random_data) + [[t] for t in macro_negatives]

        LOFTIPS.train(self, (augmented_demos, augmented_random))

    # ------------------------------------------------------------------
    # Macro infrastructure
    # ------------------------------------------------------------------

    def _create_macro_from_actions(self, act1, act2):
        """Create (or retrieve) a macro predicate for the (act1, act2) pair.

        Arguments of both sub-actions are unified: objects appearing in both
        get a single shared slot. Two index vectors record the mapping from
        the macro's unified argument list back to each sub-action's arguments.
        Returns (macro_action, None, None); caller fills in the states.
        """
        unique_args = []
        seen_map = {}   # obj → index in unique_args

        indices1 = []
        indices2 = []

        for arg in act1.variables:
            if arg not in seen_map:
                seen_map[arg] = len(unique_args)
                unique_args.append(arg)
            indices1.append(seen_map[arg])

        for arg in act2.variables:
            if arg not in seen_map:
                seen_map[arg] = len(unique_args)
                unique_args.append(arg)
            indices2.append(seen_map[arg])

        sig = (f"{tuple(indices1)}-{tuple(indices2)}"
               .replace("(", "").replace(")", "")
               .replace(", ", "_").replace(",", "_"))
        macro_name = f"Macro-{act1.predicate.name}-{act2.predicate.name}-{sig}"

        if macro_name not in self._macro_preds:
            var_types = [arg.var_type for arg in unique_args]
            self._create_macro_predicate(
                macro_name, var_types,
                act1.predicate, act2.predicate,
                indices1, indices2
            )

        macro_pred, _, _ = self._macro_preds[macro_name]
        return (macro_pred(*unique_args), None, None)

    def _generate_macro_negatives(self, demos, random_data):
        """Synthesize negative examples for macro precondition learning.

        For each macro type, sample random states from random_data and record
        a no-effect transition (state unchanged). Subsampled at 50% and capped
        at 100 per macro type to prevent class imbalance.
        """
        import random
        negatives = []
        if not self._macro_preds:
            return negatives

        rng = random.Random(42)
        cap = 100 * len(self._macro_preds)

        for traj in random_data:
            for trans in traj:
                state = trans[0]
                action = trans[1]

                if not hasattr(action, 'predicate'):
                    continue

                for macro_name, (macro_pred, (pred1, pred2),
                                 (indices1, indices2)) in self._macro_preds.items():
                    var_types = macro_pred.var_types

                    objs_by_type = {}
                    for obj in state:
                        if obj == WORLD or not hasattr(obj, 'var_type'):
                            continue
                        vt = obj.var_type
                        objs_by_type.setdefault(vt, []).append(obj)

                    try:
                        macro_args = []
                        used_objs = set()
                        for i, vt in enumerate(var_types):
                            if vt.is_continuous:
                                macro_args.append(vt(f"?cont{i}"))
                            else:
                                available = [o for o in objs_by_type.get(vt, [])
                                             if o not in used_objs]
                                if not available:
                                    raise ValueError("no objects of required type")
                                chosen = rng.choice(available)
                                macro_args.append(chosen)
                                used_objs.add(chosen)

                        discrete = [a for a in macro_args
                                    if not a.var_type.is_continuous]
                        if len(discrete) != len(set(discrete)):
                            raise ValueError("duplicate args")

                        if rng.random() < 0.5:
                            negatives.append((state, macro_pred(*macro_args), state, []))
                    except (ValueError, KeyError, IndexError):
                        continue

                if len(negatives) >= cap:
                    return negatives

        return negatives

    def _create_macro_predicate(self, name, var_types, pred1, pred2,
                                indices1, indices2):
        """Dynamically create a new macro predicate with a sequential sampler.

        The sampler first invokes pred1's sampler, simulates the intermediate
        state, then invokes pred2's sampler conditioned on that state.
        """
        original_simulator = self._original_simulator

        def macro_sampler(rng, state, *args):
            is_continuous = [vt.is_continuous for vt in var_types]
            continuous_indices = [i for i, c in enumerate(is_continuous) if c]

            u2d = {}
            d_count = 0
            for i, cont in enumerate(is_continuous):
                if not cont:
                    u2d[i] = d_count
                    d_count += 1

            sampled_values = {}

            args1_discrete = [args[u2d[i]] for i in indices1
                              if not is_continuous[i]]
            res1 = pred1.sample(rng, state, *args1_discrete)
            res1_iter = iter(res1)

            act1_args = []
            args1_d_iter = iter(args1_discrete)
            for vt in pred1.var_types:
                if vt.is_continuous:
                    val = next(res1_iter)
                    act1_args.append(vt(f"sampled", val))
                else:
                    act1_args.append(next(args1_d_iter))

            act1 = pred1(*act1_args)
            try:
                next_state = original_simulator(state, act1)
            except Exception:
                next_state = state

            for i in indices1:
                if is_continuous[i] and i not in sampled_values:
                    pass  # values already consumed above

            args2_discrete = [args[u2d[i]] for i in indices2
                              if not is_continuous[i]]
            res2 = pred2.sample(rng, next_state, *args2_discrete)
            res2_iter = iter(res2)
            for i in indices2:
                if is_continuous[i] and i not in sampled_values:
                    sampled_values[i] = next(res2_iter)

            return tuple(sampled_values[i] for i in continuous_indices)

        macro_pred = Predicate(
            name,
            arity=len(var_types),
            var_types=var_types,
            is_action_pred=True,
            sampler=macro_sampler
        )
        self._macro_preds[name] = (macro_pred, (pred1, pred2), (indices1, indices2))
        return macro_pred

    def _macro_simulator(self, state, action):
        """Expand a macro action into two sequential sub-actions for simulation."""
        if action.predicate.name in self._macro_preds:
            _, (pred1, pred2), (indices1, indices2) = \
                self._macro_preds[action.predicate.name]
            args = action.variables
            act1 = pred1(*[args[i] for i in indices1])
            act2 = pred2(*[args[i] for i in indices2])
            return self._original_simulator(
                self._original_simulator(state, act1), act2)
        return self._original_simulator(state, action)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, init_state, goal, timeout):
        """Plan and expand any macro actions in the result into individual steps."""
        plan = super().plan(init_state, goal, timeout)
        if plan is None:
            return None

        expanded = []
        for action in plan:
            if action.predicate.name in self._macro_preds:
                _, (pred1, pred2), (indices1, indices2) = \
                    self._macro_preds[action.predicate.name]
                args = action.variables
                expanded.append(pred1(*[args[i] for i in indices1]))
                expanded.append(pred2(*[args[i] for i in indices2]))
            else:
                expanded.append(action)
        return expanded
