"""Subtractive ablation variants.

Full model = LOFTMacro (safe refinements + IPS + macro-operator mining).
Each variant removes exactly one component to isolate its contribution.

  FullMinusSubsumption:  skip subsumption filtering in determinization
  FullMinusLaplace:      use raw MLE instead of Laplace smoothing
  FullMinusBacktracking: use LOFT's default uniform backtracking
  FullMinusIPS:          skip predicate pruning (IPS)
  FullMinusMacro:        LOFTIPS with no macro mining (= full minus macros)
"""

from approaches.loft_macro import LOFTMacro
from approaches.ips import LOFTIPS
from approaches.safe_refinements import LOFTSafeRefinements
from approaches.loft import LOFT


class FullMinusSubsumption(LOFTMacro):
    """Full model without subsumption filtering."""

    def _determinize_ndrs(self, ndr_set):
        return LOFT._determinize_ndrs(self, ndr_set)


class FullMinusLaplace(LOFTMacro):
    """Full model without Laplace smoothing (raw MLE)."""

    @staticmethod
    def _recover_single_ndr_probabilities(pre_transitions, outcomes,
                                          pre, lifted_action):
        return LOFT._recover_single_ndr_probabilities(
            pre_transitions, outcomes, pre, lifted_action)


class FullMinusBacktracking(LOFTMacro):
    """Full model without adaptive backtracking (uniform sampling)."""

    def _sample_continuous_values(self, init_state, goal, skeleton,
                                  expected_lits_sequence, rng,
                                  start_time, timeout):
        return LOFT._sample_continuous_values(
            self, init_state, goal, skeleton,
            expected_lits_sequence, rng, start_time, timeout)


class FullMinusIPS(LOFTMacro):
    """Full model without Iterative Predicate Selection."""

    def train(self, data):
        demos, random_data = data
        all_trajs = demos + random_data

        # Repeat macro mining from LOFTMacro but skip the IPS pruning step.
        from collections import Counter, defaultdict
        from utils import construct_effects

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

        typical_adds = {}
        typical_dels = {}
        for act_name in action_count:
            n = action_count[act_name]
            typical_adds[act_name] = {
                p for p, c in action_adds[act_name].items() if c / n > 0.5}
            typical_dels[act_name] = {
                p for p, c in action_dels[act_name].items() if c / n > 0.5}

        causal_pairs = set()
        for a1 in action_count:
            for a2 in action_count:
                if a1 == a2:
                    continue
                overlap = typical_adds[a1] & typical_dels[a2]
                if any(pred_arity.get(p, 0) > 0 for p in overlap):
                    causal_pairs.add((a1, a2))

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

        to_remove = set()
        for (a1, a2) in validated:
            if (a2, a1) in validated and (a2, a1) not in to_remove:
                if pair_freq[(a1, a2)] >= pair_freq[(a2, a1)]:
                    to_remove.add((a2, a1))
                else:
                    to_remove.add((a1, a2))
        validated -= to_remove

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
                if (act1.predicate.name, act2.predicate.name) in validated:
                    macro_info = self._create_macro_from_actions(act1, act2)
                    if macro_info is not None:
                        macro_action, _, _ = macro_info
                        macro_transitions.append(
                            (trans1[0], macro_action, trans2[2], []))

        macro_negatives = self._generate_macro_negatives(demos, random_data)

        for name, (pred, _, _) in self._macro_preds.items():
            self._action_preds.add(pred)

        augmented_demos = list(demos) + [[t] for t in macro_transitions]
        augmented_random = list(random_data) + [[t] for t in macro_negatives]

        # Use LOFTSafeRefinements.train to skip IPS pruning.
        LOFTSafeRefinements.train(self, (augmented_demos, augmented_random))


# Full minus Macro = just LOFTIPS (no macro mining at all)
FullMinusMacro = LOFTIPS
