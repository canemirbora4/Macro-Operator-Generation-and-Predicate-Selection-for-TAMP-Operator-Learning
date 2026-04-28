"""Safe Refinements for LOFT.

Three improvements over the LOFT baseline, each addressing a different
instability in operator learning or plan refinement:

1. Subsumption filtering: removes redundant operators where one effect
   set is a proper subset of another for the same preconditions.
2. Laplace smoothing: more robust probability estimation under sparse data.
3. Adaptive backtracking with early abandonment: position-based sample
   allocation (fewer attempts on early steps, more on later) plus early
   termination when the first plan step fails repeatedly.
"""

import time
from collections import defaultdict
from approaches.loft import LOFT
from utils import EnvironmentFailure


class LOFTSafeRefinements(LOFT):
    """LOFT + three safe refinements: subsumption filtering, Laplace
    smoothing, and adaptive backtracking with early abandonment.
    """

    def _sample_continuous_values(self, init_state, goal, skeleton,
                                  expected_lits_sequence, rng,
                                  start_time, timeout):
        """Adaptive backtracking: position-based sample budget + early
        abandonment when the first step fails K_early consecutive times.
        """
        cur_idx = 0
        num_tries = [0 for _ in skeleton]

        base_samples = self._cf.backtracking_num_samples_per_step
        idx_to_max_num_tries = []
        for i, a in enumerate(skeleton):
            if not any(v.is_continuous for v in a.variables):
                idx_to_max_num_tries.append(1)
            else:
                # Scale factor: 0.7x for first step → 1.3x for last step.
                progress = i / max(1, len(skeleton) - 1) if len(skeleton) > 1 else 0.5
                factor = 0.7 + 0.6 * progress
                idx_to_max_num_tries.append(max(3, int(base_samples * factor)))

        plan = [None for _ in skeleton]
        traj = [init_state] + [None for _ in skeleton]

        # Early abandonment: first step failing K_early times → bad skeleton.
        early_fail_limit = 5

        while cur_idx < len(skeleton):
            if time.time() - start_time > timeout:
                from approaches.base_approach import ApproachTimeout
                raise ApproachTimeout("Timed out in backtracking!")

            if cur_idx == 0 and num_tries[0] >= early_fail_limit:
                return None

            assert num_tries[cur_idx] < idx_to_max_num_tries[cur_idx]
            num_tries[cur_idx] += 1
            state = traj[cur_idx]
            skel_act = skeleton[cur_idx]
            act_args = self._sample_act_args(state, skel_act, rng)
            ground_act = skel_act.predicate(*act_args)
            plan[cur_idx] = ground_act

            try:
                traj[cur_idx + 1] = self._simulator(state, ground_act)
            except EnvironmentFailure:
                return None

            cur_idx += 1

            if expected_lits_sequence is None:
                if cur_idx == len(skeleton) and goal.holds(traj[cur_idx]):
                    return plan
                if cur_idx < len(skeleton):
                    continue
            else:
                assert len(traj) == len(expected_lits_sequence)
                assert len(skeleton) == len(expected_lits_sequence) - 1
                for lit in expected_lits_sequence[cur_idx]:
                    if not lit.holds(traj[cur_idx]):
                        break
                else:
                    if cur_idx == len(skeleton) and goal.holds(traj[cur_idx]):
                        return plan
                    if cur_idx < len(skeleton):
                        continue

            cur_idx -= 1
            while num_tries[cur_idx] == idx_to_max_num_tries[cur_idx]:
                num_tries[cur_idx] = 0
                plan[cur_idx] = None
                traj[cur_idx + 1] = None
                cur_idx -= 1
                if cur_idx < 0:
                    return None

        assert not skeleton
        if goal.holds(init_state):
            return []
        return None

    @staticmethod
    def _recover_single_ndr_probabilities(pre_transitions, outcomes,
                                          pre, lifted_action):
        """Laplace (add-one) smoothing: keeps rare but valid effects from
        receiving zero probability.
        """
        from utils import transition_covered

        probs = []
        for outcome in outcomes:
            num_covered = 0
            num_not_covered = 0
            for transition in pre_transitions:
                covered, assigns = transition_covered(
                    transition, pre, lifted_action, outcome,
                    ret_assignments=True)
                if not covered or len(assigns) > 1:
                    num_not_covered += 1
                else:
                    num_covered += 1
            # p_k = (n_k + 1) / (N + 2)
            probs.append((num_covered + 1) / (num_covered + num_not_covered + 2))
        return probs

    def _determinize_ndrs(self, ndr_set):
        """Subsumption filtering applied after all-outcome determinization."""
        new_ndr_set = super()._determinize_ndrs(ndr_set)
        filtered_ndrs = self._filter_subsumed(new_ndr_set.ndrs)

        from ndr.ndrs import NDRSet
        return NDRSet(
            ndr_set.action,
            filtered_ndrs,
            default_ndr=new_ndr_set.default_ndr
        )

    def _filter_subsumed(self, ndrs):
        """Remove NDRs whose effect set is a proper subset of another NDR's
        effect set when both share identical preconditions.
        """
        if len(ndrs) <= 1:
            return ndrs

        precond_to_ndrs = defaultdict(list)
        for ndr in ndrs:
            precond_key = frozenset(ndr.preconditions)
            precond_to_ndrs[precond_key].append(ndr)

        filtered_ndrs = []
        for precond_key, group in precond_to_ndrs.items():
            if len(group) == 1:
                filtered_ndrs.extend(group)
                continue

            keep = [True] * len(group)
            for i, ndr_i in enumerate(group):
                if not keep[i]:
                    continue
                effects_i = frozenset(ndr_i.effects[0])
                for j, ndr_j in enumerate(group):
                    if i == j or not keep[j]:
                        continue
                    effects_j = frozenset(ndr_j.effects[0])
                    if effects_i < effects_j:
                        keep[i] = False
                        break

            for i, ndr in enumerate(group):
                if keep[i]:
                    filtered_ndrs.append(ndr)

        return filtered_ndrs
