"""LOFT-Minimal

Improvements over original LOFT:
1. Subsumption filtering: Removes redundant operators where one effect 
   set is a proper subset of another for the same preconditions.
2. Laplace smoothing: More robust probability estimation with sparse data.
3. Early skeleton abandonment: Skip bad skeletons faster in complex domains.
4. Adaptive backtracking: Position-based sample allocation (fewer early, more late).

"""

import time
from collections import defaultdict
from approaches.loft import LOFT
from utils import EnvironmentFailure


class LOFTMinimal(LOFT):
    """LOFT with safe, minimal improvements.
    
    1. Subsumption filtering (Cover: 4→3 operators)
    2. Laplace smoothing for probability estimation
    3. Early skeleton abandonment
    4. Adaptive backtracking (Blocks planning -13%)
    """
    
    def _sample_continuous_values(self, init_state, goal, skeleton,
                                  expected_lits_sequence, rng,
                                  start_time, timeout):
        """Override with early abandonment and adaptive backtracking."""
        cur_idx = 0
        num_tries = [0 for _ in skeleton]
        
        # Adaptive Backtracking v2: Position-based sample allocation
        # - Early steps: fewer samples (wrong skeleton = waste)
        # - Later steps: more samples (invested effort, try harder)
        base_samples = self._cf.backtracking_num_samples_per_step
        idx_to_max_num_tries = []
        for i, a in enumerate(skeleton):
            if not any(v.is_continuous for v in a.variables):
                idx_to_max_num_tries.append(1)
            else:
                # Position factor: 0.7x for first steps, 1.3x for later steps
                progress = i / max(1, len(skeleton) - 1) if len(skeleton) > 1 else 0.5
                factor = 0.7 + 0.6 * progress  # 0.7 → 1.3
                idx_to_max_num_tries.append(max(3, int(base_samples * factor)))
        
        plan = [None for _ in skeleton]
        traj = [init_state] + [None for _ in skeleton]
        
        # Early abandonment: if first step fails 5+ times, likely bad skeleton
        early_fail_limit = 5
        
        while cur_idx < len(skeleton):
            if time.time() - start_time > timeout:
                from approaches.base_approach import ApproachTimeout
                raise ApproachTimeout("Timed out in backtracking!")
            
            # Early abandonment: first step repeatedly failing = bad skeleton
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
            
            # Check literal sequence constraint
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
            
            # Backtracking
            cur_idx -= 1
            while num_tries[cur_idx] == idx_to_max_num_tries[cur_idx]:
                num_tries[cur_idx] = 0
                plan[cur_idx] = None
                traj[cur_idx + 1] = None
                cur_idx -= 1
                if cur_idx < 0:
                    return None
        
        # Empty skeleton case
        assert not skeleton
        if goal.holds(init_state):
            return []
        return None
    
    @staticmethod
    def _recover_single_ndr_probabilities(pre_transitions, outcomes,
                                          pre, lifted_action):
        """Override with Laplace smoothing to prevent division by zero
        and improve estimates with sparse data.
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
            # Laplace smoothing: (x + 1) / (n + 2)
            # Prevents division by zero and gives more conservative estimates
            probs.append((num_covered + 1) / (num_covered + num_not_covered + 2))
        return probs
    
    def _determinize_ndrs(self, ndr_set):
        """Override to add subsumption filtering after determinization."""
        new_ndr_set = super()._determinize_ndrs(ndr_set)
        filtered_ndrs = self._filter_subsumed(new_ndr_set.ndrs)
        
        from ndr.ndrs import NDRSet
        return NDRSet(
            ndr_set.action, 
            filtered_ndrs, 
            default_ndr=new_ndr_set.default_ndr
        )
    
    def _filter_subsumed(self, ndrs):
        """Remove NDRs where one effect set is a proper subset of another.
        
        If two operators have identical preconditions, and one's effects
        are a proper subset of the other's, the subset one is redundant.
        """
        if len(ndrs) <= 1:
            return ndrs
        
        # Group by preconditions
        precond_to_ndrs = defaultdict(list)
        for ndr in ndrs:
            precond_key = frozenset(ndr.preconditions)
            precond_to_ndrs[precond_key].append(ndr)
        
        filtered_ndrs = []
        
        for precond_key, group in precond_to_ndrs.items():
            if len(group) == 1:
                filtered_ndrs.extend(group)
                continue
            
            # Check for subsumption within this group
            keep = [True] * len(group)
            
            for i, ndr_i in enumerate(group):
                if not keep[i]:
                    continue
                
                # After determinization, main effect is in effects[0]
                effects_i = frozenset(ndr_i.effects[0])
                
                for j, ndr_j in enumerate(group):
                    if i == j or not keep[j]:
                        continue
                    
                    effects_j = frozenset(ndr_j.effects[0])
                    
                    # If effects_i is a proper subset of effects_j, remove i
                    if effects_i < effects_j:
                        keep[i] = False
                        break
            
            # Keep only non-subsumed NDRs
            for i, ndr in enumerate(group):
                if keep[i]:
                    filtered_ndrs.append(ndr)
        
        return filtered_ndrs
        