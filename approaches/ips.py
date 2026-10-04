"""Iterative Predicate Selection (IPS).

Post-learning filter that removes predicates not referenced by any learned
operator precondition or effect. Shrinks the symbolic state parsed at every
A* node, reducing both state-parsing overhead and search branching.
"""

from approaches.loft import LOFT
from utils import extract_preds_and_types_from_ops


class LOFTIPS(LOFT):
    """LOFT + Iterative Predicate Selection."""

    # Ablation variants set this to False to skip the pruning step.
    _use_ips = True

    # Predicates kept by IPS (None until IPS has run).
    _ips_preds = None

    def train(self, data):
        super().train(data)

        if self._use_ips and self._operators:
            used_preds, _ = extract_preds_and_types_from_ops(self._operators)

            self._all_state_preds = set(self._state_preds)
            original_count = len(self._state_preds)
            self._state_preds = {p for p in self._state_preds
                                 if p.name in used_preds}
            self._ips_preds = set(self._state_preds)
            new_count = len(self._state_preds)

            removed = original_count - new_count
            print(f"[IPS] Pruned predicates: {original_count} → {new_count}"
                  + (f" (removed {removed} unused)" if removed else ""))

    def plan(self, init_state, goal, timeout):
        # Predicates of the goal are always kept, even if no learned
        # operator uses them. Otherwise such a goal could never be reached.
        if self._ips_preds is not None:
            goal_preds = {lit.predicate.name for lit in goal.literals}
            self._state_preds = self._ips_preds | {
                p for p in self._all_state_preds if p.name in goal_preds}
        return super().plan(init_state, goal, timeout)
