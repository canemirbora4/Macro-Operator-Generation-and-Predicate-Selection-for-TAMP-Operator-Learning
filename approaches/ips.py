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

    def train(self, data):
        super().train(data)

        if self._use_ips and self._operators:
            used_preds, _ = extract_preds_and_types_from_ops(self._operators)

            original_count = len(self._state_preds)
            self._state_preds = {p for p in self._state_preds
                                 if p.name in used_preds}
            new_count = len(self._state_preds)

            removed = original_count - new_count
            print(f"[IPS] Pruned predicates: {original_count} → {new_count}"
                  + (f" (removed {removed} unused)" if removed else ""))
