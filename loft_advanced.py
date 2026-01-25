"""LOFT-Advanced

Improvements:
1. Iterative Predicate Selection (IPS): Automatically prunes unused predicates.
2. Heuristic Upgrade: Uses hFF (Fast-Forward) instead of hAdd.
"""

from approaches.loft_minimal import LOFTMinimal
from utils import PyperplanHFFHeuristic, extract_preds_and_types_from_ops

class LOFTAdvanced(LOFTMinimal):
    """LOFT with IPS and hFF heuristic.
    """
    
    def train(self, data):
        """Train logic with Iterative Predicate Selection (IPS).
        """
        # 1. Train normally
        super().train(data)
        
        # 2. Identify used predicates
        if self._operators:
            used_preds, _ = extract_preds_and_types_from_ops(self._operators)
            
            # 3. Prune unused predicates from self._state_preds
            # This ensures that during planning, we only parse/track/search 
            # with predicates that actually matter.
            original_count = len(self._state_preds)
            self._state_preds = {p for p in self._state_preds if p.name in used_preds}
            new_count = len(self._state_preds)
            
            print(f"[LOFTAdvanced] IPS Pruning: {original_count} -> {new_count} predicates.")
            removed = original_count - new_count
            if removed > 0:
                print(f"[LOFTAdvanced] Removed {removed} unused predicates.")

    def _create_heuristic(self, discrete_objs, goal):
        """Use hFF instead of hAdd.
        """
        return PyperplanHFFHeuristic(self._operators, discrete_objs, goal)
