"""LOFT-Advanced-Minimal

Improvements:
1. Iterative Predicate Selection (IPS): Automatically prunes unused predicates.
2. Heuristic: Retains hAdd (standard) for robustness in hybrid domains.
3. Minimal fixes: Inherits subsumption filtering, Laplace smoothing, etc.
"""

from approaches.loft_minimal import LOFTMinimal
from utils import extract_preds_and_types_from_ops

class LOFTAdvancedMinimal(LOFTMinimal):
    """LOFT with IPS but standard hAdd heuristic.
    """
    
    def train(self, data):
        """Train logic with Iterative Predicate Selection (IPS).
        """
        # 1. Train normally (using Minimal's robust training)
        super().train(data)
        
        # 2. Identify used predicates
        if self._operators:
            used_preds, _ = extract_preds_and_types_from_ops(self._operators)
            
            # 3. Prune unused predicates from self._state_preds
            original_count = len(self._state_preds)
            self._state_preds = {p for p in self._state_preds if p.name in used_preds}
            new_count = len(self._state_preds)
            
            print(f"[LOFTAdvancedMinimal] IPS Pruning: {original_count} -> {new_count} predicates.")
            removed = original_count - new_count
            if removed > 0:
                print(f"[LOFTAdvancedMinimal] Removed {removed} unused predicates.")
