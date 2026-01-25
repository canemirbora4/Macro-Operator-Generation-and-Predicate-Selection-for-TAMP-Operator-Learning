"""LOFT with Plan Repair.

Extends LOFTMacro to add a fallback mechanism. If macro-based planning fails
(e.g. due to sampling difficulties), it falls back to primitive-only planning.
"""

import copy
from approaches.loft_macro import LOFTMacro
from approaches import ApproachFailed, ApproachTimeout

class LOFTRepair(LOFTMacro):
    """LOFT with Plan Repair (Fallback).
    """
    def __init__(self, config, simulator, state_preds, action_preds):
        super().__init__(config, simulator, state_preds, action_preds)
        self._num_repairs = 0

    def plan(self, init_state, goal, timeout):
        """Try macro planning, fall back to primitives on failure.
        """
        # 1. Try standard LOFTMacro planning
        try:
            # Reserve some time for fallback if needed, but macros should be fast.
            # If macros work, they work instantly (ms). If they hang, we want to fail fast.
            # Heuristic: verify if we should split timeout. 
            # Given macros take <0.1s when they work, a short timeout is reasonable.
            # But the 'timeout' argument is global.
            
            # Let's try with full timeout first. If it fails quickly (skeleton exhaustion), we catch it.
            # If it hangs in sampling... BaseApproach checks timeout.
            # Strategy: Plan with Macros.
            return super().plan(init_state, goal, timeout)
            
        except (ApproachFailed, ApproachTimeout) as e:
            # 2. Repair: Fallback to primitive-only planning
            print(f"[LOFTRepair] Macro planning failed ({e}). Initiating Repair (Fallback to Primitives)...")
            self._num_repairs += 1
            
            # Temporarily filter operators to remove macros
            original_ops = self._operators
            primitive_ops = [op for op in self._operators if not op.name.startswith("Macro")]
            
            self._operators = primitive_ops
            try:
                # Plan with primitives
                # Note: super().plan() calls _skeleton_generator which uses self._operators
                # We need to call BaseApproach.plan directly or LOFTAdvancedMinimal.plan?
                # LOFTMacro.plan calls super().plan() then expands. 
                # If we use primitive ops, there are no macros to expand, so expansion loop is no-op.
                # So calling super().plan() is safe (it's LOFTMacro.plan).
                return super().plan(init_state, goal, timeout)
            finally:
                # Restore macros for next problem
                self._operators = original_ops
