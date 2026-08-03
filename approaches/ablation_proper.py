"""Subtractive ablation variants.

Full model = LOFTMacro (macro-operator generation + IPS).
Each variant removes exactly one component to isolate its contribution.

  FullMinusIPS:   skip predicate pruning (IPS)
  FullMinusMacro: LOFTIPS with no macro generation (= full minus macros)
"""

from approaches.loft_macro import LOFTMacro
from approaches.ips import LOFTIPS


class FullMinusIPS(LOFTMacro):
    """Full model without Iterative Predicate Selection."""
    _use_ips = False


# Full minus Macro = just LOFTIPS (no macro generation at all)
FullMinusMacro = LOFTIPS
