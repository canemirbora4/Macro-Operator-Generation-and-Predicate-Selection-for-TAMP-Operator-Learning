# LOFT Extensions: Variants and Improvements

This repository contains extended implementations of the **LOFT** (Learning Symbolic Operators for Task and Motion Planning) approach. It builds upon the original codebase (from Silver et al., IROS 2021) by introducing mechanisms for automatic **Macro-Operator Mining**, **Plan Repair**, and several algorithmic refinements.

The original implementation can be found in the [original repository](https://github.com/ronuchit/LOFT_IROS_2021).

## Key Differences & Contributions

While the original LOFT approach learns symbolic operators from data, this implementation adds several variants to improve planning efficiency, robustness, and learning quality:

### 1. Macro-Operator Mining (`LOFTMacro`)
*Implemented in `approaches/loft_macro.py`*

This variant automatically discovers useful **macro-operators**—sequences of primitive actions that frequently occur together—from the demonstration data.

*   **How it works**: The system scans training trajectories for recurring patterns, such as a `Pick` action followed immediately by a `Place`.
*   **Significance**: By treating these sequences as single "Macro" operators, the planner can take larger steps in the search space, significantly reducing the effective horizon and planning time.
*   **Implementation**: New macro predicates are dynamically created and registered during the training phase.

### 2. Plan Repair / Fallback (`LOFTRepair`)
*Implemented in `approaches/loft_repair.py`*

This approach functions as a wrapper around the Macro-based planner to ensure robustness.

*   **The Problem**: Macro-operators can occasionally fail if specific conditions aren't met or if sampling continuous parameters becomes difficult.
*   **The Solution**: `LOFTRepair` attempts to plan using accelerated macro-operators first. If that attempt fails or times out, it automatically falls back to standard primitive-based operators.
*   **Benefit**: This ensures "safe" acceleration—it performs no worse than the baseline in terms of solvability but gains speed whenever macros are applicable.

### 3. LOFT Minimal (`LOFTMinimal`)
*Implemented in `approaches/loft_minimal.py`*

A refined version of LOFT containing "safe" algorithmic improvements that generally enhance performance without changing the core logic.

*   **Subsumption Filtering**: Removes redundant operators where one operator's effects are a subset of another's for the same preconditions. This reduces the branching factor.
*   **Laplace Smoothing**: Uses `(x + 1) / (n + 2)` for probability estimation, providing more robust estimates when data is sparse.
*   **Adaptive Backtracking & Early Abandonment**: Optimizes the search by allocating fewer samples to early steps (quickly discarding bad skeletons) and more samples to later steps, preventing wasted effort.

### 4. LOFT Advanced (`LOFTAdvanced`)
*Implemented in `approaches/loft_advanced.py`*

Builds on `LOFTMinimal` with more aggressive optimizations.

*   **Iterative Predicate Selection (IPS)**: Automatically identifies and prunes predicates that are never used in learned operators. This significantly reduces the state space and overhead during planning.
*   **Heuristic Upgrade (hFF)**: Replaces the standard `hAdd` heuristic with the Fast-Forward (`hFF`) heuristic, which is often more informed for satisfying planning.

### 5. LOFT Advanced Minimal (`LOFTAdvancedMinimal`)
*Implemented in `approaches/loft_advanced_minimal.py`*

A hybrid approach that combines the benefits of IPS with the robustness of the standard heuristic.

*   **IPS Enabled**: Prunes unused predicates to speed up state operations.
*   **Standard Heuristic (hAdd)**: Retains the `hAdd` heuristic instead of `hFF`. This is sometimes preferred in hybrid domains where `hFF` might be misleading or too computationally expensive to compute for every state.

## Repository Structure

*   `approaches/loft_macro.py`: Macro-mining logic.
*   `approaches/loft_repair.py`: Fallback logic (Macro -> Primitive).
*   `approaches/loft_minimal.py`: Algorithmic refinements (Subsumption, Smoothing).
*   `approaches/loft_advanced.py`: IPS and hFF heuristic.


To work and test these new approaches, they must be added to the original repo.
