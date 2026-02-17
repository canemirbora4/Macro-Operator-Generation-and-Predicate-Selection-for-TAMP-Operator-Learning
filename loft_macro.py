"""LOFT Macro (Minimal+IPS + Macro-Operators)

Variant 4 in our hierarchy - the full system:
1. LOFT (original)
2. Minimal (+ subsumption, Laplace, adaptive backtracking, early abandonment)
3. Minimal+IPS (+ predicate pruning)
4. Macro (extends 3, adds macro-operator mining)
"""

import copy
from approaches.loft_advanced_minimal import LOFTAdvancedMinimal
from structs import Predicate, Literal, WORLD

class LOFTMacro(LOFTAdvancedMinimal):
    """Macro: Minimal+IPS + automatic macro-operator mining from data.
    """
    def __init__(self, config, simulator, state_preds, action_preds):
        super().__init__(config, simulator, state_preds, action_preds)
        self._macro_preds = {}  # name -> (pred, sub_actions_info, indices_info)
        self._original_simulator = simulator
        self._simulator = self._macro_simulator

    def train(self, data):
        """Augment data with macros before training.
        """
        demos, random_data = data
        all_trajs = demos + random_data
        
        # Phase 1: Discover macro patterns and create predicates
        macro_transitions = []  # Positive examples (from demos)
        
        for trajectory in all_trajs:
            for i in range(len(trajectory) - 1): 
                trans1 = trajectory[i]
                trans2 = trajectory[i+1]
                
                act1 = trans1[1]
                act2 = trans2[1]
                
                if not hasattr(act1, 'predicate') or not hasattr(act2, 'predicate'):
                    continue
                    
                name1 = act1.predicate.name.lower()
                name2 = act2.predicate.name.lower()
                
                # Define rule: Pick+Place or Pick+Stack
                is_macro = False
                if "pick" in name1 and ("place" in name2 or "stack" in name2 or "put" in name2):
                    is_macro = True
                
                if is_macro:
                    macro_info = self._create_macro_from_actions(act1, act2)
                    if macro_info is not None:
                        macro_action, start_state, end_state = macro_info
                        start_state = trans1[0]
                        end_state = trans2[2]
                        macro_transitions.append((start_state, macro_action, end_state, []))

        print(f"[LOFTMacro] Generated {len(macro_transitions)} macro-trajectories.")
        
        # Phase 2: Generate NEGATIVE examples for macros
        # This is critical for learning preconditions!
        macro_negatives = self._generate_macro_negatives(demos, random_data)
        print(f"[LOFTMacro] Generated {len(macro_negatives)} negative macro examples.")
        
        # Phase 3: Add macro predicates to action_preds
        for name, (pred, _, _) in self._macro_preds.items():
            self._action_preds.add(pred)
        
        # Phase 4: Combine data
        # Add macro positives to demos, negatives to random_data
        augmented_demos = list(demos) + [[t] for t in macro_transitions]
        augmented_random = list(random_data) + [[t] for t in macro_negatives]
        
        new_data = (augmented_demos, augmented_random)
        
        super().train(new_data)

    def _create_macro_from_actions(self, act1, act2):
        """Create or retrieve macro predicate for action pair.
        Returns (macro_action, None, None) - states filled by caller.
        """
        # Unify arguments to avoid duplicates
        unique_args = []
        seen_map = {}  # obj -> index in unique_args
        
        indices1 = []
        indices2 = []
        
        # Process act1 args
        for arg in act1.variables:
            if arg not in seen_map:
                seen_map[arg] = len(unique_args)
                unique_args.append(arg)
            indices1.append(seen_map[arg])
            
        # Process act2 args
        for arg in act2.variables:
            if arg not in seen_map:
                seen_map[arg] = len(unique_args)
                unique_args.append(arg)
            indices2.append(seen_map[arg])
            
        # Generate signature based on indices
        sig_str = f"{tuple(indices1)}-{tuple(indices2)}"
        signature = sig_str.replace("(", "").replace(")", "").replace(", ", "_").replace(",", "_")
        
        macro_name = f"Macro-{act1.predicate.name}-{act2.predicate.name}-{signature}"
        
        if macro_name not in self._macro_preds:
            var_types = [arg.var_type for arg in unique_args]
            self._create_macro_predicate(
                macro_name, var_types, 
                act1.predicate, act2.predicate, 
                indices1, indices2
            )
            
        macro_pred, _, _ = self._macro_preds[macro_name]
        macro_action = macro_pred(*unique_args)
        
        return (macro_action, None, None)

    def _generate_macro_negatives(self, demos, random_data):
        """Generate negative examples for macros using random_data.
        
        Simple approach: For each failed primitive action in random_data,
        if it's a Pick that failed, create a macro negative with the same state.
        
        This leverages the existing negative examples (random actions that fail)
        to teach the system when macros would also fail.
        """
        negatives = []
        
        if not self._macro_preds:
            return negatives
        
        import random
        rng = random.Random(42)
        
        # Use states from random_data where Pick-like actions might fail
        for traj in random_data:
            for trans in traj:
                state = trans[0]
                action = trans[1]
                next_state = trans[2]
                
                # If state didn't change much, this might be a "failed" action
                # or we can just use random states with random macro groundings
                if not hasattr(action, 'predicate'):
                    continue
                
                # For each macro, create a negative example using this state
                for macro_name, (macro_pred, (pred1, pred2), (indices1, indices2)) in self._macro_preds.items():
                    var_types = macro_pred.var_types
                    
                    # Get objects of each required type from state
                    objs_by_type = {}
                    for obj in state:
                        if obj == WORLD or not hasattr(obj, 'var_type'):
                            continue
                        vt = obj.var_type
                        if vt not in objs_by_type:
                            objs_by_type[vt] = []
                        objs_by_type[vt].append(obj)
                    
                    # Try to construct macro with random objects
                    try:
                        macro_args = []
                        used_objs = set()  # Track used objects to avoid duplicates
                        
                        for i, vt in enumerate(var_types):
                            if vt.is_continuous:
                                # Use unique lifted variable names
                                macro_args.append(vt(f"?cont{i}"))
                            else:
                                if vt not in objs_by_type or not objs_by_type[vt]:
                                    raise ValueError("No objects of required type")
                                # Get available objects (not already used)
                                available = [o for o in objs_by_type[vt] if o not in used_objs]
                                if not available:
                                    raise ValueError("Not enough unique objects")
                                chosen = rng.choice(available)
                                macro_args.append(chosen)
                                used_objs.add(chosen)
                        
                        # Verify all args are unique (excluding continuous)
                        discrete_args = [a for a in macro_args if not a.var_type.is_continuous]
                        if len(discrete_args) != len(set(discrete_args)):
                            raise ValueError("Duplicate arguments")
                        
                        macro_action = macro_pred(*macro_args)
                        
                        # 50% chance to add as negative (to avoid too many)
                        if rng.random() < 0.5:
                            negatives.append((state, macro_action, state, []))
                            
                    except (ValueError, KeyError, IndexError):
                        continue
                
                # Limit negatives per macro
                if len(negatives) > 100 * len(self._macro_preds):
                    break
            
            if len(negatives) > 100 * len(self._macro_preds):
                break
        
        return negatives

    def _create_macro_predicate(self, name, var_types, pred1, pred2, indices1, indices2):
        """Dynamically create a new macro predicate.
        """
        # Store references for the sampler closure
        original_simulator = self._original_simulator
        
        def macro_sampler(rng, state, *args):
            """Sample continuous parameters for macro action.
            
            args contains ONLY the discrete arguments of the macro.
            Returns continuous values in the order matching var_types.
            
            IMPORTANT: Handles shared continuous args (same var used by both sub-actions).
            """
            # Identify which var_types indices are continuous
            is_continuous = [vt.is_continuous for vt in var_types]
            continuous_indices = [i for i, c in enumerate(is_continuous) if c]
            
            # Create map: unique_arg_index -> discrete_arg_index (for discrete args only)
            u2d = {}
            d_count = 0
            for i, cont in enumerate(is_continuous):
                if not cont:
                    u2d[i] = d_count
                    d_count += 1
            
            # Sample values for each unique continuous index ONCE
            # We'll use pred1's sampler for indices in indices1, pred2's for indices in indices2
            sampled_values = {}  # continuous_index -> sampled_value
            
            # Get discrete args for pred1
            args1_discrete = [args[u2d[i]] for i in indices1 if not is_continuous[i]]
            
            # Sample for pred1's continuous args
            res1 = pred1.sample(rng, state, *args1_discrete)
            res1_iter = iter(res1)
            for i in indices1:
                if is_continuous[i] and i not in sampled_values:
                    sampled_values[i] = next(res1_iter)
            
            # Reconstruct act1 to simulate for pred2 sampling
            act1_args = []
            res1_iter = iter(res1)
            args1_d_iter = iter(args1_discrete)
            for vt in pred1.var_types:
                if vt.is_continuous:
                    val = next(res1_iter)
                    act1_args.append(vt(f"sampled", val))
                else:
                    act1_args.append(next(args1_d_iter))
            
            act1 = pred1(*act1_args)
            
            # Simulate act1 to get next state
            try:
                next_state = original_simulator(state, act1)
            except Exception:
                next_state = state
            
            # Sample for pred2's continuous args (only those not already sampled)
            args2_discrete = [args[u2d[i]] for i in indices2 if not is_continuous[i]]
            res2 = pred2.sample(rng, next_state, *args2_discrete)
            res2_iter = iter(res2)
            for i in indices2:
                if is_continuous[i] and i not in sampled_values:
                    sampled_values[i] = next(res2_iter)
            
            # Return values in order of continuous_indices
            return tuple(sampled_values[i] for i in continuous_indices)

        # Create predicate
        macro_pred = Predicate(
            name, 
            arity=len(var_types), 
            var_types=var_types, 
            is_action_pred=True, 
            sampler=macro_sampler
        )
                               
        self._macro_preds[name] = (macro_pred, (pred1, pred2), (indices1, indices2))
        return macro_pred

    def _macro_simulator(self, state, action):
        """Simulate macro actions by expanding them.
        """
        if action.predicate.name in self._macro_preds:
            # It's a macro - expand and execute
            _, (pred1, pred2), (indices1, indices2) = self._macro_preds[action.predicate.name]
            
            unique_args = action.variables
            
            # Reconstruct grounded primitive actions
            args1 = [unique_args[i] for i in indices1]
            args2 = [unique_args[i] for i in indices2]
            
            act1 = pred1(*args1)
            act2 = pred2(*args2)
            
            # Simulate sequentially
            next_state = self._original_simulator(state, act1)
            final_state = self._original_simulator(next_state, act2)
            return final_state
        else:
            return self._original_simulator(state, action)

    def plan(self, init_state, goal, timeout):
        """Plan and then expand macros in the result.
        """
        # Run standard planning (may use macros)
        plan = super().plan(init_state, goal, timeout)
        
        if plan is None:
            return None
            
        # Expand macros back to primitives
        expanded_plan = []
        for action in plan:
            if action.predicate.name in self._macro_preds:
                # It's a macro - expand
                _, (pred1, pred2), (indices1, indices2) = self._macro_preds[action.predicate.name]
                
                unique_args = action.variables
                args1 = [unique_args[i] for i in indices1]
                args2 = [unique_args[i] for i in indices2]
                
                act1 = pred1(*args1)
                act2 = pred2(*args2)
                
                expanded_plan.append(act1)
                expanded_plan.append(act2)
            else:
                expanded_plan.append(action)
                
        return expanded_plan
