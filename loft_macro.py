"""LOFT with Macro-Operators.

Learns macro-actions (sequences of primitives) from data to accelerate planning.
"""

import copy
from approaches.loft_advanced_minimal import LOFTAdvancedMinimal
from structs import Predicate, Literal

class LOFTMacro(LOFTAdvancedMinimal):
    """LOFT with Macro-Operators.
    """
    def __init__(self, config, simulator, state_preds, action_preds):
        super().__init__(config, simulator, state_preds, action_preds)
        self._macro_preds = {} # name -> (pred, sub_actions_info)
        self._original_simulator = simulator
        self._simulator = self._macro_simulator


    def train(self, data):
        """Augment data with macros before training.
        """
        demos, random_data = data
        all_trajs = demos + random_data
        
        augmented_trajectories = []
        
        for trajectory in all_trajs:
            for i in range(len(trajectory) - 1): 
                trans1 = trajectory[i]
                trans2 = trajectory[i+1]
                
                act1 = trans1[1]
                act2 = trans2[1]
                
                if hasattr(act1, 'predicate'):
                     name1 = act1.predicate.name.lower()
                else: 
                     continue
                
                if hasattr(act2, 'predicate'):
                     name2 = act2.predicate.name.lower()
                else:
                     continue
                
                # Define rule: Pick+Place or Pick+Stack
                is_macro = False
                if "pick" in name1 and ("place" in name2 or "stack" in name2 or "put" in name2):
                    is_macro = True
                
                if is_macro:
                    # Unify arguments to avoid duplicates (which LOFT dislikes)
                    raw_args = act1.variables + act2.variables
                    # Find unique args and their indices
                    unique_args = []
                    seen_map = {} # obj -> index in unique_args
                    
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
                        
                    # Generate a signature based on indices to distinguishing different aliasing patterns
                    # e.g. Pick(A)+Place(A) -> 0,0 vs Pick(A)+Place(B) -> 0,1
                    sig_str = f"{tuple(indices1)}-{tuple(indices2)}"
                    # Sanitize signature for PDDL (remove parents, commas, spaces)
                    signature = sig_str.replace("(", "").replace(")", "").replace(", ", "_").replace(",", "_")
                    
                    macro_name = f"Macro-{act1.predicate.name}-{act2.predicate.name}-{signature}"
                    
                    if macro_name not in self._macro_preds:
                        # Derive types from unique_args
                        # Wait, we need types of the predicate slots, not the object instances.
                        # But we don't have the lifted variables here.
                        # However, unique_args are TypedEntity objects, so they have .var_type
                        var_types = [arg.var_type for arg in unique_args]
                        
                        self._create_macro_predicate(macro_name, var_types, act1.predicate, act2.predicate, indices1, indices2)
                        
                    macro_pred, _, _ = self._macro_preds[macro_name]
                    
                    # Create the combined grounded action
                    macro_action = macro_pred(*unique_args)
                    
                    # Create transition (s_t, Macro, s_{t+2})
                    start_state = trans1[0]
                    end_state = trans2[2]
                    
                    # Helper: extract transition data expects (s, a, ns, info)
                    augmented_trajectories.append([(start_state, macro_action, end_state, [])])

        print(f"[LOFTMacro] Generated {len(augmented_trajectories)} macro-trajectories.")
        
        # Combine data
        new_demos = demos + augmented_trajectories
        new_data = (new_demos, random_data)
        
        super().train(new_data)

    def _create_macro_predicate(self, name, var_types, pred1, pred2, indices1, indices2):
        """Dynamically create a new macro predicate.
        """
        # Define combined sampler
        def macro_sampler(rng, state, *args):
            # args contains ONLY the discrete arguments of the macro
            # We need to map from unique_args indices to these discrete args
            
            # Identify which unique_args indices are discrete
            is_discrete = [not vt.is_continuous for vt in var_types]
            
            # Create map: unique_arg_index -> discrete_arg_index
            u2d = {}
            d_count = 0
            for i, discrete in enumerate(is_discrete):
                if discrete:
                    u2d[i] = d_count
                    d_count += 1
            
            # Reconstruct discrete args for sub-actions
            # Filter indices1 to only discrete args, then map to input args
            args1_discrete = [args[u2d[i]] for i in indices1 if is_discrete[i]]
            args2_discrete = [args[u2d[i]] for i in indices2 if is_discrete[i]]
            
            res1 = pred1.sample(rng, state, *args1_discrete)
            
            # Reconstruct act1 to simulate next state
            # We need to interleave discrete and continuous args for act1
            # Note: PDDLGym separates them in logic usually, but here we need to follow var_types order
            start_d = 0
            start_c = 0
            act1_args = []
            for vt in pred1.var_types:
                if vt.is_continuous:
                    act1_args.append(res1[start_c])
                    start_c += 1
                else:
                    act1_args.append(args1_discrete[start_d])
                    start_d += 1
            
            act1 = pred1(*act1_args)
            
            # Simulate act1 to get next state for sampling act2
            try:
                next_state = self._original_simulator(state, act1)
            except Exception:
                # If simulation fails (e.g. precondition), fallback to current state 
                # or just let pred2 sample from current state (best effort)
                next_state = state
            
            res2 = pred2.sample(rng, next_state, *args2_discrete)
            
            return res1 + res2

        # Create predicate
        macro_pred = Predicate(name, 
                               arity=len(var_types), 
                               var_types=var_types, 
                               is_action_pred=True, 
                               sampler=macro_sampler)
                               
        self._macro_preds[name] = (macro_pred, (pred1, pred2), (indices1, indices2))
        return macro_pred

    def _macro_simulator(self, state, action):
        """Simulate macro actions by expanding them.
        """
        if action.predicate.name in self._macro_preds:
            # It's a macro
            _, (pred1, pred2), (indices1, indices2) = self._macro_preds[action.predicate.name]
            
            # Action uses unique args
            unique_args = action.variables
            
            # Reconstruct grounded primitive actions using stored indices
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
        """Plan and then expand macros.
        """
        # Run standard planning
        plan = super().plan(init_state, goal, timeout)
        
        if plan is None:
            return None
            
        # Expand macros
        expanded_plan = []
        for action in plan:
            if action.predicate.name in self._macro_preds:
                # It's a macro
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
