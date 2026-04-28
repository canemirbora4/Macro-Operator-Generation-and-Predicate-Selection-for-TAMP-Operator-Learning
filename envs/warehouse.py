"""Warehouse environment.

A simple domain designed to showcase macro-operator strength.
Robot must move packages from table to shelf: Pick → Place for each package.
With many packages, macros dramatically reduce planning horizon.
"""

import numpy as np
import structs
from structs import WORLD
from envs import BaseEnv


class Warehouse(BaseEnv):
    """Warehouse environment.
    
    With N packages:
    - Without macros: 2*N action steps (Pick, Place for each)
    - With Pick-Place macro: N macro steps
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Types
        self.pkg_type = structs.Type("pkg")
        self.pose_type = structs.ContinuousType("pose")
        self.pose_type.set_sampler(lambda rng: rng.uniform(0, 1))
        
        # State Predicates
        self.HandEmpty = structs.Predicate(
            "HandEmpty", 0, is_action_pred=False,
            holds=self._HandEmpty_holds, var_types=[])
        
        self.Holding = structs.Predicate(
            "Holding", 1, is_action_pred=False,
            holds=self._Holding_holds, var_types=[self.pkg_type])
        
        self.OnTable = structs.Predicate(
            "OnTable", 1, is_action_pred=False,
            holds=self._OnTable_holds, var_types=[self.pkg_type])
        
        self.OnShelf = structs.Predicate(
            "OnShelf", 1, is_action_pred=False,
            holds=self._OnShelf_holds, var_types=[self.pkg_type])
        
        # Extra predicates for IPS testing (may be unused)
        self.IsHeavy = structs.Predicate(
            "IsHeavy", 1, is_action_pred=False,
            holds=self._IsHeavy_holds, var_types=[self.pkg_type])
        
        self.IsFragile = structs.Predicate(
            "IsFragile", 1, is_action_pred=False,
            holds=self._IsFragile_holds, var_types=[self.pkg_type])
        
        # Action Predicates
        self.Pick = structs.Predicate(
            "Pick", 2, is_action_pred=True,
            var_types=[self.pkg_type, self.pose_type],
            sampler=self._pick_sampler)
        
        self.Place = structs.Predicate(
            "Place", 1, is_action_pred=True,
            var_types=[self.pose_type],
            sampler=self._place_sampler)
        
        self._packages = []
    
    def get_state_predicates(self):
        return {
            self.HandEmpty, self.Holding, self.OnTable, self.OnShelf,
            self.IsHeavy, self.IsFragile  # Extra for IPS
        }
    
    def get_action_predicates(self):
        return {self.Pick, self.Place}
    
    def _get_demo_problems(self, num):
        return self._get_problems(
            num=num, 
            num_packages_options=self._cf.warehouse_demo_num_pkgs
        )
    
    def get_test_problems(self):
        return self._get_problems(
            num=self._cf.warehouse_num_test_problems,
            num_packages_options=self._cf.warehouse_test_num_pkgs
        )
    
    def _get_problems(self, num, num_packages_options):
        problems = []
        for i in range(num):
            num_pkgs = num_packages_options[i % len(num_packages_options)]
            self._packages = [self.pkg_type(f"pkg{j}") for j in range(num_pkgs)]
            init_state = self._create_initial_state()
            goal = structs.LiteralConjunction(
                [self.OnShelf(pkg) for pkg in self._packages]
            )
            problems.append((init_state, goal))
        return problems
    
    def _create_initial_state(self):
        state = {}
        
        for i, pkg in enumerate(self._packages):
            pkg_state = {}
            pkg_state["on_table"] = True
            pkg_state["on_shelf"] = False
            pkg_state["holding"] = False
            pkg_state["pose"] = self._rng.uniform(0.1, 0.9)
            pkg_state["heavy"] = self._rng.random() > 0.5
            pkg_state["fragile"] = self._rng.random() > 0.5
            state[pkg] = pkg_state
        
        world_state = {
            "holding": None,
            "flat": np.array([0.0]),
            "flat_names": np.array(["holding_idx"])
        }
        state[WORLD] = world_state
        
        return state
    
    def get_next_state(self, state, action):
        next_state = self._copy_state(state)
        
        if action.predicate == self.Pick:
            pkg = action.variables[0]
            if pkg in state and state[pkg]["on_table"] and state[WORLD]["holding"] is None:
                next_state[pkg]["on_table"] = False
                next_state[pkg]["holding"] = True
                next_state[WORLD]["holding"] = pkg
        
        elif action.predicate == self.Place:
            held_pkg = state[WORLD]["holding"]
            if held_pkg is not None:
                next_state[held_pkg]["holding"] = False
                next_state[held_pkg]["on_shelf"] = True
                next_state[WORLD]["holding"] = None
        
        return next_state
    
    # Predicate implementations
    @staticmethod
    def _HandEmpty_holds(state):
        return state[WORLD]["holding"] is None
    
    @staticmethod
    def _Holding_holds(state, pkg):
        return pkg in state and state[pkg]["holding"]
    
    @staticmethod
    def _OnTable_holds(state, pkg):
        return pkg in state and state[pkg]["on_table"]
    
    @staticmethod
    def _OnShelf_holds(state, pkg):
        return pkg in state and state[pkg]["on_shelf"]
    
    @staticmethod
    def _IsHeavy_holds(state, pkg):
        return pkg in state and state[pkg]["heavy"]
    
    @staticmethod
    def _IsFragile_holds(state, pkg):
        return pkg in state and state[pkg]["fragile"]
    
    # Samplers
    @staticmethod
    def _pick_sampler(rng, state, pkg):
        if pkg.var_type != "pkg" or pkg not in state:
            return (0.5,)
        return (state[pkg]["pose"] + rng.uniform(-0.1, 0.1),)
    
    @staticmethod
    def _place_sampler(rng, state):
        return (rng.uniform(0.3, 0.7),)
    
    def get_random_action(self, state):
        packages = [obj for obj in state if obj != WORLD and hasattr(obj, 'var_type') and obj.var_type == self.pkg_type]
        
        actions = []
        for pkg in packages:
            if state[pkg]["on_table"]:
                actions.append((self.Pick, [pkg]))
        
        if state[WORLD]["holding"] is not None:
            actions.append((self.Place, []))
        
        if not actions:
            if packages:
                return self._sample_ground_act(state, self.Pick, [packages[0]])
            raise Exception("No valid actions")
        
        pred, discrete_args = actions[self._rng.choice(len(actions))]
        return self._sample_ground_act(state, pred, discrete_args)
