"""Kitchen environment.
"""

import numpy as np
import pybullet as p
import structs
from structs import WORLD
from envs import BaseEnv
from envs.pybullet_utils import get_kinematic_chain, inverse_kinematics, \
    get_asset_path, aabb_overlap


class Kitchen(BaseEnv):
    """Kitchen environment.
    """
    table_height = 0.4   # table2.urdf top surface
    table_center_x = 1.65
    table_center_y = 0.75
    # Areas on table (URDF places table at 1.65, 0.75)
    burner_pos = [1.65, 0.4, 0.4]   # Red
    board_pos = [1.65, 0.6, 0.4]    # Brown
    dish_pos = [1.65, 0.8, 0.4]     # Green
    area_radius = 0.08

    # Table bounds for random placement
    table_lb = -0.5
    table_ub = 1.5

    obj_height = 0.05
    obj_radius = 0.03
    robot_x = 0.8
    robot_z = -0.25

    # Ingredient visuals: shape + color so objects look like different foods (labels in English)
    INGREDIENT_VISUALS = [
        {"shape": "sphere", "color": (0.9, 0.25, 0.2, 1), "label": "Tomato"},
        {"shape": "cylinder", "color": (0.95, 0.55, 0.15, 1), "label": "Carrot"},
        {"shape": "box", "color": (0.25, 0.7, 0.3, 1), "label": "Pepper"},
        {"shape": "sphere", "color": (0.95, 0.85, 0.2, 1), "label": "Lemon"},
        {"shape": "box", "color": (0.55, 0.35, 0.2, 1), "label": "Potato"},
        {"shape": "cylinder", "color": (0.75, 0.35, 0.75, 1), "label": "Eggplant"},
        {"shape": "sphere", "color": (0.2, 0.6, 0.85, 1), "label": "Onion"},
        {"shape": "box", "color": (0.9, 0.6, 0.4, 1), "label": "Cheese"},
    ]

    def __init__(self, config):
        super().__init__(config)
        # Types
        self.obj_type = structs.Type("ingredient")
        self.pose_type = structs.ContinuousType("pose")
        self.pose_type.set_sampler(lambda rng: rng.uniform(-10, 10, size=3))

        # Predicates
        self.OnTable = structs.Predicate(
            "OnTable", 1, is_action_pred=False,
            holds=self._OnTable_holds, var_types=[self.obj_type])
        self.OnBurner = structs.Predicate(
            "OnBurner", 1, is_action_pred=False,
            holds=self._OnBurner_holds, var_types=[self.obj_type])
        self.OnBoard = structs.Predicate(
            "OnBoard", 1, is_action_pred=False,
            holds=self._OnBoard_holds, var_types=[self.obj_type])
        self.InDish = structs.Predicate(
            "InDish", 1, is_action_pred=False,
            holds=self._InDish_holds, var_types=[self.obj_type])
        
        self.IsChopped = structs.Predicate(
            "IsChopped", 1, is_action_pred=False,
            holds=self._IsChopped_holds, var_types=[self.obj_type])
        self.IsCooked = structs.Predicate(
            "IsCooked", 1, is_action_pred=False,
            holds=self._IsCooked_holds, var_types=[self.obj_type])
            
        self.Holding = structs.Predicate(
            "Holding", 1, is_action_pred=False,
            holds=self._Holding_holds, var_types=[self.obj_type])
        self.HandEmpty = structs.Predicate(
            "HandEmpty", 0, is_action_pred=False,
            holds=self._HandEmpty_holds, var_types=[])

        self.IsTablePose = structs.Predicate(
            "IsTablePose", 1, is_action_pred=False,
            holds=self._IsTablePose_holds,
            get_satisfying_args=self._IsTablePose_get_satisfying_args,
            var_types=[self.pose_type])

        # Action predicates
        self.Pick = structs.Predicate(
            "Pick", 1, is_action_pred=True,
            sampler=self._pick_sampler,
            var_types=[self.obj_type])
        self.PlaceOnBurner = structs.Predicate(
            "PlaceOnBurner", 0, is_action_pred=True,
            sampler=self._place_on_burner_sampler,
            var_types=[])
        self.PlaceOnBoard = structs.Predicate(
            "PlaceOnBoard", 0, is_action_pred=True,
            sampler=self._place_on_board_sampler,
            var_types=[])
        self.PlaceInDish = structs.Predicate(
            "PlaceInDish", 0, is_action_pred=True,
            sampler=self._place_in_dish_sampler,
            var_types=[])
        self.PlaceOnTable = structs.Predicate(
            "PlaceOnTable", 0, is_action_pred=True,
            sampler=self._place_on_table_sampler,
            var_types=[])
        self.Chop = structs.Predicate(
            "Chop", 1, is_action_pred=True,
            sampler=self._chop_sampler,
            var_types=[self.obj_type])
        self.Cook = structs.Predicate(
            "Cook", 1, is_action_pred=True,
            sampler=self._cook_sampler,
            var_types=[self.obj_type])

        self._obj_to_pybullet_obj = None
        self._initial_pybullet_setup(with_gui=getattr(self._cf, "with_gui", False))

    def get_state_predicates(self):
        return {self.OnTable, self.OnBurner, self.OnBoard, self.InDish,
                self.IsChopped, self.IsCooked,
                self.Holding, self.HandEmpty}

    def get_action_predicates(self):
        return {self.Pick, self.PlaceOnBurner, self.PlaceOnBoard,
                self.PlaceInDish, self.PlaceOnTable, self.Chop, self.Cook}

    def _get_problems(self, num_problems, all_num_objs):
        problems = []
        for i in range(num_problems):
            j = i % len(all_num_objs)
            num_objs = all_num_objs[j]
            problems.append(self._create_problem(num_objs))
        return problems

    def _get_demo_problems(self, num):
        return self._get_problems(
            num, self._cf.kitchen_demo_num_objs)

    def get_test_problems(self):
        return self._get_problems(
            self._cf.kitchen_num_test_problems,
            self._cf.kitchen_test_num_objs)
    
    def get_random_action(self, state):
        objs = list(sorted(state.keys()))
        objs.remove(WORLD)
        act_pred = self._rng.randint(7)  # 7 action types: Pick, 4 Place variants, Chop, Cook
        if act_pred == 0:  # Pick
            obj = objs[self._rng.choice(len(objs))]
            return self._sample_ground_act(state, self.Pick, [obj])
        if act_pred == 1:  # PlaceOnBurner
            return self._sample_ground_act(state, self.PlaceOnBurner, [])
        if act_pred == 2:  # PlaceOnBoard
            return self._sample_ground_act(state, self.PlaceOnBoard, [])
        if act_pred == 3:  # PlaceInDish
            return self._sample_ground_act(state, self.PlaceInDish, [])
        if act_pred == 4:  # PlaceOnTable
            return self._sample_ground_act(state, self.PlaceOnTable, [])
        if act_pred == 5:  # Chop
            obj = objs[self._rng.choice(len(objs))]
            return self._sample_ground_act(state, self.Chop, [obj])
        if act_pred == 6:  # Cook
            obj = objs[self._rng.choice(len(objs))]
            return self._sample_ground_act(state, self.Cook, [obj])
        raise Exception("Can never reach here")

    def _create_problem(self, num_objs):
        state = {}
        obj_poses = []
        goals = []
        world_state = {}
        world_state["cur_holding"] = None
        world_state["cur_grip"] = [1.0, self.table_center_y, 0.75]  # In front of robot
        
        # Create ingredients
        for i in range(num_objs):
            obj = self.obj_type(f"ing{i}")
            pose = self._sample_initial_object_pose(obj_poses)
            obj_poses.append(pose)
            
            # Start raw
            obj_state = {
                "pose": pose, 
                "held": False,
                "chopped": False, 
                "cooked": False
            }
            state[obj] = obj_state
            
            # Goals: Processed and Served
            goals.append(self.InDish(obj))
            goals.append(self.IsChopped(obj))
            goals.append(self.IsCooked(obj))
            
        # Add flat features (simplified)
        world_state["flat"] = np.array([int(world_state["cur_holding"] is not None)])
        world_state["flat_names"] = np.array(["flat:holding_something"])
        state[WORLD] = world_state
        return state, structs.LiteralConjunction(goals)

    def _sample_initial_object_pose(self, existing_poses):
        existing_ys = [p[1] for p in existing_poses]
        for _ in range(1000):
            this_y = self._rng.uniform(self.table_lb, self.table_ub)
            # Ensure it's far from special areas
            if abs(this_y - self.burner_pos[1]) < 0.1 or \
               abs(this_y - self.board_pos[1]) < 0.1 or \
               abs(this_y - self.dish_pos[1]) < 0.1:
                continue

            if all(abs(this_y-other_y) > 1.1*self.obj_radius for other_y in existing_ys):
                # Place on table (table center x=1.65)
                return [self.table_center_x, this_y, self.table_height + self.obj_height/2]
        raise Exception("Could not sample initial pose")

    # --- Next State Logic ---

    def get_next_state(self, state, action):
        if action.predicate.var_types != [var.var_type for var in action.variables]:
            return self._copy_state(state)
        
        if action.predicate == self.Pick:
            return self._get_next_state_pick(state, action)
        if action.predicate in [self.PlaceOnBurner, self.PlaceOnBoard,
                                self.PlaceInDish, self.PlaceOnTable]:
            return self._get_next_state_place_discrete(state, action)
        if action.predicate == self.Chop:
            return self._get_next_state_chop(state, action)
        if action.predicate == self.Cook:
            return self._get_next_state_cook(state, action)
        
        raise Exception(f"Unexpected action: {action}")

    def _get_next_state_pick(self, state, action):
        assert action.predicate == self.Pick
        next_state = self._copy_state(state)
        obj = action.variables[0]
        
        # Cannot pick if already holding something
        if state[WORLD]["cur_holding"] is not None:
            return next_state
        
        # Execute pick
        next_state[WORLD]["cur_holding"] = obj
        next_state[obj]["held"] = True
        next_state[WORLD]["cur_grip"] = state[obj]["pose"]
        return next_state

    def _get_next_state_place_discrete(self, state, action):
        next_state = self._copy_state(state)
        
        # Cannot place if not holding
        if state[WORLD]["cur_holding"] is None:
            return next_state
        
        # Sampler determines pose implicitly, here we just set logic
        # Logic: If constraint holds (implicit), it succeeds.
        # But we need the POSE to update state.
        # Since logic is deterministic, we must invoke sampler to get pose
        # OR, since we are in simulation, we can re-sample or use a canonical pose.
        # BUT, standard PDDLGym `get_next_state` assumes effects are deterministic given inputs.
        # If input has no pose variable, how do we update key 'pose'?
        # We must deterministically choose a pose if one isn't provided.
        # Standard hack: place at center of area?
        # Or pass explicit pose via sampler -> action args?
        # Ideally action args should contain pose.
        # But we wanted to hide it.
        # If we hide it, we imply the pose doesn't matter for logic, only for rendering.
        # So we can pick a canonical pose in the center.
        
        obj = state[WORLD]["cur_holding"]
        target_center = None
        if action.predicate == self.PlaceOnBurner:
            target_center = self.burner_pos
        elif action.predicate == self.PlaceOnBoard:
            target_center = self.board_pos
        elif action.predicate == self.PlaceInDish:
            target_center = self.dish_pos
        elif action.predicate == self.PlaceOnTable:
            target_center = [self.table_center_x, self.table_center_y, self.table_height]
            
        # Execute place
        pose = list(target_center)
        pose[2] = self.table_height + self.obj_height/2
        if action.predicate in [self.PlaceOnBurner, self.PlaceOnBoard, self.PlaceInDish]:
            pose[2] += 0.01 # Slightly above
            
        next_state[WORLD]["cur_holding"] = None
        next_state[obj]["held"] = False
        next_state[obj]["pose"] = pose
        next_state[WORLD]["cur_grip"] = pose
        return next_state

    def _get_next_state_chop(self, state, action):
        assert action.predicate == self.Chop
        next_state = self._copy_state(state)
        obj = action.variables[0]
        
        # Can only chop if not holding (or holding tools, but we simplify)
        # Actually in this domain, we assume robot chops with hand/tool
        # But critically: Obj must be ON BOARD
        if not self._OnBoard_holds(state, obj):
            return next_state
            
        # Execute chop
        next_state[obj]["chopped"] = True
        return next_state

    def _get_next_state_cook(self, state, action):
        assert action.predicate == self.Cook
        next_state = self._copy_state(state)
        obj = action.variables[0]
        
        # Must be ON BURNER
        if not self._OnBurner_holds(state, obj):
            return next_state
            
        # Execute cook
        next_state[obj]["cooked"] = True
        return next_state

    # --- Predicate Logic ---

    def _in_area(self, pose, area_center):
        return np.linalg.norm(np.array(pose[:2]) - np.array(area_center[:2])) < self.area_radius

    def _OnTable_holds(self, state, obj):
        if state[obj]["held"]: return False
        # Not in any special area, but on table height
        pose = state[obj]["pose"]
        if self._in_area(pose, self.burner_pos): return False
        if self._in_area(pose, self.board_pos): return False
        if self._in_area(pose, self.dish_pos): return False
        return True # Simplified: assummed falling off table isn't an issue

    def _OnBurner_holds(self, state, obj):
        if state[obj]["held"]: return False
        return self._in_area(state[obj]["pose"], self.burner_pos)

    def _OnBoard_holds(self, state, obj):
        if state[obj]["held"]: return False
        return self._in_area(state[obj]["pose"], self.board_pos)

    def _InDish_holds(self, state, obj):
        if state[obj]["held"]: return False
        return self._in_area(state[obj]["pose"], self.dish_pos)

    def _IsChopped_holds(self, state, obj):
        return state[obj]["chopped"]

    def _IsCooked_holds(self, state, obj):
        return state[obj]["cooked"]

    def _Holding_holds(self, state, obj):
        return state[WORLD]["cur_holding"] == obj

    def _HandEmpty_holds(self, state):
        return state[WORLD]["cur_holding"] is None

    def _IsBurnerPose_holds(self, state, pose):
        return self._in_area(pose, self.burner_pos)
    
    def _IsBurnerPose_get_satisfying_args(self, state):
        return set()

    def _IsBoardPose_holds(self, state, pose):
        return self._in_area(pose, self.board_pos)

    def _IsBoardPose_get_satisfying_args(self, state):
        return set()

    def _IsDishPose_holds(self, state, pose):
        return self._in_area(pose, self.dish_pos)

    def _IsDishPose_get_satisfying_args(self, state):
        return set()

    def _IsTablePose_holds(self, state, pose):
        if self._in_area(pose, self.burner_pos): return False
        if self._in_area(pose, self.board_pos): return False
        if self._in_area(pose, self.dish_pos): return False
        # Simplified check
        return True

    def _IsTablePose_get_satisfying_args(self, state):
        return set()

    def _IsBurnerPose_sampler(self, rng, state):
        # Sample directly from burner area
        center = self.burner_pos
        pos = np.array(center) + rng.uniform(-self.area_radius/2, self.area_radius/2, size=3)
        pos[2] = center[2] + self.obj_height/2 + 0.01 # Place on top
        return (pos,)

    def _IsBoardPose_sampler(self, rng, state):
        center = self.board_pos
        pos = np.array(center) + rng.uniform(-self.area_radius/2, self.area_radius/2, size=3)
        pos[2] = center[2] + self.obj_height/2 + 0.01
        return (pos,)

    def _IsDishPose_sampler(self, rng, state):
        center = self.dish_pos
        pos = np.array(center) + rng.uniform(-self.area_radius/2, self.area_radius/2, size=3)
        pos[2] = center[2] + self.obj_height/2 + 0.01
        return (pos,)

    def _IsTablePose_sampler(self, rng, state):
        while True:
            rand_y = rng.uniform(self.table_lb, self.table_ub)
            pos = [self.table_center_x, rand_y, self.table_height + self.obj_height/2]
            # Check not in other areas
            if not self._in_area(pos, self.burner_pos) and \
               not self._in_area(pos, self.board_pos) and \
               not self._in_area(pos, self.dish_pos):
                 return (pos,)

    # --- Samplers ---
    
    def _pick_sampler(self, rng, state, *args):
        # Deterministic dummy
        return tuple()

    def _place_on_burner_sampler(self, rng, state, *args):
        center = self.burner_pos
        pos = np.array(center) + rng.uniform(-0.01, 0.01, size=3)
        pos[2] = center[2] + self.obj_height/2 + 0.01
        return tuple() # No args

    def _place_on_board_sampler(self, rng, state, *args):
        center = self.board_pos
        pos = np.array(center) + rng.uniform(-0.01, 0.01, size=3)
        pos[2] = center[2] + self.obj_height/2 + 0.01
        return tuple()

    def _place_in_dish_sampler(self, rng, state, *args):
        center = self.dish_pos
        pos = np.array(center) + rng.uniform(-0.01, 0.01, size=3)
        pos[2] = center[2] + self.obj_height/2 + 0.01
        return tuple()

    def _place_on_table_sampler(self, rng, state, *args):
        rand_y = rng.uniform(self.table_lb, self.table_ub)
        pos = [self.table_center_x, rand_y, self.table_height + self.obj_height/2]
        # Avoid areas
        while True:
             if not self._in_area(pos, self.burner_pos) and \
                not self._in_area(pos, self.board_pos) and \
                not self._in_area(pos, self.dish_pos):
                 break
             pos[1] = rng.uniform(self.table_lb, self.table_ub)
        return tuple()

    def _chop_sampler(self, rng, state, *args):
        return tuple()

    def _cook_sampler(self, rng, state, *args):
        return tuple()

    # --- PyBullet ---

    def _get_camera_params(self):
        camera_distance = 2.2
        yaw = 90
        pitch = -15
        # Target between robot (0.8) and table (1.65)
        camera_target = [1.2, 0.6, 0.4]
        return camera_distance, yaw, pitch, camera_target

    def _reset_camera(self):
        camera_distance, yaw, pitch, camera_target = self._get_camera_params()
        p.resetDebugVisualizerCamera(
            camera_distance, yaw, pitch,
            camera_target, physicsClientId=self._physics_client_id)

    def _initial_pybullet_setup(self, with_gui=False):
        self._obj_to_pybullet_obj = {}
        if with_gui:
            self._physics_client_id = p.connect(p.GUI)
        else:
            self._physics_client_id = p.connect(p.DIRECT)
        if with_gui:
            self._reset_camera()
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0,
                                   physicsClientId=self._physics_client_id)
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setAdditionalSearchPath("envs/assets/")
        p.loadURDF(get_asset_path("urdf/plane.urdf"), [0, 0, -1], useFixedBase=True, physicsClientId=self._physics_client_id)
        
        # Table: basePosition (0,0,0) so URDF origin (1.65,0.75,0.2) places table in front of robot
        table_urdf = get_asset_path("urdf/table2.urdf")
        self._table_id = p.loadURDF(table_urdf, basePosition=(0, 0, 0), useFixedBase=True, physicsClientId=self._physics_client_id)
        
        # Robot: place in front of table (Blocks layout)
        self._fetch_id = p.loadURDF(
            get_asset_path("urdf/robots/fetch.urdf"), useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._fetch_id, [0.8, 0.75, 0.0], [0., 0., 0., 1.],
            physicsClientId=self._physics_client_id)
        # Fetch arm joints for IK (so robot pose matches cur_grip in visualization)
        joint_names = [p.getJointInfo(
            self._fetch_id, i,
            physicsClientId=self._physics_client_id)[1].decode("utf-8")
            for i in range(p.getNumJoints(
                self._fetch_id, physicsClientId=self._physics_client_id))]
        self._ee_id = joint_names.index("gripper_axis")
        self._ee_orn_down = p.getQuaternionFromEuler((0, np.pi/2, -np.pi))
        self._arm_joints = get_kinematic_chain(
            self._fetch_id, self._ee_id,
            physics_client_id=self._physics_client_id)
        self._left_finger_id = joint_names.index("l_gripper_finger_joint")
        self._right_finger_id = joint_names.index("r_gripper_finger_joint")
        self._arm_joints_with_fingers = list(self._arm_joints) + [self._left_finger_id, self._right_finger_id]
        self._arm_joints = list(self._arm_joints)

        # Visual Areas on table (x=1.65 from URDF, y=0.4/0.6/0.8): Burner, Chopping Board, Dish
        self._create_visual_box(self.burner_pos, [0.1, 0.1, 0.01], [0.8, 0.2, 0.2, 1])
        self._create_visual_box(self.board_pos, [0.1, 0.1, 0.01], [0.6, 0.4, 0.2, 1])
        self._create_visual_box(self.dish_pos, [0.1, 0.1, 0.05], [0.2, 0.8, 0.2, 1])
        # Fixed labels for the three areas (short names so they don't overlap)
        if with_gui:
            self._area_label_ids = []
            for center, name in [
                (self.burner_pos, "Burner"),
                (self.board_pos, "Board"),
                (self.dish_pos, "Dish"),
            ]:
                text_pos = [center[0], center[1], center[2] + 0.08]
                tid = p.addUserDebugText(
                    name, text_pos,
                    textColorRGB=[0.0, 0.0, 0.0], textSize=1.4,
                    physicsClientId=self._physics_client_id)
                self._area_label_ids.append(tid)
        else:
            self._area_label_ids = []

    def _create_visual_box(self, center, extents, color):
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=np.array(extents)/2, rgbaColor=color, physicsClientId=self._physics_client_id)
        p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=center, physicsClientId=self._physics_client_id)

    def _reset_pybullet_from_state(self, state, held_obj_constraint=None):
        """Update PyBullet scene to match symbolic state for visualization."""
        # 1) Set robot arm to cur_grip so the robot appears to hold/move objects
        if hasattr(self, "_arm_joints") and self._arm_joints:
            cur_grip = state[WORLD].get("cur_grip")
            if cur_grip is not None:
                target_position = np.add(list(cur_grip), [0.0, 0.0, 0.075])
            else:
                target_position = [1.2, 0.75, 0.5]
            hint_joint_values = [0.48, -1.58, 1.88, 0.83, 1.37, -0.23, -0.33, 0.035, 0.035]
            for joint_idx, joint_val in zip(self._arm_joints_with_fingers, hint_joint_values):
                p.resetJointState(
                    self._fetch_id, joint_idx, joint_val,
                    physicsClientId=self._physics_client_id)
            joint_values = inverse_kinematics(
                self._fetch_id, self._ee_id, target_position,
                self._ee_orn_down, self._arm_joints,
                physics_client_id=self._physics_client_id)
            for joint_idx, joint_val in zip(self._arm_joints, joint_values):
                p.resetJointState(
                    self._fetch_id, joint_idx, joint_val,
                    physicsClientId=self._physics_client_id)
            # Close gripper if holding, else open
            finger_val = 0.025 if state[WORLD].get("cur_holding") else 0.035
            for fid in [self._left_finger_id, self._right_finger_id]:
                p.resetJointState(
                    self._fetch_id, fid, finger_val,
                    physicsClientId=self._physics_client_id)
        # 2) Rebuild objects if state set changed
        objs = [o for o in state if o != WORLD]
        if set(self._obj_to_pybullet_obj) | {WORLD} != set(state):
            self._rebuild_pybullet_objects(state)
        held = state[WORLD].get("cur_holding")
        for obj in objs:
            if obj not in self._obj_to_pybullet_obj:
                continue
            py_obj = self._obj_to_pybullet_obj[obj]
            if state[obj].get("held") and held == obj:
                pos = state[WORLD].get("cur_grip", state[obj]["pose"])
            else:
                pos = state[obj]["pose"]
            orn = [0, 0, 0, 1]
            p.resetBasePositionAndOrientation(
                py_obj, pos, orn,
                physicsClientId=self._physics_client_id)

    def _rebuild_pybullet_objects(self, state):
        for obj_id in list(self._obj_to_pybullet_obj.values()):
            p.removeBody(obj_id, physicsClientId=self._physics_client_id)
        self._obj_to_pybullet_obj = {}
        self._obj_to_label = {}
        objs = sorted([o for o in state if o != WORLD], key=lambda x: x.name)
        r, h = self.obj_radius, self.obj_height / 2
        for i, obj in enumerate(objs):
            vis = self.INGREDIENT_VISUALS[i % len(self.INGREDIENT_VISUALS)]
            color = vis["color"]
            shape = vis["shape"]
            if shape == "sphere":
                col_id = p.createCollisionShape(
                    p.GEOM_SPHERE, radius=r, physicsClientId=self._physics_client_id)
                vis_id = p.createVisualShape(
                    p.GEOM_SPHERE, radius=r, rgbaColor=color,
                    physicsClientId=self._physics_client_id)
            elif shape == "cylinder":
                col_id = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=r, height=2*h,
                    physicsClientId=self._physics_client_id)
                vis_id = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=r, length=2*h, rgbaColor=color,
                    physicsClientId=self._physics_client_id)
            else:
                half = [r * 0.9, r * 0.9, h]
                col_id = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half, physicsClientId=self._physics_client_id)
                vis_id = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=half, rgbaColor=color,
                    physicsClientId=self._physics_client_id)
            pid = p.createMultiBody(
                baseMass=0.05, baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id, basePosition=[0, 0, 0],
                physicsClientId=self._physics_client_id)
            self._obj_to_pybullet_obj[obj] = pid