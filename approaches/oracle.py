"""Oracle, only used for data collection.
"""

import abc
from ndr.ndrs import NDR, NDRSet, NOISE_OUTCOME
from pddlgym.structs import Anti
from utils import ndrs_to_operators
from approaches import BaseApproach


class Oracle(BaseApproach):
    """Oracle implementation.
    """
    def _get_operators(self, data):
        del data  # unused
        return get_gt_ops(self._cf, self._state_preds, self._action_preds)


def get_gt_ops(config, state_predicates, action_predicates):
    """Get the ground truth operators.
    """
    if config.env == "painting":
        cls = PaintingOperators(state_predicates, action_predicates)
    elif config.env == "cover":
        cls = CoverOperators(state_predicates, action_predicates)
    elif config.env == "blocks":
        cls = BlocksOperators(state_predicates, action_predicates)
    elif config.env == "warehouse":
        cls = WarehouseOperators(state_predicates, action_predicates)
    elif config.env == "kitchen":
        cls = KitchenOperators(state_predicates, action_predicates)
    else:
        raise Exception(f"Unrecognized env: {config.env}")
    return cls.get_operators()


class GroundTruthOperators:
    """Generic interface for ground truth operators.
    """
    def __init__(self, state_predicates, action_predicates):
        self._preds = {}
        for pred in state_predicates:
            self._preds[pred.name] = pred
        for pred in action_predicates:
            self._preds[pred.name] = pred

    @abc.abstractmethod
    def get_operators(self):
        """Return the set of operators.
        """
        raise NotImplementedError("Override me!")


class CoverOperators(GroundTruthOperators):
    """Ground truth operators for the cover environment.
    """
    def get_operators(self):
        IsBlock = self._preds["IsBlock"]
        IsTarget = self._preds["IsTarget"]
        Covers = self._preds["Covers"]
        HandEmpty = self._preds["HandEmpty"]
        Holding = self._preds["Holding"]
        Pick = self._preds["Pick"]
        Place = self._preds["Place"]

        all_ndrs = {}

        # Pick
        pick_ndrs = []
        action = Pick("?block", "?pose")
        preconditions = [IsBlock("?block"), HandEmpty()]
        effects = [{Holding("?block"), Anti(HandEmpty())}, {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        pick_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, pick_ndrs)

        # Place
        place_ndrs = []
        action = Place("?target", "?pose")
        preconditions = [IsTarget("?target"), IsBlock("?block"),
                         Holding("?block")]
        # NOTE: the Covers effect is optimistic!
        effects = [{Anti(Holding("?block")), HandEmpty(),
                    Covers("?block", "?target")}, {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        place_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, place_ndrs)

        return ndrs_to_operators(all_ndrs)


class BlocksOperators(GroundTruthOperators):
    """Ground truth operators for the blocks environment.
    """
    def get_operators(self):
        On = self._preds["On"]
        OnTable = self._preds["OnTable"]
        Holding = self._preds["Holding"]
        Clear = self._preds["Clear"]
        HandEmpty = self._preds["HandEmpty"]
        Pick = self._preds["Pick"]
        PutOnTable = self._preds["PutOnTable"]
        Stack = self._preds["Stack"]

        all_ndrs = {}

        pick_ndrs = []
        action = Pick("?block")

        # Pick up from table
        preconditions = [Clear("?block"), OnTable("?block"), HandEmpty()]
        effects = [{Anti(Clear("?block")), Anti(OnTable("?block")),
                    Anti(HandEmpty()), Holding("?block")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        pick_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        # Pick up from block
        preconditions = [HandEmpty(), On("?block", "?otherblock"),
                         Clear("?block")]
        effects = [{Anti(HandEmpty()), Anti(On("?block", "?otherblock")),
                    Anti(Clear("?block")), Holding("?block"),
                    Clear("?otherblock")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        pick_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        all_ndrs[action] = pick_ndrs

        put_on_table_ndrs = []
        action = PutOnTable("?pose")

        # Put on table
        preconditions = [Holding("?block")]
        effects = [{Anti(Holding("?block")), Clear("?block"), OnTable("?block"),
                    HandEmpty()},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        put_on_table_ndrs.append(NDR(action, preconditions,
                                     effect_probs, effects))

        all_ndrs[action] = put_on_table_ndrs

        stack_ndrs = []
        action = Stack("?otherblock")

        # Stack on other block
        preconditions = [Holding("?block"), Clear("?otherblock")]
        effects = [{Anti(Holding("?block")), Anti(Clear("?otherblock")),
                    HandEmpty(), Clear("?block"),
                    On("?block", "?otherblock")}, {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        stack_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        all_ndrs[action] = stack_ndrs

        return ndrs_to_operators(all_ndrs)


class PaintingOperators(GroundTruthOperators):
    """Ground truth operators for the painting environment.
    """
    def get_operators(self):
        OnTable = self._preds["OnTable"]
        Holding = self._preds["Holding"]
        HoldingSide = self._preds["HoldingSide"]
        HoldingTop = self._preds["HoldingTop"]
        InShelf = self._preds["InShelf"]
        InBox = self._preds["InBox"]
        HandEmpty = self._preds["HandEmpty"]
        IsDry = self._preds["IsDry"]
        IsWet = self._preds["IsWet"]
        IsClean = self._preds["IsClean"]
        IsDirty = self._preds["IsDirty"]
        IsShelfColor = self._preds["IsShelfColor"]
        IsBoxColor = self._preds["IsBoxColor"]
        IsBlank = self._preds["IsBlank"]
        Pick = self._preds["Pick"]
        Place = self._preds["Place"]
        Dry = self._preds["Dry"]
        Wash = self._preds["Wash"]
        Paint = self._preds["Paint"]

        all_ndrs = {}

        # Pick
        pick_ndrs = []

        ## Side grasp
        action = Pick("?obj", "?base", "?grip")
        preconditions = [OnTable("?obj"), HandEmpty()]
        effects = [{Anti(OnTable("?obj")), Anti(HandEmpty()),
                    HoldingSide("?obj"), Holding("?obj")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        pick_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        ## Top grasp
        action = Pick("?obj", "?base", "?grip")
        preconditions = [OnTable("?obj"), HandEmpty()]
        effects = [{Anti(OnTable("?obj")), Anti(HandEmpty()),
                    HoldingTop("?obj"), Holding("?obj")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        pick_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        all_ndrs[action] = NDRSet(action, pick_ndrs)

        # Place
        place_ndrs = []

        ## Place into shelf while HoldingSide
        action = Place("?base", "?grip")
        preconditions = [HoldingSide("?obj"), Holding("?obj")]
        effects = [{Anti(HoldingSide("?obj")), Anti(Holding("?obj")),
                    HandEmpty(), InShelf("?obj")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        place_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        ## Place into box while HoldingTop
        action = Place("?base", "?grip")
        preconditions = [HoldingTop("?obj"), Holding("?obj")]
        effects = [{Anti(HoldingTop("?obj")), Anti(Holding("?obj")),
                    HandEmpty(), InBox("?obj")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        place_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        all_ndrs[action] = NDRSet(action, place_ndrs)

        # Wash
        wash_ndrs = []

        action = Wash("?obj", "?water")
        preconditions = [Holding("?obj"), IsDirty("?obj"), IsDry("?obj")]
        effects = [{Anti(IsDirty("?obj")), Anti(IsDry("?obj")),
                    IsClean("?obj"), IsWet("?obj")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        wash_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        all_ndrs[action] = NDRSet(action, wash_ndrs)

        # Dry
        dry_ndrs = []

        action = Dry("?obj", "?heat")
        preconditions = [Holding("?obj"), IsWet("?obj")]
        effects = [{Anti(IsWet("?obj")), IsDry("?obj")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        dry_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        all_ndrs[action] = NDRSet(action, dry_ndrs)

        # Paint
        paint_ndrs = []

        action = Paint("?color")
        preconditions = [Holding("?obj"), IsDry("?obj"), IsClean("?obj"),
                         IsBlank("?obj")]
        effects = [{IsShelfColor("?obj"), Anti(IsBlank("?obj"))},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        paint_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        action = Paint("?color")
        preconditions = [Holding("?obj"), IsDry("?obj"), IsClean("?obj"),
                         IsBlank("?obj")]
        effects = [{IsBoxColor("?obj"), Anti(IsBlank("?obj"))},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        paint_ndrs.append(NDR(action, preconditions, effect_probs, effects))

        all_ndrs[action] = NDRSet(action, paint_ndrs)

        return ndrs_to_operators(all_ndrs)


class WarehouseOperators(GroundTruthOperators):
    """Ground truth operators for the warehouse environment.
    
    Simple Pick → Place domain for macro showcase.
    """
    def get_operators(self):
        HandEmpty = self._preds["HandEmpty"]
        Holding = self._preds["Holding"]
        OnTable = self._preds["OnTable"]
        OnShelf = self._preds["OnShelf"]
        Pick = self._preds["Pick"]
        Place = self._preds["Place"]

        all_ndrs = {}

        # Pick: Pick up a package from table
        pick_ndrs = []
        action = Pick("?pkg", "?pose")
        preconditions = [OnTable("?pkg"), HandEmpty()]
        effects = [{Anti(OnTable("?pkg")), Anti(HandEmpty()), Holding("?pkg")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        pick_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, pick_ndrs)

        # Place: Place held package onto shelf
        place_ndrs = []
        action = Place("?pose")
        preconditions = [Holding("?pkg")]
        effects = [{Anti(Holding("?pkg")), HandEmpty(), OnShelf("?pkg")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        place_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, place_ndrs)

        return ndrs_to_operators(all_ndrs)


class KitchenOperators(GroundTruthOperators):
    """Ground truth operators for the kitchen environment.
    
    Complex multi-step cooking domain.
    Sequence: Pick → Place(board) → Chop → Pick → Place(burner) → Cook → Pick → Place(dish)
    """
    def get_operators(self):
        HandEmpty = self._preds["HandEmpty"]
        Holding = self._preds["Holding"]
        OnTable = self._preds["OnTable"]
        OnBurner = self._preds["OnBurner"]
        OnBoard = self._preds["OnBoard"]
        InDish = self._preds["InDish"]
        IsChopped = self._preds["IsChopped"]
        IsCooked = self._preds["IsCooked"]
        Pick = self._preds["Pick"]
        PlaceOnBurner = self._preds["PlaceOnBurner"]
        PlaceOnBoard = self._preds["PlaceOnBoard"]
        PlaceInDish = self._preds["PlaceInDish"]
        PlaceOnTable = self._preds["PlaceOnTable"]
        Chop = self._preds["Chop"]
        Cook = self._preds["Cook"]

        all_ndrs = {}

        # Pick from table
        pick_ndrs = []
        action = Pick("?obj")
        preconditions = [OnTable("?obj"), HandEmpty()]
        effects = [{Anti(OnTable("?obj")), Anti(HandEmpty()), Holding("?obj")},
                   {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        pick_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        
        # Pick from board
        preconditions2 = [OnBoard("?obj"), HandEmpty()]
        effects2 = [{Anti(OnBoard("?obj")), Anti(HandEmpty()), Holding("?obj")},
                   {NOISE_OUTCOME}]
        pick_ndrs.append(NDR(action, preconditions2, effect_probs, effects2))
        
        # Pick from burner
        preconditions3 = [OnBurner("?obj"), HandEmpty()]
        effects3 = [{Anti(OnBurner("?obj")), Anti(HandEmpty()), Holding("?obj")},
                   {NOISE_OUTCOME}]
        pick_ndrs.append(NDR(action, preconditions3, effect_probs, effects3))
        all_ndrs[action] = NDRSet(action, pick_ndrs)

        # PlaceOnBoard (for chopping)
        place_board_ndrs = []
        action = PlaceOnBoard()
        preconditions = [Holding("?obj")]
        effects = [{Anti(Holding("?obj")), HandEmpty(), OnBoard("?obj")},
                   {NOISE_OUTCOME}]
        place_board_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, place_board_ndrs)

        # PlaceOnBurner (for cooking)
        place_burner_ndrs = []
        action = PlaceOnBurner()
        preconditions = [Holding("?obj")]
        effects = [{Anti(Holding("?obj")), HandEmpty(), OnBurner("?obj")},
                   {NOISE_OUTCOME}]
        place_burner_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, place_burner_ndrs)

        # PlaceInDish (final)
        place_dish_ndrs = []
        action = PlaceInDish()
        preconditions = [Holding("?obj")]
        effects = [{Anti(Holding("?obj")), HandEmpty(), InDish("?obj")},
                   {NOISE_OUTCOME}]
        place_dish_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, place_dish_ndrs)

        # PlaceOnTable
        place_table_ndrs = []
        action = PlaceOnTable()
        preconditions = [Holding("?obj")]
        effects = [{Anti(Holding("?obj")), HandEmpty(), OnTable("?obj")},
                   {NOISE_OUTCOME}]
        place_table_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, place_table_ndrs)

        # Chop (requires OnBoard)
        chop_ndrs = []
        action = Chop("?obj")
        preconditions = [OnBoard("?obj")]
        effects = [{IsChopped("?obj")}, {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        chop_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, chop_ndrs)

        # Cook (requires OnBurner)
        cook_ndrs = []
        action = Cook("?obj")
        preconditions = [OnBurner("?obj")]
        effects = [{IsCooked("?obj")}, {NOISE_OUTCOME}]
        effect_probs = [1.0, 0.0]
        cook_ndrs.append(NDR(action, preconditions, effect_probs, effects))
        all_ndrs[action] = NDRSet(action, cook_ndrs)

        return ndrs_to_operators(all_ndrs)
