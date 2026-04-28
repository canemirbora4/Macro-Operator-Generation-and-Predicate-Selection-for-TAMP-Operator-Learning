"""Contains global, immutable settings.

Anything that varies between runs should be a command-line arg (args.py).
"""

class Config:
    """Config class
    """
    env = None
    start_seed = None
    num_seeds = None
    collect_data = None

    data_dir = "data/"

    cover_num_blocks = 2
    cover_num_targets = 2
    cover_block_widths = [0.1, 0.07]
    cover_target_widths = [0.05, 0.03]
    cover_num_test_problems = 20

    blocks_demo_num_objs = [3, 4, 5]
    blocks_test_num_objs = [6]
    blocks_num_test_problems = 20

    painting_demo_num_objs = [3, 4, 5]
    painting_test_num_objs = [6, 7, 8, 9, 10]
    painting_num_test_problems = 20
    
    kitchen_demo_num_objs = [3, 4, 5]
    kitchen_test_num_objs = [6, 7, 8]
    kitchen_num_test_problems = 20

    # Warehouse domain - simple Pick→Place, many objects (macro showcase)
    warehouse_demo_num_pkgs = [3, 4, 5]  # Small for demos
    warehouse_test_num_pkgs = [8, 10, 12, 15]  # Larger: 16-30 steps without macros
    warehouse_num_test_problems = 20

    backtracking_num_samples_per_step = 10
    effect_prob_threshold = 0.001
    num_demos = 20
    
    # Defaults
    num_negatives = None
    max_test_episode_length = None
    approach_timeout = None

    builp_max_unique_lifted_effects = 10
    builp_learn_probabilities = True
    builp_learn_empty_effects = False
    builp_max_preconditions_per_effect = 100000
    builp_max_search_iters = 100
    builp_max_rule_size = float("inf")
    builp_true_pos_weight = 10
    builp_false_pos_weight = 1
    builp_var_count_weight = 1e-1
    builp_rule_size_weight = 1e-2
    builp_referenced_objects_only = True


def create_config(args):
    """Create config class
    """
    Config.env = args.env
    Config.start_seed = args.start_seed
    Config.num_seeds = args.num_seeds
    Config.collect_data = args.collect_data

    Config.num_negatives = {
        "cover": 100,
        "painting": 2500,
        "blocks": 100,
        "warehouse": 200,
        "kitchen": 1000,
    }[args.env]

    Config.max_test_episode_length = {
        "cover": 5,
        "painting": 50,
        "blocks": 50,
        "warehouse": 30,
        "kitchen": 50,
    }[args.env]

    Config.approach_timeout = {
        "cover": 1,
        "painting": 10,
        "blocks": 10,
        "warehouse": 10,
        "kitchen": 10,
    }[args.env]

    Config.with_gui = getattr(args, "gui", False)
    return Config
