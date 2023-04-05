import argparse

import habitat_corl.sac_n
import habitat_corl.dt
from habitat_baselines.config.default import get_config
from habitat_corl.shortest_path_dataset import register_new_sensors

scene_dict = {
    "medium": "17DRP5sb8fy",
    "small": "Pm6F8kyY3z2",
    "large": "XcA2TqTSSAj",
    "long_hallway": "Vt2qJdWjCF2",
    "xl": "uNb9QFRL6hY",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sacn",
        choices=["sacn", "dt", "bc"],
        help="Algorithm to use",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="singlegoal",
        choices=["pointnav", "objectnav", "singlegoal", "pointnav_depth"],
        help="Task to use",
    )
    parser.add_argument(
        "--ignore_stop",
        action="store_true",
        help="Ignore stop action in the environment",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=20,
        help="Number of episodes to evaluate on",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed to use"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="medium",
        choices=["medium", "large", "small", "xl"],
        help="Scene to use",
    )

    args = parser.parse_args()

    algorithm = args.algorithm
    task = args.task
    ignore_stop = args.ignore_stop
    n_eval_episodes = args.n_eval_episodes
    seed = args.seed
    scene = args.scene

    if algorithm == "sacn" and task != "objectnav":
        base_config = "habitat_corl/configs/sacn_pointnav.yaml"
        base_config = get_config(base_config, [])
        base_config.defrost()
        base_config.RL.SAC_N.ignore_stop = True
        base_config.RL.SAC_N.eval_episodes = n_eval_episodes
        if task == "singlegoal":
            base_config.GROUP = "SingleGoal"
            base_config.RL.SAC_N.single_goal = True
            base_config.RL.SAC_N.used_inpts = ["position", "heading_vec"]
        elif task == "pointnav_depth":
            base_config.GROUP = "PointNavDepth"
            base_config.RL.SAC_N.single_goal = False
            base_config.RL.SAC_N.used_inpts = ["depth", "pointgoal_with_gps_compass", "heading_vec"]
        elif task == "pointnav":
            base_config.GROUP = "PointNav"
            base_config.RL.SAC_N.single_goal = False
            base_config.RL.SAC_N.used_inpts = ["position", "heading_vec", "goal_position"]

        base_config.freeze()

        register_new_sensors(base_config.TASK_CONFIG)
        habitat_corl.sac_n.train(base_config)

    elif algorithm == "dt":
        config = "habitat_corl/configs/dt_pointnav.yaml"
        config = get_config(config, [])
        config.defrost()
        config.RL.DT.ignore_stop = ignore_stop
        config.RL.DT.eval_episodes = n_eval_episodes
        config.SEED = seed
        config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_dict[scene]]
        if ignore_stop:
            config.NAME += "-ignore_stop"
        if task == "singlegoal":
            config.GROUP = "SingleGoal"
            config.RL.DT.target_returns = "(1.0, 10.0)"
            config.RL.DT.single_goal = True
            config.RL.DT.used_inputs = ["position", "heading_vec"]
        elif task == "pointnav_depth":
            config.GROUP = "PointNavDepth"
            config.RL.DT.single_goal = False
            config.RL.DT.used_inputs = ["depth", "pointgoal_with_gps_compass", "heading_vec"]
        elif task == "pointnav":
            config.GROUP = "PointNav"
            config.RL.DT.single_goal = False
            config.RL.DT.used_inputs = ["position", "heading_vec", "goal_position"]

        config.freeze()
        register_new_sensors(config.TASK_CONFIG)
        habitat_corl.dt.train(config)

if __name__ == "__main__":
    main()
