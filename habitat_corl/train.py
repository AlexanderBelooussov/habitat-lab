import argparse

import habitat_corl.sac_n
from habitat_baselines.config.default import get_config
from habitat_corl.shortest_path_dataset import register_position_sensor

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
        base_config.SEED = seed
        base_config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_dict[scene]]
        if task == "singlegoal":
            base_config.RL.SAC_N.single_goal = True
            base_config.RL.SAC_N.used_inpts = ["position", "heading"]
        elif task == "pointnav_depth":
            base_config.RL.SAC_N.single_goal = False
            base_config.RL.SAC_N.used_inpts = ["depth", "pointgoal_with_gps_compass"]
        elif task == "pointnav":
            base_config.RL.SAC_N.single_goal = False
            base_config.RL.SAC_N.used_inpts = ["position", "heading", "goal_position"]

        base_config.freeze()

        register_position_sensor(base_config.TASK_CONFIG)
        habitat_corl.sac_n.train(base_config)

if __name__ == "__main__":
    main()
