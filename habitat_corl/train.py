import argparse

import torch

import habitat_corl.sac_n
import habitat_corl.dt
import habitat_corl.any_percent_bc
import habitat_corl.td3_bc
from habitat_baselines.config.default import get_config
from habitat_corl.shortest_path_dataset import register_new_sensors

from tqdm import tqdm
from functools import partialmethod

scene_dict = {
    "medium": "17DRP5sb8fy",
    "debug": "17DRP5sb8fy",
    "small": "Pm6F8kyY3z2",
    "large": "XcA2TqTSSAj",
    "long_hallway": "Vt2qJdWjCF2",
    "xl": "uNb9QFRL6hY",
}
dataset_dict = {
    "medium": "data/sp_datasets/datasets_medium_no_depth.hdf5",
    "debug": "data/sp_datasets/debug_datasets_medium_no_depth.hdf5",
    "small": "data/sp_datasets/datasets_small_no_depth.hdf5",
    "large": "data/sp_datasets/datasets_large_no_depth.hdf5",
    "long_hallway": "data/sp_datasets/datasets_long_hallway_no_depth.hdf5",
    "xl": "data/sp_datasets/datasets_xl_no_depth.hdf5",
}
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda:0":
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sacn",
        choices=["sacn", "dt", "bc", "td3_bc", "sac_n", "td3bc", "bc_10",
                 "bc10"],
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
        default=100,
        help="Number of episodes to evaluate on",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed to use"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="medium",
        choices=["medium", "large", "small", "xl", "debug"],
        help="Scene to use",
    )
    parser.add_argument(
        "--web_dataset",
        action="store_true",
        help="Use web dataset",
    )

    args = parser.parse_args()

    algorithm = args.algorithm.replace("_", "")
    task = args.task
    ignore_stop = args.ignore_stop
    n_eval_episodes = args.n_eval_episodes
    seed = args.seed
    scene = args.scene

    if algorithm == "sacn" and task != "objectnav":
        config = "habitat_corl/configs/sacn_pointnav.yaml"
        config = get_config(config, [])
        config.defrost()
        algo_config = config.RL.SAC_N

    elif algorithm == "dt":
        config = "habitat_corl/configs/dt_pointnav.yaml"
        config = get_config(config, [])
        config.defrost()
        algo_config = config.RL.DT

    elif algorithm == "bc" or algorithm == "bc10":
        config = "habitat_corl/configs/bc_pointnav.yaml"
        config = get_config(config, [])
        config.defrost()
        algo_config = config.RL.BC
        if algorithm == "bc10":
            algo_config.FRAC = 0.1
            config.NAME += "-10"
    elif algorithm == "td3bc":
        config = "habitat_corl/configs/td3_bc_pointnav.yaml"
        config = get_config(config, [])
        config.defrost()
        algo_config = config.RL.TD3_BC
    else:
        raise ValueError("Invalid algorithm/task combination")

    algo_config.ignore_stop = ignore_stop
    algo_config.eval_episodes = n_eval_episodes
    config.SEED = seed
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_dict[scene]]

    if ignore_stop:
        config.NAME += "-ignore_stop"
    if task == "singlegoal":
        config.GROUP = f"SingleGoal_{scene}"
        if algorithm == "dt":
            algo_config.target_returns = "(1.0, 10.0)"
        algo_config.single_goal = True
        config.MODEL.used_inputs = ["position", "heading_vec"]
    elif task == "pointnav_depth":
        config.GROUP = f"PointNavDepth_{scene}"
        algo_config.single_goal = False
        config.MODEL.used_inputs = ["depth", "pointgoal_with_gps_compass",
                                    "heading_vec"]
    elif task == "pointnav":
        config.GROUP = f"PointNav_{scene}"
        algo_config.single_goal = False
        config.MODEL.used_inputs = ["position", "heading_vec", "goal_position"]

    config.TASK_CONFIG.DATASET.SP_DATASET_PATH = dataset_dict[scene]
    if args.web_dataset:
        config.TASK_CONFIG.DATASET.WEB_DATASET_PATH = f"data/web_datasets/web_dataset_{scene}.hdf5"

    if scene == "debug":
        algo_config.eval_episodes = 10

    config.freeze()
    register_new_sensors(config.TASK_CONFIG)

    if algorithm == "sacn":
        habitat_corl.sac_n.train(config)
    elif algorithm == "dt":
        habitat_corl.dt.train(config)
    elif algorithm == "bc" or algorithm == "bc10":
        habitat_corl.any_percent_bc.train(config)
    elif algorithm == "td3bc":
        habitat_corl.td3_bc.train(config)
    else:
        raise ValueError("Invalid algorithm")



if __name__ == "__main__":
    main()
