import argparse

import torch

import habitat_corl.sac_n_discrete
import habitat_corl.sac_n
import habitat_corl.dt
import habitat_corl.any_percent_bc
import habitat_corl.td3_bc
import habitat_corl.td3_bc_discrete
import habitat_corl.iql
import habitat_corl.edac
import habitat_corl.random_agent
import habitat_corl.cql
import habitat_corl.lb_sac
from habitat_baselines.config.default import get_config
from habitat_corl.shortest_path_dataset import register_new_sensors

from tqdm import tqdm
from functools import partialmethod

scene_dict = {
    "medium": "17DRP5sb8fy",
    "debug": "17DRP5sb8fy",
    "small": "Pm6F8kyY3z2",
    # "large": "XcA2TqTSSAj",
    "large": "ac26ZMwG7aT",
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sacn",
        choices=["sacn", "dt", "bc", "td3_bc", "sac_n", "td3bc", "bc_10",
                 "bc10", "iql", "edac", "sacnd", "sacn_d", "td3bcd",
                 "td3bc_d", "random", "cql", "cql_d", "cqld", "lb_sac", "lbsac"],
        help="Algorithm to use",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="singlegoal",
        choices=["pointnav", "objectnav", "singlegoal", "pointnavdepth"],
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
    parser.add_argument(
        "--web_dataset_only",
        action="store_true",
        help="Use web dataset, without shortest path dataset",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="Comment to add to the run name",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="",
        help="Additional change to group name",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=-1,
        help="Number of layers for the network, -1 for default",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    algorithm = args.algorithm.replace("_", "")
    task = args.task
    ignore_stop = args.ignore_stop
    n_eval_episodes = args.n_eval_episodes
    seed = args.seed
    scene = args.scene
    comment = args.comment
    group = args.group
    n_layers = args.n_layers
    noise = args.noise

    if device == "cuda:0" and scene != "debug":
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    if task == "pointnavdepth":
        base_config = "configs/tasks/pointnav_mp3d_depth.yaml"
    else:
        base_config = "configs/tasks/pointnav_mp3d_medium.yaml"

    if algorithm == "sacn" and task != "objectnav":
        config = "habitat_corl/configs/sacn_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.SAC_N

    elif algorithm == "dt":
        config = "habitat_corl/configs/dt_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.DT

    elif algorithm == "bc" or algorithm == "bc10":
        config = "habitat_corl/configs/bc_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.BC
        if algorithm == "bc10":
            algo_config.FRAC = 0.1
            config.NAME += "-10"
    elif algorithm == "td3bc":
        config = "habitat_corl/configs/td3_bc_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.TD3_BC
    elif algorithm == "td3bcd":
        config = "habitat_corl/configs/td3_bc_d_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.TD3_BC
        algo_config.continuous = False
    elif algorithm == "iql":
        config = "habitat_corl/configs/iql_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.IQL
    elif algorithm == "edac":
        config = "habitat_corl/configs/edac_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.EDAC
    elif algorithm == "sacnd":
        config = "habitat_corl/configs/sacnd_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.SAC_N
        algo_config.continuous = False
    elif algorithm == "random":
        config = "habitat_corl/configs/random_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.RANDOM
    elif algorithm == "cql":
        config = "habitat_corl/configs/cql_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.CQL
    # elif algorithm == "cqld":
    #     config = "habitat_corl/configs/cql_d_pointnav.yaml"
    #     config = get_config(config, [])
    #     config.defrost()
    #     algo_config = config.RL.CQL
    elif algorithm == "lbsac":
        config = "habitat_corl/configs/lbsac_pointnav.yaml"
        config = get_config(config, ["BASE_TASK_CONFIG_PATH", base_config])
        config.defrost()
        algo_config = config.RL.LB_SAC
    else:
        raise ValueError("Invalid algorithm/task combination")

    algo_config.ignore_stop = ignore_stop
    algo_config.eval_episodes = n_eval_episodes
    config.SEED = seed
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_dict[scene]]
    config.VIDEO_DIR += f"/{scene}"

    if ignore_stop:
        config.NAME += "-ignore_stop"

    if task == "singlegoal":
        config.GROUP = f"SingleGoal_{scene}"
        if algorithm == "dt":
            algo_config.target_returns = "(1.0, 10.0)"
        algo_config.single_goal = True
        config.MODEL.used_inputs = ["position", "heading_vec"]
        algo_config.ignore_stop = True
    elif task == "pointnavdepth":
        config.GROUP = f"PointNavDepth_{scene}"
        algo_config.single_goal = False
        config.MODEL.used_inputs = ["depth", "pointgoal_with_gps_compass",
                                    "heading_vec"]
        config.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav_mp3d_depth.yaml"
    elif task == "pointnav":
        config.GROUP = f"PointNav_{scene}"
        algo_config.single_goal = False
        config.MODEL.used_inputs = ["position", "heading_vec", "goal_position"]
    else:
        raise ValueError("Invalid task")

    if n_layers > 0:
        algo_config.n_layers = n_layers
        algo_config.n_actor_layers = n_layers
        algo_config.n_critic_layers = n_layers

    config.TASK_CONFIG.DATASET.SP_DATASET_PATH = dataset_dict[scene]
    if args.web_dataset:
        config.TASK_CONFIG.DATASET.WEB_DATASET_PATH = f"data/web_datasets/web_dataset_{scene}.hdf5"
    if args.web_dataset_only:
        config.TASK_CONFIG.DATASET.SP_DATASET_PATH = f"data/web_datasets/web_dataset_{scene}.hdf5"
    if scene == "debug":
        algo_config.eval_episodes = 100
    else:
        algo_config.eval_episodes = 100

    if comment != "":
        config.NAME += f"-{comment}"
    if group != "":
        config.GROUP += f"-{group}"
    config.NAME += f"-seed{config.SEED}"

    config.algorithm = algorithm
    config.CHECKPOINT_FOLDER += f"/{config.GROUP}/{config.NAME}/{config.SEED}"

    config.noise = noise

    config.freeze()
    register_new_sensors(config.TASK_CONFIG)

    if algorithm == "sacn":
        habitat_corl.sac_n.train(config)
    elif algorithm == "sacnd":
        habitat_corl.sac_n_discrete.train(config)
    elif algorithm == "dt":
        habitat_corl.dt.train(config)
    elif algorithm == "bc" or algorithm == "bc10":
        habitat_corl.any_percent_bc.train(config)
    elif algorithm == "td3bc":
        habitat_corl.td3_bc.train(config)
    elif algorithm == "td3bcd":
        habitat_corl.td3_bc_discrete.train(config)
    elif algorithm == "iql":
        habitat_corl.iql.train(config)
    elif algorithm == "edac":
        habitat_corl.edac.train(config)
    elif algorithm == "random":
        habitat_corl.random_agent.train(config)
    elif algorithm == "cql":
        habitat_corl.cql.train(config)
    # elif algorithm == "cqld":
    #     habitat_corl.cql_discrete.train(config)
    elif algorithm == "lbsac":
        habitat_corl.lb_sac.train(config)
    else:
        raise ValueError("Invalid algorithm")



if __name__ == "__main__":
    main()
