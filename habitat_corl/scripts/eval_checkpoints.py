import os

import numpy as np
import torch

import habitat
from habitat_corl.common.utils import set_seed, eval_actor, train_eval_split
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.replay_buffer import get_input_dims
from habitat_corl.shortest_path_dataset import register_new_sensors, \
    calc_mean_std


def algo_to_cfgname(algo):
    if algo in ["sac_n_d", "td3_bc_d"]:
        return algo[:-2].upper()
    return algo.upper()


def eval_checkpoint(checkpoint_path, make_trainer, algorithm, continuous):
    from habitat.config.default import get_config as cfg_env
    # get config in same directory
    dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path).split(".")[0]
    config_path = os.path.join(dir, "config.yaml")
    config = cfg_env(config_path)
    register_new_sensors(config.TASK_CONFIG)

    # add depth, rgb and top-down map to inputs
    config.defrost()
    config.TASK_CONFIG.TASK.MEASUREMENTS += ["TOP_DOWN_MAP"]
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS += ['DEPTH_SENSOR',
                                                     'RGB_SENSOR']
    config.freeze()

    algo_config = getattr(config.RL, algo_to_cfgname(algorithm))
    task_config = config.TASK_CONFIG
    set_seed(config.SEED,
             deterministic_torch=getattr(algo_config, "deterministic_torch",
                                         False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean_std = calc_mean_std(task_config,
                             used_inputs=config.MODEL.used_inputs)

    # data, evaluation, env setup
    with wrap_env(
        habitat.Env(config=task_config),
        state_mean=mean_std["used"][0],
        state_std=mean_std["used"][1],
        used_inputs=config.MODEL.used_inputs,
        continuous=continuous,
        ignore_stop=algo_config.ignore_stop,
        turn_angle=task_config.SIMULATOR.TURN_ANGLE,
    ) as env:
        state_dim = get_input_dims(config)
        if continuous:
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])
            action_space_shape = env.action_space.shape
        else:
            action_dim = env.action_space.n
            max_action = 1.0
            action_space_shape = (1,)

        train_episodes, eval_episodes = train_eval_split(
            env=env,
            config=config,
            n_eval_episodes=algo_config.eval_episodes,
            single_goal=algo_config.single_goal,
        )

        trainer = make_trainer(
            algo_config=algo_config,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            max_action=max_action,
            action_space_shape=action_space_shape,
        )

        trainer.load_state_dict(
            torch.load(checkpoint_path, map_location=device))
        actor = trainer.actor
        eval_scores = eval_actor(
            env,
            actor,
            device=device,
            episodes=eval_episodes,
            seed=config.SEED,
            used_inputs=config.MODEL.used_inputs,
            video=True,
            # video=False,
            video_dir=config.VIDEO_DIR,
            video_prefix=f"{algorithm}/{checkpoint_name}/{algorithm}",
            success_distance=task_config.TASK.SUCCESS_DISTANCE,
            ignore_stop=algo_config.ignore_stop,
        )
        print("---------------------------------------")
        print(
            f"Evaluation over {algo_config.eval_episodes} episodes: "
        )
        for key in eval_scores:
            print(f"{key}: {np.mean(eval_scores[key])}")
        print("---------------------------------------")


def main():
    algorithm = "sac_n"
    scene = "medium"
    checkpoint_nr = 1000000
    seed = 0
    if algorithm == "sac_n":
        from habitat_corl.sac_n import init_trainer
        eval_checkpoint(
            f"checkpoints/SingleGoal_{scene}-defaults/SAC-N-seed{seed}/{seed}/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=True,
        )
    elif algorithm == "td3_bc":
        from habitat_corl.td3_bc import init_trainer
        eval_checkpoint(
            f"checkpoints/SingleGoal_{scene}-defaults/TD3_BC-seed{seed}/{seed}/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=True,
        )
    elif algorithm == "sac_n_d":
        from habitat_corl.sac_n_discrete import init_trainer
        eval_checkpoint(
            f"checkpoints/SingleGoal_{scene}-defaults/SAC-N-D-seed{seed}/{seed}/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=False,
        )
    elif algorithm == "td3_bc_d":
        from habitat_corl.td3_bc_discrete import init_trainer
        eval_checkpoint(
            f"checkpoints/SingleGoal_{scene}-defaults/TD3_BC_Discrete-seed{seed}/{seed}/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=False,
        )
    elif algorithm == "cql":
        from habitat_corl.cql import init_trainer
        eval_checkpoint(
            f"checkpoints/SingleGoal_{scene}-defaults/CQL-seed{seed}/{seed}/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=True,
        )
    elif algorithm == "bc":
        from habitat_corl.any_percent_bc import init_trainer
        eval_checkpoint(
            f"checkpoints/SingleGoal_{scene}-defaults/BC-seed{seed}/{seed}/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=False,
        )
    elif algorithm == "bc10":
        from habitat_corl.any_percent_bc import init_trainer
        eval_checkpoint(
            f"checkpoints/SingleGoal_{scene}-defaults/BC-10-seed{seed}/{seed}/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=False,
        )
    elif algorithm == "iql":
        from habitat_corl.iql import init_trainer
        eval_checkpoint(
            f"checkpoints/SingleGoal_{scene}-defaults/IQL-seed{seed}/{seed}/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=True,
        )
    elif algorithm == "lb_sac":
        from habitat_corl.lb_sac import init_trainer
        eval_checkpoint(
            f"checkpoints/SingleGoal_{scene}-defaults/LB_SAC-seed{seed}/{seed}/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=True,
        )


if __name__ == "__main__":
    main()
