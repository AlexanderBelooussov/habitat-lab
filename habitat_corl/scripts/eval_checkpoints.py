import os

import numpy as np
import torch
from tqdm import trange

import habitat
from habitat_corl.common.utils import set_seed, eval_actor, train_eval_split
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.common.replay_buffer import get_input_dims
from habitat_corl.common.shortest_path_dataset import register_new_sensors, \
    calc_mean_std


def algo_to_cfgname(algo):
    if algo in ["sac_n_d", "td3_bc_d"]:
        return algo[:-2].upper()
    return algo.upper()


def eval_checkpoint_dt(checkpoint_path, task="SingleGoal"):
    from habitat.config.default import get_config as cfg_env
    from habitat_corl.dt import DecisionTransformer, eval_rollout, strin_to_tuple
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
    dt_config = getattr(config.RL, algo_to_cfgname("dt"))
    task_config = config.TASK_CONFIG
    seed = config.SEED
    set_seed(config.SEED,
             deterministic_torch=getattr(dt_config, "deterministic_torch",
                                         False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean_std = calc_mean_std(task_config,
                             used_inputs=config.MODEL.used_inputs)

    cp = torch.load(checkpoint_path, map_location=device)
    state_dict = cp["model_state"]
    state_mean = cp["state_mean"]
    state_std = cp["state_std"]
    eval_env = wrap_env(
        habitat.Env(config=task_config),
        model_config=config.MODEL,
        state_mean=state_mean,
        state_std=state_std,
        reward_scale=dt_config.reward_scale,
        ignore_stop=dt_config.ignore_stop,
        continuous=False
    )
    # model & optimizer & scheduler setup
    state_dim = get_input_dims(config)
    action_dim = eval_env.action_space.n
    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embedding_dim=dt_config.embedding_dim,
        seq_len=dt_config.seq_len,
        episode_len=dt_config.episode_len,
        num_layers=dt_config.num_layers,
        num_heads=dt_config.num_heads,
        attention_dropout=dt_config.attention_dropout,
        residual_dropout=dt_config.residual_dropout,
        embedding_dropout=dt_config.embedding_dropout,
        max_action=dt_config.max_action,
    ).to(device)
    model.load_state_dict(state_dict)
    if isinstance(dt_config.target_returns, str):
        target_returns = strin_to_tuple(dt_config.target_returns,
                                        float)
    else:
        target_returns = dt_config.target_returns
    for target_return in target_returns:
        for i in trange(dt_config.eval_episodes, desc="Evaluation",
                        leave=False):
            eval_rollout(
                model=model,
                env=eval_env,
                target_return=target_return * dt_config.reward_scale,
                device=device,
                video=True,
                video_dir=config.VIDEO_DIR,
                video_prefix=f"dt/{task}/seed{seed}/{checkpoint_name}/{target_return}/dt",
                eval_iteration=i,
                success_distance=config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
                ignore_stop=dt_config.ignore_stop,
            )


def eval_checkpoint(checkpoint_path, make_trainer, algorithm, continuous,
                    task="SingleGoal"):
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
    if 'DEPTH_SENSOR' not in config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS:
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS += ['DEPTH_SENSOR']
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS += ['RGB_SENSOR']
    config.freeze()

    algo_config = getattr(config.RL, algo_to_cfgname(algorithm))
    task_config = config.TASK_CONFIG
    seed = config.SEED
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
        model_config=config.MODEL,
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
            video_prefix=f"{algorithm}/{task}/seed{seed}/{checkpoint_name}/{algorithm}",
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
    algorithm = "bc"
    scene = "small"
    checkpoint_nr = 999999
    seed = 2
    task = "PointNavDepth"
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
    elif algorithm == "bc" and task == "SingleGoal":
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
    elif algorithm == "bc" and "PointNav" in task:
        from habitat_corl.any_percent_bc import init_trainer
        eval_checkpoint(
            f"checkpoints/{task}/BC/checkpoint_{checkpoint_nr}.pt",
            init_trainer,
            algorithm,
            continuous=False,
            task=task,
        )
    elif algorithm == "dt" and task == "PointNav":
        eval_checkpoint_dt(
            f"checkpoints/PointNav/DT/dt_checkpoint.pt",
            task=task,
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
