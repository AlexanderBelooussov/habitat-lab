# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import trange

import habitat
from habitat_baselines.config.default import get_config
from habitat_corl.common.utils import restructure_results, train_eval_split, \
    eval_actor, wandb_init, set_seed, get_goal
from habitat_corl.common.wrappers import wrap_env
from habitat_corl.replay_buffer import get_input_dims, ReplayBuffer
from habitat_corl.shortest_path_dataset import register_new_sensors, \
    calc_mean_std, batch_generator

TensorBatch = List[torch.Tensor]


class Actor(nn.Module):
    def __init__(self, action_dim: int):
        super(Actor, self).__init__()
        self.n_actions = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        action = np.random.choice(self.n_actions)
        return action


def train(config):
    algo_config = config.RL.RANDOM
    task_config = config.TASK_CONFIG
    set_seed(config.SEED)
    wandb_init(config)

    with wrap_env(
        habitat.Env(config=task_config),
        used_inputs=config.MODEL.used_inputs,
        continuous=algo_config.continuous,
        ignore_stop=algo_config.ignore_stop,
        turn_angle=task_config.SIMULATOR.TURN_ANGLE,
    ) as env:
        action_dim = env.action_space.n

        train_episodes, eval_episodes = train_eval_split(
            env=env,
            config=config,
            n_eval_episodes=algo_config.eval_episodes,
            single_goal=algo_config.single_goal,
        )

        # Set seeds
        seed = config.SEED
        set_seed(seed, env)

        actor = Actor(action_dim)

        print("---------------------------------------")
        print(f"Random Agent, Scene: {task_config.DATASET.CONTENT_SCENES}"
              f", Seed: {seed}, ignore_stop: {algo_config.ignore_stop}, "
              f"single_goal: {algo_config.single_goal}")
        print("---------------------------------------")

        evaluations = []
        eval_scores = eval_actor(
            env,
            actor,
            device="cpu",
            episodes=eval_episodes,
            seed=config.SEED,
            used_inputs=config.MODEL.used_inputs,
            video=True,
            video_dir=config.VIDEO_DIR,
            video_prefix="td3_bc/td3_bc",
            success_distance=task_config.TASK.SUCCESS_DISTANCE,
            ignore_stop=algo_config.ignore_stop,
        )
        evaluations.append(eval_scores)
        print("---------------------------------------")
        print(
            f"Evaluation over {algo_config.eval_episodes} episodes: "
        )
        for key in eval_scores:
            print(f"{key}: {np.mean(eval_scores[key])}")
        print("---------------------------------------")
        for key in eval_scores:
            wandb.log(
                {
                    f"eval/{key}_mean": np.mean(eval_scores[key]),
                    f"eval/{key}_std": np.std(eval_scores[key])
                },
                step=0,
            )

        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="habitat_corl/configs/random_pointnav.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    register_new_sensors(config.TASK_CONFIG)
    train(config)


if __name__ == "__main__":
    main()
